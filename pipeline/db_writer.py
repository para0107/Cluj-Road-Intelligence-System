"""
pipeline/db_writer.py
---------------------
Stage 8 of the road damage detection inference pipeline.

Responsibilities:
  - Accept frames from Stage 7 (deduplicated.json["frames"])
  - Write every retained (non-duplicate) detection to the PostgreSQL 15 +
    PostGIS database, table: detections
  - Upsert logic: if a detection at the same GPS point (within 2 m) and
    same class already exists in the database, update the existing record:
      * Increment detection_count
      * Update last_detection_date
      * Recalculate deterioration_rate = (new_severity - old_severity) / days_since
      * Update surrounding_density (count of detections within 50 m)
  - New detections are inserted with detection_count=1
  - Computes priority_score = severity_weight × road_weight × infra_weight
    × log(detection_count + 1) after every upsert
  - Skips detections without GPS (cannot be stored in PostGIS)
  - Returns a DbWriteResult summary

Database schema (created by scripts/setup_db.py):
  CREATE EXTENSION IF NOT EXISTS postgis;
  CREATE TABLE detections (
      id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      geom                  GEOMETRY(POINT, 4326),
      damage_type           TEXT,
      confidence            FLOAT,
      surface_area          FLOAT,
      edge_sharpness        FLOAT,
      interior_contrast     FLOAT,
      mask_compactness      FLOAT,
      depth_estimate        FLOAT,
      depth_confidence      FLOAT,
      severity              TEXT,
      severity_confidence   FLOAT,
      lighting_condition    TEXT,
      shadow_geometry_score FLOAT,
      street_name           VARCHAR,
      road_importance       SMALLINT,
      infra_proximity       FLOAT,
      weather               JSONB,
      first_detection_date  TIMESTAMP,
      last_detection_date   TIMESTAMP,
      detection_count       INT DEFAULT 1,
      deterioration_rate    FLOAT DEFAULT 0.0,
      surrounding_density   INT DEFAULT 0,
      priority_score        FLOAT
  );
  CREATE INDEX detections_geom_idx ON detections USING GIST(geom);

Connection:
  Configured via DbWriterConfig or environment variables:
    RIDS_DB_HOST, RIDS_DB_PORT, RIDS_DB_NAME, RIDS_DB_USER, RIDS_DB_PASSWORD

Usage (module):
    from pipeline.db_writer import DbWriter, DbWriterConfig
    result = DbWriter(DbWriterConfig()).run(deduplicated_frames)

Usage (CLI):
    python pipeline/db_writer.py
        --input   data/validation_nrdd_2024/deduplicated/deduplicated.json
        [--host localhost] [--port 5432] [--dbname rids]
        [--user postgres] [--password postgres]
        [--dry_run]
        [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority score weights
# Mirrors Eq. (priority) in the thesis.
# severity_weight: maps S1–S5 to numeric weight
# road_weight:     maps road_importance 1–3 to weight
# infra_weight:    distance-decayed weight from infra_proximity
# ---------------------------------------------------------------------------
_SEVERITY_WEIGHT: Dict[str, float] = {
    "S1": 0.10,
    "S2": 0.25,
    "S3": 0.50,
    "S4": 0.75,
    "S5": 1.00,
}

_ROAD_WEIGHT: Dict[int, float] = {
    1: 0.5,   # residential
    2: 0.75,  # secondary
    3: 1.0,   # primary / arterial
}

# Infrastructure proximity weight: 1.0 if within 50 m, decays to 0.1 beyond 500 m
_INFRA_MAX_WEIGHT   = 1.0
_INFRA_MIN_WEIGHT   = 0.1
_INFRA_CLOSE_M      = 50.0
_INFRA_FAR_M        = 500.0

# Neighbour search radius for surrounding_density (metres)
_DENSITY_RADIUS_M   = 50.0

# Upsert proximity threshold — reuse existing record if within this distance
_UPSERT_RADIUS_M    = 2.0


def _infra_weight(infra_proximity: Optional[float]) -> float:
    if infra_proximity is None:
        return _INFRA_MIN_WEIGHT
    if infra_proximity <= _INFRA_CLOSE_M:
        return _INFRA_MAX_WEIGHT
    if infra_proximity >= _INFRA_FAR_M:
        return _INFRA_MIN_WEIGHT
    # Linear decay between CLOSE and FAR
    frac = (infra_proximity - _INFRA_CLOSE_M) / (_INFRA_FAR_M - _INFRA_CLOSE_M)
    return _INFRA_MAX_WEIGHT - frac * (_INFRA_MAX_WEIGHT - _INFRA_MIN_WEIGHT)


def _priority_score(
    severity_level:  str,
    road_importance: Optional[int],
    infra_proximity: Optional[float],
    detection_count: int,
) -> float:
    """
    priority_score = w_severity × w_road × w_infra × log(detection_count + 1)
    """
    w_sev  = _SEVERITY_WEIGHT.get(severity_level, 0.25)
    w_road = _ROAD_WEIGHT.get(road_importance or 1, 0.5)
    w_infra = _infra_weight(infra_proximity)
    return round(w_sev * w_road * w_infra * math.log(detection_count + 1), 6)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DbWriterConfig:
    """
    Database connection parameters for Stage 8.
    Values are read first from the dataclass fields, then from environment
    variables (RIDS_DB_*) as fallback, then from the defaults below.

    dry_run:
        If True, build all SQL but do not execute or commit.
        Useful for testing the pipeline without a live database.
    """
    host:     str  = "localhost"
    port:     int  = 5432
    dbname:   str  = "rids"
    user:     str  = "postgres"
    password: str  = "postgres"
    dry_run:  bool = False

    def __post_init__(self) -> None:
        # Allow environment variable overrides
        self.host     = os.environ.get("RIDS_DB_HOST",     self.host)
        self.port     = int(os.environ.get("RIDS_DB_PORT", self.port))
        self.dbname   = os.environ.get("RIDS_DB_NAME",     self.dbname)
        self.user     = os.environ.get("RIDS_DB_USER",     self.user)
        self.password = os.environ.get("RIDS_DB_PASSWORD", self.password)


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------
@dataclass
class DbWriteResult:
    n_frames_processed: int
    n_inserted:         int
    n_updated:          int
    n_skipped_no_gps:   int
    n_skipped_duplicate: int
    n_errors:           int
    dry_run:            bool
    elapsed_s:          float

    def log_summary(self) -> None:
        mode = "DRY RUN — no data was committed" if self.dry_run else "COMMITTED"
        logger.info("=== DB Write complete (%s) ===", mode)
        logger.info("  Frames processed       : %d", self.n_frames_processed)
        logger.info("  Rows inserted (new)    : %d", self.n_inserted)
        logger.info("  Rows updated (upsert)  : %d", self.n_updated)
        logger.info("  Skipped (no GPS)       : %d", self.n_skipped_no_gps)
        logger.info("  Skipped (duplicate)    : %d", self.n_skipped_duplicate)
        logger.info("  Errors                 : %d", self.n_errors)
        logger.info("  Elapsed                : %.1f s", self.elapsed_s)


# ---------------------------------------------------------------------------
# DbWriter class
# ---------------------------------------------------------------------------

class DbWriter:
    """
    Stage 8 — PostgreSQL/PostGIS database write with upsert logic.

    psycopg2 is the only runtime dependency. Install with:
        pip install psycopg2-binary

    The class does NOT import psycopg2 at module level so that
    the rest of the pipeline can be imported in dry_run mode or
    on machines without a PostgreSQL installation.
    """

    def __init__(self, cfg: DbWriterConfig) -> None:
        self.cfg = cfg
        self._conn = None
        self._cur  = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        frames: List[dict],
        output_dir: Optional[str] = None,
    ) -> DbWriteResult:
        """
        Write all retained detections from deduplicated frames to the DB.

        Parameters
        ----------
        frames     : list of frame dicts from deduplicated.json["frames"]
        output_dir : if given, saves db_write_summary.json there

        Returns
        -------
        DbWriteResult
        """
        import time
        t_start = time.perf_counter()

        n_inserted        = 0
        n_updated         = 0
        n_skipped_no_gps  = 0
        n_skipped_dup     = 0
        n_errors          = 0

        if not self.cfg.dry_run:
            self._connect()

        try:
            for frame in frames:
                lat = frame.get("latitude")
                lon = frame.get("longitude")
                gps_ok = lat is not None and lon is not None

                ts_raw  = frame.get("timestamp_s", 0.0)
                try:
                    detection_dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    detection_dt = datetime.now(tz=timezone.utc)

                lighting = frame.get("lighting", "unknown")

                for box in frame.get("boxes", []):
                    # Dedup metadata — skip duplicates
                    dedup = box.get("dedup", {})
                    if dedup.get("is_duplicate", False):
                        n_skipped_dup += 1
                        continue

                    if not gps_ok:
                        n_skipped_no_gps += 1
                        continue

                    # Extract all fields
                    try:
                        row = self._box_to_row(
                            box, lat, lon, detection_dt, lighting
                        )
                    except Exception as exc:
                        logger.error("Failed to parse box: %s — %s", box, exc)
                        n_errors += 1
                        continue

                    if self.cfg.dry_run:
                        logger.debug("DRY RUN — would upsert: %s at (%.5f, %.5f)",
                                     row["damage_type"], lat, lon)
                        n_inserted += 1
                        continue

                    try:
                        action = self._upsert(row, lat, lon)
                        if action == "insert":
                            n_inserted += 1
                        else:
                            n_updated += 1
                    except Exception as exc:
                        logger.error(
                            "DB upsert error for %s at (%.5f, %.5f): %s",
                            row["damage_type"], lat, lon, exc,
                        )
                        n_errors += 1
                        if self._conn:
                            self._conn.rollback()

            if not self.cfg.dry_run and self._conn:
                self._conn.commit()
                logger.info("Transaction committed.")

        finally:
            self._disconnect()

        elapsed = time.perf_counter() - t_start

        result = DbWriteResult(
            n_frames_processed  = len(frames),
            n_inserted          = n_inserted,
            n_updated           = n_updated,
            n_skipped_no_gps    = n_skipped_no_gps,
            n_skipped_duplicate = n_skipped_dup,
            n_errors            = n_errors,
            dry_run             = self.cfg.dry_run,
            elapsed_s           = round(elapsed, 2),
        )
        result.log_summary()

        if output_dir:
            self._save_summary(result, output_dir)

        return result

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _box_to_row(
        self,
        box:          dict,
        lat:          float,
        lon:          float,
        detection_dt: datetime,
        lighting:     str,
    ) -> dict:
        """
        Build a flat dict ready for SQL insertion from an enriched/deduped box.
        Raises ValueError if required fields are missing.
        """
        enrichment = box.get("enrichment", {}) or {}
        geometry   = box.get("geometry",   {}) or {}
        depth      = box.get("depth",      {}) or {}
        severity   = box.get("severity",   {}) or {}
        weather_d  = enrichment.get("weather") or {}

        severity_level = severity.get("severity_level", "S1")
        road_importance = enrichment.get("road_importance")
        infra_proximity = enrichment.get("infra_proximity")
        detection_count = 1  # Will be incremented on upsert

        return {
            "geom_lat":           lat,
            "geom_lon":           lon,
            "damage_type":        box.get("class_name", "unknown"),
            "confidence":         box.get("confidence"),
            "surface_area":       geometry.get("surface_area_px"),
            "edge_sharpness":     geometry.get("edge_sharpness"),
            "interior_contrast":  geometry.get("interior_contrast"),
            "mask_compactness":   geometry.get("mask_compactness"),
            "depth_estimate":     depth.get("depth_norm"),
            "depth_confidence":   depth.get("depth_confidence"),
            "severity":           severity_level,
            "severity_confidence":severity.get("severity_confidence"),
            "lighting_condition": lighting,
            "shadow_geometry_score": None,          # Placeholder; Stage 1 output
            "street_name":        enrichment.get("street_name"),
            "road_importance":    road_importance,
            "infra_proximity":    infra_proximity,
            "weather":            json.dumps(weather_d) if weather_d else None,
            "first_detection_date": detection_dt,
            "last_detection_date":  detection_dt,
            "detection_count":    detection_count,
            "deterioration_rate": 0.0,
            "surrounding_density": 0,
            "priority_score":     _priority_score(
                severity_level, road_importance, infra_proximity, detection_count
            ),
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def _upsert(self, row: dict, lat: float, lon: float) -> str:
        """
        Upsert one detection row.

        1. Search for an existing detection of the same class within
           UPSERT_RADIUS_M metres using PostGIS ST_DWithin.
        2. If found: UPDATE detection_count, last_detection_date,
           deterioration_rate, surrounding_density, priority_score.
        3. If not found: INSERT a new row.

        Returns "insert" or "update".
        """
        # Search for existing record
        self._cur.execute(
            """
            SELECT id, severity, first_detection_date, detection_count
            FROM   detections
            WHERE  damage_type = %s
            AND    ST_DWithin(
                       geom::geography,
                       ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                       %s
                   )
            ORDER  BY ST_Distance(
                          geom::geography,
                          ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                      ) ASC
            LIMIT  1;
            """,
            (
                row["damage_type"],
                lon, lat, _UPSERT_RADIUS_M,
                lon, lat,
            ),
        )
        existing = self._cur.fetchone()

        if existing is None:
            # INSERT
            self._cur.execute(
                """
                INSERT INTO detections (
                    geom, damage_type, confidence,
                    surface_area, edge_sharpness, interior_contrast, mask_compactness,
                    depth_estimate, depth_confidence,
                    severity, severity_confidence,
                    lighting_condition, shadow_geometry_score,
                    street_name, road_importance, infra_proximity,
                    weather,
                    first_detection_date, last_detection_date,
                    detection_count, deterioration_rate,
                    surrounding_density, priority_score
                ) VALUES (
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s::jsonb,
                    %s, %s,
                    %s, %s,
                    %s, %s
                );
                """,
                (
                    row["geom_lon"], row["geom_lat"],
                    row["damage_type"], row["confidence"],
                    row["surface_area"], row["edge_sharpness"],
                    row["interior_contrast"], row["mask_compactness"],
                    row["depth_estimate"], row["depth_confidence"],
                    row["severity"], row["severity_confidence"],
                    row["lighting_condition"], row["shadow_geometry_score"],
                    row["street_name"], row["road_importance"], row["infra_proximity"],
                    row["weather"],
                    row["first_detection_date"], row["last_detection_date"],
                    row["detection_count"], row["deterioration_rate"],
                    row["surrounding_density"], row["priority_score"],
                ),
            )

            # Update surrounding_density for this new point
            self._update_surrounding_density(lon, lat, row["damage_type"])
            return "insert"

        else:
            # UPDATE
            existing_id          = existing[0]
            old_severity_str     = existing[1]
            first_dt             = existing[2]
            old_count            = existing[3]
            new_count            = old_count + 1

            # Deterioration rate: change in severity score per day
            old_sev_score = _SEVERITY_WEIGHT.get(old_severity_str, 0.0)
            new_sev_score = _SEVERITY_WEIGHT.get(row["severity"], 0.0)
            now           = row["last_detection_date"]
            if first_dt and isinstance(first_dt, datetime):
                days_since = max((now - first_dt).total_seconds() / 86400.0, 1.0)
            else:
                days_since = 1.0
            deterioration_rate = round((new_sev_score - old_sev_score) / days_since, 6)

            new_priority = _priority_score(
                row["severity"],
                row["road_importance"],
                row["infra_proximity"],
                new_count,
            )

            self._cur.execute(
                """
                UPDATE detections
                SET    last_detection_date = %s,
                       detection_count     = %s,
                       deterioration_rate  = %s,
                       priority_score      = %s,
                       severity            = %s,
                       severity_confidence = %s,
                       confidence          = %s
                WHERE  id = %s;
                """,
                (
                    now,
                    new_count,
                    deterioration_rate,
                    new_priority,
                    row["severity"],
                    row["severity_confidence"],
                    row["confidence"],
                    existing_id,
                ),
            )

            self._update_surrounding_density(lon, lat, row["damage_type"])
            return "update"

    def _update_surrounding_density(
        self, lon: float, lat: float, damage_type: str
    ) -> None:
        """
        Update surrounding_density for all records within DENSITY_RADIUS_M
        of the newly inserted/updated detection.
        """
        self._cur.execute(
            """
            UPDATE detections d
            SET    surrounding_density = (
                       SELECT COUNT(*) - 1
                       FROM   detections d2
                       WHERE  ST_DWithin(
                                  d.geom::geography,
                                  d2.geom::geography,
                                  %s
                              )
                   )
            WHERE  ST_DWithin(
                       d.geom::geography,
                       ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                       %s
                   );
            """,
            (_DENSITY_RADIUS_M, lon, lat, _DENSITY_RADIUS_M),
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            import psycopg2
        except ImportError:
            logger.error(
                "psycopg2 is required for Stage 8. "
                "Install with: pip install psycopg2-binary"
            )
            raise

        logger.info(
            "Connecting to PostgreSQL: %s@%s:%d/%s",
            self.cfg.user, self.cfg.host, self.cfg.port, self.cfg.dbname,
        )
        self._conn = psycopg2.connect(
            host     = self.cfg.host,
            port     = self.cfg.port,
            dbname   = self.cfg.dbname,
            user     = self.cfg.user,
            password = self.cfg.password,
        )
        self._conn.autocommit = False
        self._cur = self._conn.cursor()
        logger.info("Connected.")

    def _disconnect(self) -> None:
        if self._cur:
            try:
                self._cur.close()
            except Exception:
                pass
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._cur  = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_summary(result: DbWriteResult, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "db_write_summary.json"
        payload = {
            "n_frames_processed":   result.n_frames_processed,
            "n_inserted":           result.n_inserted,
            "n_updated":            result.n_updated,
            "n_skipped_no_gps":     result.n_skipped_no_gps,
            "n_skipped_duplicate":  result.n_skipped_duplicate,
            "n_errors":             result.n_errors,
            "dry_run":              result.dry_run,
            "elapsed_s":            result.elapsed_s,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("DB write summary saved: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 8 — PostgreSQL/PostGIS database write."
    )
    parser.add_argument("--input",    required=True,
                        help="deduplicated.json from Stage 7")
    parser.add_argument("--output",   default=None,
                        help="Directory to save db_write_summary.json (optional)")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=5432)
    parser.add_argument("--dbname",   default="rids")
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Parse and validate without writing to DB")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input not found: %s", in_path)
        raise SystemExit(1)

    with in_path.open("r", encoding="utf-8") as f:
        dedup_data = json.load(f)

    frames = dedup_data.get("frames", [])
    logger.info("Loaded %d frames from %s", len(frames), in_path)

    cfg = DbWriterConfig(
        host     = args.host,
        port     = args.port,
        dbname   = args.dbname,
        user     = args.user,
        password = args.password,
        dry_run  = args.dry_run,
    )
    DbWriter(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()
