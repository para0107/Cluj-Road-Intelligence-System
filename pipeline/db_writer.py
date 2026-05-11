"""
pipeline/db_writer.py
---------------------
Stage 7 of the RIDS inference pipeline (renumbered from Stage 8 after
enrichment removal).

Responsibilities
----------------
- Accept deduplicated frame dicts from Stage 6 (Deduplicator)
- Write every retained, GPS-equipped detection to the PostgreSQL/PostGIS DB
- Upsert: UPDATE if same class exists within DEDUP_CLUSTER_RADIUS_M metres,
  INSERT otherwise
- Recompute surrounding_density for all rows within
  SURROUNDING_DENSITY_RADIUS_M metres of each written point
- Compute priority_score = w_severity × log(detection_count + 1)
  (road_importance and infra_proximity removed with the enrichment stage)

What was removed from the previous version
-------------------------------------------
- street_name        — no longer enriched or stored
- road_importance    — no longer used in priority formula
- infra_proximity_m  — no longer used in priority formula
- weather            — no longer fetched
- nearest_infra_type — was never in the INSERT; removed from row dict
- _ROAD_WEIGHT       — no longer needed
- _infra_weight()    — no longer needed

Priority formula (simplified)
------------------------------
    priority_score = w_severity × log(detection_count + 1)

    w_severity: S1=0.10  S2=0.25  S3=0.50  S4=0.75  S5=1.00

    Rationale: without road classification data the only signal that
    meaningfully differentiates urgency is damage severity and persistence
    (detection_count). This is documented as a known limitation in the thesis.

Environment variables (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
    POSTGRES_USER, POSTGRES_PASSWORD
    DEDUP_CLUSTER_RADIUS_M       — upsert proximity threshold (default 2.0)
    SURROUNDING_DENSITY_RADIUS_M — density search radius (default 50.0)

Usage (CLI):
    python pipeline/db_writer.py
        --input   data/.../deduplicated/deduplicated.json
        [--output data/.../db_write/]
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection — all from .env
# ---------------------------------------------------------------------------
_DB_HOST     = os.environ.get("POSTGRES_HOST",     "localhost")
_DB_PORT     = int(os.environ.get("POSTGRES_PORT", "5432"))
_DB_NAME     = os.environ.get("POSTGRES_DB",       "cluj_monitor")
_DB_USER     = os.environ.get("POSTGRES_USER",     "postgres")
_DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

_UPSERT_RADIUS_M  = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "2.0"))
_DENSITY_RADIUS_M = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))

# ---------------------------------------------------------------------------
# Severity → SMALLINT
# ---------------------------------------------------------------------------
_SEVERITY_TO_INT: Dict[str, int] = {
    "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
}
_SEVERITY_WEIGHT: Dict[str, float] = {
    "S1": 0.10, "S2": 0.25, "S3": 0.50, "S4": 0.75, "S5": 1.00,
}


def _priority_score(severity_level: str, detection_count: int) -> float:
    """
    priority_score = w_severity × log(detection_count + 1)

    Road importance and infrastructure proximity are no longer available
    (enrichment stage removed). Severity and detection persistence are the
    only signals used for prioritisation.
    """
    w_sev = _SEVERITY_WEIGHT.get(severity_level, 0.25)
    return round(w_sev * math.log(detection_count + 1), 6)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DbWriterConfig:
    """
    DB credentials are read from .env in __post_init__.
    Never pass passwords as constructor arguments.
    """
    dry_run:  bool = False
    host:     str  = ""
    port:     int  = 0
    dbname:   str  = ""
    user:     str  = ""
    password: str  = ""

    def __post_init__(self) -> None:
        self.host     = _DB_HOST
        self.port     = _DB_PORT
        self.dbname   = _DB_NAME
        self.user     = _DB_USER
        self.password = _DB_PASSWORD
        if not self.password and not self.dry_run:
            logger.warning(
                "POSTGRES_PASSWORD is not set in .env — "
                "connection will likely fail."
            )


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------
@dataclass
class DbWriteResult:
    n_frames_processed:  int
    n_inserted:          int
    n_updated:           int
    n_skipped_no_gps:    int
    n_skipped_duplicate: int
    n_errors:            int
    dry_run:             bool
    elapsed_s:           float

    def log_summary(self) -> None:
        mode = "DRY RUN — nothing committed" if self.dry_run else "COMMITTED"
        logger.info("=== DB Write complete (%s) ===", mode)
        logger.info("  Frames processed       : %d", self.n_frames_processed)
        logger.info("  Rows inserted (new)    : %d", self.n_inserted)
        logger.info("  Rows updated (upsert)  : %d", self.n_updated)
        logger.info("  Skipped (no GPS)       : %d", self.n_skipped_no_gps)
        logger.info("  Skipped (duplicate)    : %d", self.n_skipped_duplicate)
        logger.info("  Errors                 : %d", self.n_errors)
        logger.info("  Elapsed                : %.1f s", self.elapsed_s)


# ---------------------------------------------------------------------------
# DbWriter
# ---------------------------------------------------------------------------

class DbWriter:
    """Stage 7 — PostgreSQL/PostGIS database write."""

    def __init__(self, cfg: DbWriterConfig) -> None:
        self.cfg   = cfg
        self._conn = None
        self._cur  = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        frames:     List[dict],
        output_dir: Optional[str] = None,
    ) -> DbWriteResult:
        t_start = time.perf_counter()
        n_inserted = n_updated = n_skipped_no_gps = n_skipped_dup = n_errors = 0

        if not self.cfg.dry_run:
            self._connect()

        try:
            for frame in frames:
                lat    = frame.get("latitude")
                lon    = frame.get("longitude")
                gps_ok = lat is not None and lon is not None

                # Resolve wall-clock date from wall_time ISO string
                wall_time_str = frame.get("wall_time")
                if wall_time_str:
                    try:
                        dt = datetime.fromisoformat(wall_time_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        dt = datetime.now(tz=timezone.utc)
                else:
                    dt = datetime.now(tz=timezone.utc)
                det_date   = dt.date()

                lighting   = frame.get("lighting", "unknown")
                video_file = frame.get("frame_path", "")

                for box in frame.get("boxes", []):
                    # Skip duplicates marked by DBSCAN
                    dedup = box.get("dedup", {}) or {}
                    if dedup.get("is_duplicate", False):
                        n_skipped_dup += 1
                        continue

                    # Skip GPS-less detections (PostGIS needs a POINT)
                    if not gps_ok:
                        n_skipped_no_gps += 1
                        continue

                    try:
                        row = self._build_row(
                            box, lat, lon, det_date, lighting, video_file
                        )
                    except Exception as exc:
                        logger.error(
                            "Row parse error — %s at (%.5f, %.5f): %s",
                            box.get("class_name", "?"), lat, lon, exc,
                        )
                        n_errors += 1
                        continue

                    if self.cfg.dry_run:
                        logger.debug(
                            "DRY RUN — %s  sev=%d  pri=%.4f  at (%.5f, %.5f)",
                            row["damage_type"], row["severity_int"],
                            row["priority_score"], lat, lon,
                        )
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
                            "DB upsert error — %s at (%.5f, %.5f): %s",
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
        result  = DbWriteResult(
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
    # Columns stored: spatial, detection, SAM geometry, depth, severity,
    # lighting, temporal, priority.
    # Removed: street_name, road_importance, infra_proximity_m, weather.
    # ------------------------------------------------------------------

    def _build_row(
        self,
        box:        dict,
        lat:        float,
        lon:        float,
        det_date:   date,
        lighting:   str,
        video_file: str,
    ) -> dict:
        geometry = box.get("geometry", {}) or {}
        depth    = box.get("depth",    {}) or {}
        severity = box.get("severity", {}) or {}

        sev_level = severity.get("severity_level", "S1")
        sev_int   = _SEVERITY_TO_INT.get(sev_level, 1)

        return {
            # spatial
            "geom_lon":          lon,
            "geom_lat":          lat,
            "latitude":          lat,
            "longitude":         lon,
            # detection
            "damage_type":       box.get("class_name", "unknown"),
            "confidence":        float(box.get("confidence") or 0.0),
            "frame_path":        video_file,
            # SAM geometry
            "surface_area_cm2":  float(geometry.get("surface_area_px")   or 0.0),
            "edge_sharpness":    float(geometry.get("edge_sharpness")     or 0.0),
            "interior_contrast": float(geometry.get("interior_contrast")  or 0.0),
            "mask_compactness":  float(geometry.get("mask_compactness")   or 0.0),
            # depth (relative, not metric)
            "depth_estimate_cm": float(depth.get("depth_norm")            or 0.0),
            "depth_confidence":  float(depth.get("depth_confidence")      or 1.0),
            # lighting
            "lighting_condition": lighting,
            # severity
            "severity_int":         sev_int,
            "severity_level":       sev_level,
            "severity_confidence":  float(severity.get("severity_confidence") or 1.0),
            # temporal
            "first_detected":    det_date,
            "last_detected":     det_date,
            "detection_count":   1,
            "deterioration_rate": 0.0,
            # derived
            "surrounding_density": 0,
            "priority_score":    _priority_score(sev_level, 1),
            # survey
            "survey_date":       det_date,
            "survey_video_file": video_file,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def _upsert(self, row: dict, lat: float, lon: float) -> str:
        self._cur.execute(
            """
            SELECT id, severity, first_detected, detection_count
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
            (row["damage_type"], lon, lat, _UPSERT_RADIUS_M, lon, lat),
        )
        existing = self._cur.fetchone()

        if existing is None:
            self._cur.execute(
                """
                INSERT INTO detections (
                    geom, latitude, longitude,
                    damage_type, confidence, frame_path,
                    surface_area_cm2, edge_sharpness,
                    interior_contrast, mask_compactness,
                    depth_estimate_cm, depth_confidence,
                    lighting_condition,
                    severity, severity_confidence,
                    first_detected, last_detected,
                    detection_count, deterioration_rate,
                    surrounding_density, priority_score,
                    survey_date, survey_video_file
                ) VALUES (
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                );
                """,
                (
                    row["geom_lon"],     row["geom_lat"],
                    row["latitude"],     row["longitude"],
                    row["damage_type"],  row["confidence"],  row["frame_path"],
                    row["surface_area_cm2"], row["edge_sharpness"],
                    row["interior_contrast"], row["mask_compactness"],
                    row["depth_estimate_cm"], row["depth_confidence"],
                    row["lighting_condition"],
                    row["severity_int"], row["severity_confidence"],
                    row["first_detected"], row["last_detected"],
                    row["detection_count"], row["deterioration_rate"],
                    row["surrounding_density"], row["priority_score"],
                    row["survey_date"],  row["survey_video_file"],
                ),
            )
            self._update_density(lon, lat)
            logger.debug(
                "INSERT  %-30s  sev=%d  pri=%.4f",
                row["damage_type"], row["severity_int"], row["priority_score"],
            )
            return "insert"

        else:
            existing_id = existing[0]
            old_sev_int = existing[1] or 1
            first_det   = existing[2]
            old_count   = existing[3]
            new_count   = old_count + 1

            today      = row["last_detected"]
            days_since = max(
                (today - first_det).days if isinstance(first_det, date) else 1,
                1,
            )
            detn_rate = round(
                (row["severity_int"] / 5.0 - old_sev_int / 5.0) / days_since, 6
            )
            new_pri = _priority_score(row["severity_level"], new_count)

            self._cur.execute(
                """
                UPDATE detections
                SET    last_detected       = %s,
                       detection_count     = %s,
                       deterioration_rate  = %s,
                       priority_score      = %s,
                       severity            = %s,
                       severity_confidence = %s,
                       confidence          = %s,
                       updated_at          = NOW()
                WHERE  id = %s;
                """,
                (
                    today, new_count, detn_rate, new_pri,
                    row["severity_int"], row["severity_confidence"],
                    row["confidence"], existing_id,
                ),
            )
            self._update_density(lon, lat)
            logger.debug(
                "UPDATE  %-30s  id=%s  count=%d→%d  pri=%.4f",
                row["damage_type"], existing_id, old_count, new_count, new_pri,
            )
            return "update"

    def _update_density(self, lon: float, lat: float) -> None:
        self._cur.execute(
            """
            UPDATE detections d
            SET    surrounding_density = (
                       SELECT COUNT(*) - 1
                       FROM   detections d2
                       WHERE  ST_DWithin(
                                  d.geom::geography, d2.geom::geography, %s
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
            logger.error("psycopg2 is required. Install: pip install psycopg2-binary")
            raise
        logger.info(
            "Connecting: %s@%s:%d/%s (password from .env)",
            self.cfg.user, self.cfg.host, self.cfg.port, self.cfg.dbname,
        )
        self._conn = psycopg2.connect(
            host=self.cfg.host, port=self.cfg.port,
            dbname=self.cfg.dbname, user=self.cfg.user, password=self.cfg.password,
        )
        self._conn.autocommit = False
        self._cur = self._conn.cursor()
        logger.info("Connected to PostgreSQL.")

    def _disconnect(self) -> None:
        for obj in (self._cur, self._conn):
            if obj:
                try:
                    obj.close()
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
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "n_frames_processed":   result.n_frames_processed,
                "n_inserted":           result.n_inserted,
                "n_updated":            result.n_updated,
                "n_skipped_no_gps":     result.n_skipped_no_gps,
                "n_skipped_duplicate":  result.n_skipped_duplicate,
                "n_errors":             result.n_errors,
                "dry_run":              result.dry_run,
                "elapsed_s":            result.elapsed_s,
            }, f, indent=2)
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
        description="Stage 7 — DB write. Credentials from .env."
    )
    parser.add_argument("--input",   required=True,
                        help="deduplicated.json from Stage 6")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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

    cfg = DbWriterConfig(dry_run=args.dry_run)
    DbWriter(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()