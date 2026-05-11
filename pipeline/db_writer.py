"""
pipeline/db_writer.py
---------------------
Stage 8 of the road damage detection inference pipeline.

Responsibilities:
  - Accept frames from Stage 7 (deduplicated.json["frames"])
  - Write every retained (non-duplicate) GPS-equipped detection to the
    PostgreSQL 15 + PostGIS database (table: detections, created by setup_db.py)
  - Upsert: if same class already exists within DEDUP_CLUSTER_RADIUS_M metres,
    UPDATE; otherwise INSERT
  - Updates surrounding_density for all records within
    SURROUNDING_DENSITY_RADIUS_M metres of each write
  - Computes priority_score = w_severity × w_road × w_infra × log(count + 1)
  - Detections without GPS are always skipped (PostGIS requires a POINT geometry)

Environment variables read exclusively from .env (via python-dotenv):
    POSTGRES_HOST                — default localhost
    POSTGRES_PORT                — default 5432
    POSTGRES_DB                  — default cluj_monitor
    POSTGRES_USER                — default postgres
    POSTGRES_PASSWORD            — no default (must be set in .env)
    DATABASE_URL                 — SQLAlchemy URL (read as fallback reference)
    DEDUP_CLUSTER_RADIUS_M       — upsert proximity threshold (default 2.0)
    SURROUNDING_DENSITY_RADIUS_M — density search radius (default 50.0)

NO credentials are hardcoded in this file. All secrets come from .env.

Schema alignment with setup_db.py:
    surface_area_px  → surface_area_cm2    (px used as proxy until metric depth available)
    depth_norm       → depth_estimate_cm   (relative, not metric — name kept for compat)
    infra_proximity  → infra_proximity_m
    severity string  → severity SMALLINT   (S1=1 … S5=5)
    first/last dates → first_detected, last_detected  (DATE columns)

Usage (module):
    from pipeline.db_writer import DbWriter, DbWriterConfig
    result = DbWriter(DbWriterConfig()).run(deduplicated_frames)

Usage (CLI):
    python pipeline/db_writer.py
        --input   data/validation_nrdd_2024/deduplicated/deduplicated.json
        [--output data/validation_nrdd_2024/db_write/]
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
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env — must be called before reading os.environ
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection parameters — all from .env, no defaults for the password
# ---------------------------------------------------------------------------
_DB_HOST     = os.environ.get("POSTGRES_HOST",     "localhost")
_DB_PORT     = int(os.environ.get("POSTGRES_PORT", "5432"))
_DB_NAME     = os.environ.get("POSTGRES_DB",       "cluj_monitor")
_DB_USER     = os.environ.get("POSTGRES_USER",     "postgres")
_DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

# Radii from .env
_UPSERT_RADIUS_M  = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "2.0"))
_DENSITY_RADIUS_M = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))

# ---------------------------------------------------------------------------
# Severity level → SMALLINT (setup_db.py uses SMALLINT for severity)
# ---------------------------------------------------------------------------
_SEVERITY_TO_INT: Dict[str, int] = {
    "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
}

# Score weights for priority formula
_SEVERITY_WEIGHT: Dict[str, float] = {
    "S1": 0.10, "S2": 0.25, "S3": 0.50, "S4": 0.75, "S5": 1.00,
}
_ROAD_WEIGHT: Dict[int, float] = {
    1: 0.50,   # residential
    2: 0.75,   # secondary
    3: 1.00,   # primary / arterial
}
_INFRA_MAX_WEIGHT = 1.0
_INFRA_MIN_WEIGHT = 0.1
_INFRA_CLOSE_M    = 50.0
_INFRA_FAR_M      = 500.0


def _infra_weight(infra_m: Optional[float]) -> float:
    if infra_m is None:
        return _INFRA_MIN_WEIGHT
    if infra_m <= _INFRA_CLOSE_M:
        return _INFRA_MAX_WEIGHT
    if infra_m >= _INFRA_FAR_M:
        return _INFRA_MIN_WEIGHT
    frac = (infra_m - _INFRA_CLOSE_M) / (_INFRA_FAR_M - _INFRA_CLOSE_M)
    return _INFRA_MAX_WEIGHT - frac * (_INFRA_MAX_WEIGHT - _INFRA_MIN_WEIGHT)


def _priority_score(
    severity_level:  str,
    road_importance: Optional[int],
    infra_proximity: Optional[float],
    detection_count: int,
) -> float:
    """
    priority_score = w_severity × w_road × w_infra × log(detection_count + 1)
    Matches Eq. (priority) in the thesis.
    """
    w_sev   = _SEVERITY_WEIGHT.get(severity_level, 0.25)
    w_road  = _ROAD_WEIGHT.get(road_importance or 1, 0.50)
    w_infra = _infra_weight(infra_proximity)
    return round(w_sev * w_road * w_infra * math.log(detection_count + 1), 6)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DbWriterConfig:
    """
    All DB connection parameters are read from .env at class instantiation.
    Do not pass credentials as constructor arguments — use the .env file.

    dry_run:
        If True, parse all rows and log what would be written, but do not
        open a DB connection or execute any SQL. Safe without a running DB.
    """
    dry_run: bool = False

    # These are populated from .env in __post_init__
    host:     str = ""
    port:     int = 0
    dbname:   str = ""
    user:     str = ""
    password: str = ""

    def __post_init__(self) -> None:
        # Read exclusively from .env (already loaded above)
        self.host     = _DB_HOST
        self.port     = _DB_PORT
        self.dbname   = _DB_NAME
        self.user     = _DB_USER
        self.password = _DB_PASSWORD

        if not self.password and not self.dry_run:
            logger.warning(
                "POSTGRES_PASSWORD is not set in .env. "
                "Connection will likely fail. "
                "Set POSTGRES_PASSWORD in your .env file."
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
    """
    Stage 8 — PostgreSQL/PostGIS database write with upsert logic.

    All credentials come from .env. psycopg2 is imported lazily so the
    module can be imported in dry_run mode or on machines without a
    PostgreSQL client.
    """

    def __init__(self, cfg: DbWriterConfig) -> None:
        self.cfg   = cfg
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
        import time
        t_start = time.perf_counter()

        n_inserted       = 0
        n_updated        = 0
        n_skipped_no_gps = 0
        n_skipped_dup    = 0
        n_errors         = 0

        if not self.cfg.dry_run:
            self._connect()

        try:
            for frame in frames:
                lat    = frame.get("latitude")
                lon    = frame.get("longitude")
                gps_ok = lat is not None and lon is not None

                # Use wall_time (ISO string) when available.
                # timestamp_s is a relative offset (seconds since drive start),
                # NOT a Unix epoch — datetime.fromtimestamp() must not be called on it.
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
                det_date = dt.date()

                lighting   = frame.get("lighting", "unknown")
                video_file = frame.get("frame_path", "")

                for box in frame.get("boxes", []):
                    dedup = box.get("dedup", {})
                    if dedup.get("is_duplicate", False):
                        n_skipped_dup += 1
                        continue

                    if not gps_ok:
                        n_skipped_no_gps += 1
                        continue

                    try:
                        row = self._build_row(box, lat, lon, det_date,
                                              lighting, video_file)
                    except Exception as exc:
                        logger.error(
                            "Row parse error — class=%s (%.5f, %.5f): %s",
                            box.get("class_name", "?"), lat or 0, lon or 0, exc,
                        )
                        n_errors += 1
                        continue

                    if self.cfg.dry_run:
                        logger.debug(
                            "DRY RUN — would upsert: %s sev=%d pri=%.4f "
                            "at (%.5f, %.5f)",
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
    # Row construction — column names match setup_db.py exactly
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
        enrichment = box.get("enrichment", {}) or {}
        geometry   = box.get("geometry",   {}) or {}
        depth      = box.get("depth",      {}) or {}
        severity   = box.get("severity",   {}) or {}
        weather_d  = enrichment.get("weather") or {}

        sev_level   = severity.get("severity_level", "S1")
        sev_int     = _SEVERITY_TO_INT.get(sev_level, 1)
        road_imp    = enrichment.get("road_importance")
        infra_prox  = enrichment.get("infra_proximity")

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
            # SAM geometry — surface_area_px stored in surface_area_cm2 until
            # metric calibration is available (documented in setup_db.py comment)
            "surface_area_cm2":  float(geometry.get("surface_area_px") or 0.0),
            "edge_sharpness":    float(geometry.get("edge_sharpness")  or 0.0),
            "interior_contrast": float(geometry.get("interior_contrast") or 0.0),
            "mask_compactness":  float(geometry.get("mask_compactness")  or 0.0),
            # depth — relative, not metric
            "depth_estimate_cm": float(depth.get("depth_norm") or 0.0),
            "depth_confidence":  float(depth.get("depth_confidence") or 1.0),
            # lighting
            "lighting_condition":     lighting,
            "shadow_geometry_score":  None,
            # severity (SMALLINT 1-5)
            "severity_int":      sev_int,
            "severity_level":    sev_level,   # kept for internal priority calc
            "severity_confidence": float(severity.get("severity_confidence") or 1.0),
            # enrichment
            "street_name":       enrichment.get("street_name"),
            "road_importance":   road_imp,
            "infra_proximity_m": infra_prox,
            "weather":           json.dumps(weather_d) if weather_d else None,
            # temporal (DATE columns in setup_db.py)
            "first_detected":    det_date,
            "last_detected":     det_date,
            "detection_count":   1,
            "deterioration_rate": 0.0,
            # derived
            "surrounding_density": 0,
            "priority_score":    _priority_score(sev_level, road_imp, infra_prox, 1),
            # survey
            "survey_date":       det_date,
            "survey_video_file": video_file,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def _upsert(self, row: dict, lat: float, lon: float) -> str:
        """
        Search for an existing detection of the same class within
        _UPSERT_RADIUS_M (from .env DEDUP_CLUSTER_RADIUS_M).
        UPDATE if found; INSERT if not.
        Returns "insert" or "update".
        """
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
            # INSERT new row
            self._cur.execute(
                """
                INSERT INTO detections (
                    geom, latitude, longitude,
                    damage_type, confidence, frame_path,
                    surface_area_cm2, edge_sharpness,
                    interior_contrast, mask_compactness,
                    depth_estimate_cm, depth_confidence,
                    lighting_condition, shadow_geometry_score,
                    severity, severity_confidence,
                    street_name, road_importance, infra_proximity_m,
                    weather,
                    first_detected, last_detected,
                    detection_count, deterioration_rate,
                    surrounding_density, priority_score,
                    survey_date, survey_video_file
                ) VALUES (
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s::jsonb,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                );
                """,
                (
                    row["geom_lon"], row["geom_lat"],
                    row["latitude"], row["longitude"],
                    row["damage_type"], row["confidence"], row["frame_path"],
                    row["surface_area_cm2"], row["edge_sharpness"],
                    row["interior_contrast"], row["mask_compactness"],
                    row["depth_estimate_cm"], row["depth_confidence"],
                    row["lighting_condition"], row["shadow_geometry_score"],
                    row["severity_int"], row["severity_confidence"],
                    row["street_name"], row["road_importance"],
                    row["infra_proximity_m"],
                    row["weather"],
                    row["first_detected"], row["last_detected"],
                    row["detection_count"], row["deterioration_rate"],
                    row["surrounding_density"], row["priority_score"],
                    row["survey_date"], row["survey_video_file"],
                ),
            )
            self._update_density(lon, lat)
            logger.debug(
                "INSERT  %s  sev=%d  pri=%.4f",
                row["damage_type"], row["severity_int"], row["priority_score"],
            )
            return "insert"

        else:
            # UPDATE existing row
            existing_id      = existing[0]
            old_sev_int      = existing[1] or 1
            first_det        = existing[2]
            old_count        = existing[3]
            new_count        = old_count + 1

            # Deterioration rate: Δseverity per day
            today      = row["last_detected"]
            days_since = max(
                (today - first_det).days if isinstance(first_det, date) else 1,
                1,
            )
            detn_rate = round(
                (row["severity_int"] / 5.0 - old_sev_int / 5.0) / days_since, 6
            )
            new_pri = _priority_score(
                row["severity_level"],
                row["road_importance"],
                row["infra_proximity_m"],
                new_count,
            )

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
                "UPDATE  %s  id=%s  count=%d→%d  pri=%.4f",
                row["damage_type"], existing_id, old_count, new_count, new_pri,
            )
            return "update"

    def _update_density(self, lon: float, lat: float) -> None:
        """
        Recalculate surrounding_density for all records within
        _DENSITY_RADIUS_M (from .env SURROUNDING_DENSITY_RADIUS_M)
        of the point just written.
        """
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
            logger.error(
                "psycopg2 is required for Stage 8.\n"
                "Install: pip install psycopg2-binary"
            )
            raise

        logger.info(
            "Connecting: %s@%s:%d/%s (password from .env)",
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
        description=(
            "Stage 8 — PostgreSQL/PostGIS database write. "
            "DB credentials are read from .env (POSTGRES_*)."
        )
    )
    parser.add_argument("--input",   required=True,
                        help="deduplicated.json from Stage 7")
    parser.add_argument("--output",  default=None,
                        help="Directory to save db_write_summary.json (optional)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and validate without writing to DB")
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

    # Credentials come entirely from .env — no CLI arguments for secrets
    cfg = DbWriterConfig(dry_run=args.dry_run)
    DbWriter(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()