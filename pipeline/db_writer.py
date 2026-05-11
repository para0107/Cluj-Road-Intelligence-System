"""
pipeline/db_writer.py
---------------------
Stage 7 — PostgreSQL/PostGIS database write.

This version is aligned with the intentionally simplified `detections` table
created by `scripts/setup_db.py` (no enrichment/weather/solar columns).

Writes:
- Inserts new detections
- Upserts (updates) an existing detection if same damage_type exists within
  DEDUP_CLUSTER_RADIUS_M metres.

Skips:
- Duplicates (dedup.is_duplicate=True)
- Detections without GPS

Priority:
    priority_score = w_severity * log(detection_count + 1)

Env (.env):
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  DEDUP_CLUSTER_RADIUS_M, SURROUNDING_DENSITY_RADIUS_M
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
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection — all from .env
# ---------------------------------------------------------------------------
_DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
_DB_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
_DB_NAME = os.environ.get("POSTGRES_DB", "cluj_monitor")
_DB_USER = os.environ.get("POSTGRES_USER", "postgres")
_DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

_UPSERT_RADIUS_M = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M", "2.0"))
_DENSITY_RADIUS_M = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))

# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------
_SEVERITY_TO_INT: Dict[str, int] = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
_SEVERITY_WEIGHT: Dict[str, float] = {"S1": 0.10, "S2": 0.25, "S3": 0.50, "S4": 0.75, "S5": 1.00}


def _priority_score(severity_level: str, detection_count: int) -> float:
    w_sev = _SEVERITY_WEIGHT.get(severity_level, 0.25)
    return round(w_sev * math.log(detection_count + 1), 6)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_wall_time_iso(wall_time_str: Optional[str]) -> datetime:
    if wall_time_str:
        try:
            dt = datetime.fromisoformat(wall_time_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _extract_severity(box: dict) -> Tuple[str, int, float]:
    """
    Accept multiple schema variants:
    - box["severity"] may be:
        * dict: {"severity_level": "S3", "severity_confidence": 0.8}
        * int: 3
        * str: "S3"
    """
    sev_conf = 1.0
    sev_level = "S1"

    raw = box.get("severity")

    if isinstance(raw, dict):
        sev_level = raw.get("severity_level") or raw.get("level") or raw.get("severity") or "S1"
        sev_conf = _safe_float(raw.get("severity_confidence"), 1.0)
    elif isinstance(raw, int):
        sev_level = f"S{max(1, min(5, raw))}"
    elif isinstance(raw, str):
        sev_level = raw if raw.startswith("S") else f"S{raw}"

    sev_level = sev_level if sev_level in _SEVERITY_TO_INT else "S1"
    sev_int = _SEVERITY_TO_INT[sev_level]
    return sev_level, sev_int, float(sev_conf)


def _extract_depth(box: dict) -> Tuple[float, float]:
    """
    Accept multiple schema variants:
    - box["depth"] dict: {"depth_norm": ..., "depth_confidence": ...}
    - box fields: {"depth_estimate_cm": ..., "depth_confidence": ...}
    """
    d = box.get("depth")
    if isinstance(d, dict):
        depth_est = _safe_float(d.get("depth_norm"), 0.0)
        depth_conf = _safe_float(d.get("depth_confidence"), 1.0)
        return depth_est, depth_conf

    depth_est = _safe_float(box.get("depth_estimate_cm"), 0.0)
    depth_conf = _safe_float(box.get("depth_confidence"), 1.0)
    return depth_est, depth_conf


def _extract_geometry(box: dict) -> Tuple[float, float, float, float]:
    """
    Accept multiple schema variants:
    - box["geometry"] dict with keys from Segmentor: surface_area_px/edge_sharpness/...
    """
    g = box.get("geometry")
    if not isinstance(g, dict):
        return 0.0, 0.0, 0.0, 0.0

    # NOTE: DB column is surface_area_cm2 but many pipeline stages still produce px.
    # We store the numeric value as-is (caller decides interpretation).
    return (
        _safe_float(g.get("surface_area_px") or g.get("surface_area_cm2"), 0.0),
        _safe_float(g.get("edge_sharpness"), 0.0),
        _safe_float(g.get("interior_contrast"), 0.0),
        _safe_float(g.get("mask_compactness"), 0.0),
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DbWriterConfig:
    dry_run: bool = False
    host: str = ""
    port: int = 0
    dbname: str = ""
    user: str = ""
    password: str = ""

    def __post_init__(self) -> None:
        self.host = _DB_HOST
        self.port = _DB_PORT
        self.dbname = _DB_NAME
        self.user = _DB_USER
        self.password = _DB_PASSWORD
        if not self.password and not self.dry_run:
            logger.warning("POSTGRES_PASSWORD is not set in .env — connection may fail.")


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------
@dataclass
class DbWriteResult:
    n_frames_processed: int
    n_inserted: int
    n_updated: int
    n_skipped_no_gps: int
    n_skipped_duplicate: int
    n_errors: int
    dry_run: bool
    elapsed_s: float

    def log_summary(self) -> None:
        mode = "DRY RUN — nothing committed" if self.dry_run else "COMMITTED"
        logger.info("=== DB Write complete (%s) ===", mode)
        logger.info("  Frames processed       : %d", self.n_frames_processed)
        logger.info("  Rows inserted (new)    : %d", self.n_inserted)
        logger.info("  Rows updated (upsert)  : %d", self.n_updated)
        logger.info("  Skipped (no GPS)       : %d", self.n_skipped_no_gps)
        logger.info("  Skipped (duplicate)    : %d", self.n_skipped_duplicate)
        logger.info("  Errors                 : %d", self.n_errors)
        logger.info("  Elapsed                : %.2f s", self.elapsed_s)


class DbWriter:
    """Stage 7 — PostgreSQL/PostGIS database write."""

    def __init__(self, cfg: DbWriterConfig) -> None:
        self.cfg = cfg
        self._conn = None
        self._cur = None

    def run(self, frames: List[dict], output_dir: Optional[str] = None) -> DbWriteResult:
        t_start = time.perf_counter()

        n_inserted = 0
        n_updated = 0
        n_skipped_no_gps = 0
        n_skipped_dup = 0
        n_errors = 0

        if not self.cfg.dry_run:
            self._connect()

        try:
            for frame in frames:
                lat = frame.get("latitude")
                lon = frame.get("longitude")
                gps_ok = lat is not None and lon is not None

                dt = _parse_wall_time_iso(frame.get("wall_time"))
                det_date = dt.date()

                lighting = frame.get("lighting", "unknown")
                frame_path = frame.get("frame_path", "") or ""

                for box in frame.get("boxes", []):
                    dedup = box.get("dedup", {}) or {}
                    if dedup.get("is_duplicate", False):
                        n_skipped_dup += 1
                        continue

                    if not gps_ok:
                        n_skipped_no_gps += 1
                        continue

                    try:
                        row = self._build_row(
                            box=box,
                            lat=float(lat),
                            lon=float(lon),
                            det_date=det_date,
                            lighting=lighting,
                            frame_path=frame_path,
                        )
                    except Exception:
                        logger.exception("Row build error (frame=%s)", frame_path)
                        n_errors += 1
                        continue

                    if self.cfg.dry_run:
                        n_inserted += 1
                        continue

                    try:
                        action = self._upsert(row)
                        if action == "insert":
                            n_inserted += 1
                        else:
                            n_updated += 1
                    except Exception:
                        logger.exception(
                            "DB upsert error (damage_type=%s lat=%.6f lon=%.6f)",
                            row["damage_type"], row["latitude"], row["longitude"],
                        )
                        n_errors += 1
                        if self._conn:
                            self._conn.rollback()

            if not self.cfg.dry_run and self._conn:
                self._conn.commit()
                logger.info("Transaction committed.")
        finally:
            self._disconnect()

        elapsed = round(time.perf_counter() - t_start, 2)
        result = DbWriteResult(
            n_frames_processed=len(frames),
            n_inserted=n_inserted,
            n_updated=n_updated,
            n_skipped_no_gps=n_skipped_no_gps,
            n_skipped_duplicate=n_skipped_dup,
            n_errors=n_errors,
            dry_run=self.cfg.dry_run,
            elapsed_s=elapsed,
        )
        result.log_summary()
        if output_dir:
            self._save_summary(result, output_dir)
        return result

    def _build_row(
        self,
        box: dict,
        lat: float,
        lon: float,
        det_date: date,
        lighting: str,
        frame_path: str,
    ) -> dict:
        sev_level, sev_int, sev_conf = _extract_severity(box)
        depth_est, depth_conf = _extract_depth(box)
        area, edge, contrast, compact = _extract_geometry(box)

        damage_type = box.get("class_name") or box.get("damage_type") or "unknown"
        confidence = _safe_float(box.get("confidence"), 0.0)

        # These are REQUIRED in DB schema (NOT NULL): first_detected/last_detected/survey_date
        return {
            "latitude": lat,
            "longitude": lon,
            "damage_type": str(damage_type),
            "confidence": float(confidence),
            "frame_path": frame_path,

            "surface_area_cm2": float(area),
            "edge_sharpness": float(edge),
            "interior_contrast": float(contrast),
            "mask_compactness": float(compact),

            "depth_estimate_cm": float(depth_est),
            "depth_confidence": float(depth_conf),

            "lighting_condition": str(lighting),

            "severity": int(sev_int),
            "severity_level": sev_level,  # internal use for priority
            "severity_confidence": float(sev_conf),

            "surrounding_density": 0,

            "first_detected": det_date,
            "last_detected": det_date,
            "detection_count": 1,
            "deterioration_rate": 0.0,

            "priority_score": _priority_score(sev_level, 1),

            "survey_date": det_date,
            "survey_video_file": frame_path,
        }

    def _upsert(self, row: dict) -> str:
        lat = row["latitude"]
        lon = row["longitude"]

        self._cur.execute(
            """
            SELECT id, severity, first_detected, detection_count
            FROM detections
            WHERE damage_type = %s
              AND ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    %s
                  )
            ORDER BY ST_Distance(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                 ) ASC
            LIMIT 1;
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
                    surface_area_cm2, edge_sharpness, interior_contrast, mask_compactness,
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
                    lon, lat,  # geom point
                    row["latitude"], row["longitude"],
                    row["damage_type"], row["confidence"], row["frame_path"],
                    row["surface_area_cm2"], row["edge_sharpness"], row["interior_contrast"], row["mask_compactness"],
                    row["depth_estimate_cm"], row["depth_confidence"],
                    row["lighting_condition"],
                    row["severity"], row["severity_confidence"],
                    row["first_detected"], row["last_detected"],
                    row["detection_count"], row["deterioration_rate"],
                    row["surrounding_density"], row["priority_score"],
                    row["survey_date"], row["survey_video_file"],
                ),
            )
            self._update_density(lon, lat)
            return "insert"

        existing_id, old_sev_int, first_det, old_count = existing
        old_sev_int = int(old_sev_int or 1)
        old_count = int(old_count or 1)

        new_count = old_count + 1
        today = row["last_detected"]

        days_since = max(
            (today - first_det).days if isinstance(first_det, date) else 1,
            1,
        )
        detn_rate = round((row["severity"] / 5.0 - old_sev_int / 5.0) / days_since, 6)
        new_pri = _priority_score(row["severity_level"], new_count)

        self._cur.execute(
            """
            UPDATE detections
            SET last_detected=%s,
                detection_count=%s,
                deterioration_rate=%s,
                priority_score=%s,
                severity=%s,
                severity_confidence=%s,
                confidence=%s,
                updated_at=NOW()
            WHERE id=%s;
            """,
            (
                today, new_count, detn_rate, new_pri,
                row["severity"], row["severity_confidence"],
                row["confidence"], existing_id,
            ),
        )
        self._update_density(lon, lat)
        return "update"

    def _update_density(self, lon: float, lat: float) -> None:
        self._cur.execute(
            """
            UPDATE detections d
            SET surrounding_density = (
                SELECT COUNT(*) - 1
                FROM detections d2
                WHERE ST_DWithin(d.geom::geography, d2.geom::geography, %s)
            )
            WHERE ST_DWithin(
                d.geom::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            );
            """,
            (_DENSITY_RADIUS_M, lon, lat, _DENSITY_RADIUS_M),
        )

    def _connect(self) -> None:
        try:
            import psycopg2
        except ImportError as e:
            raise RuntimeError("psycopg2 is required. Install: pip install psycopg2-binary") from e

        logger.info("Connecting: %s@%s:%d/%s", self.cfg.user, self.cfg.host, self.cfg.port, self.cfg.dbname)
        self._conn = psycopg2.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            dbname=self.cfg.dbname,
            user=self.cfg.user,
            password=self.cfg.password,
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
        self._cur = None
        self._conn = None

    @staticmethod
    def _save_summary(result: DbWriteResult, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "db_write_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_frames_processed": result.n_frames_processed,
                    "n_inserted": result.n_inserted,
                    "n_updated": result.n_updated,
                    "n_skipped_no_gps": result.n_skipped_no_gps,
                    "n_skipped_duplicate": result.n_skipped_duplicate,
                    "n_errors": result.n_errors,
                    "dry_run": result.dry_run,
                    "elapsed_s": result.elapsed_s,
                },
                f,
                indent=2,
            )
        logger.info("DB write summary saved: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 — DB write. Credentials from .env.")
    parser.add_argument("--input", required=True, help="deduplicated.json from Stage 6")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        dedup_data = json.load(f)

    frames = dedup_data.get("frames", [])
    logger.info("Loaded %d frames from %s", len(frames), in_path)

    cfg = DbWriterConfig(dry_run=args.dry_run)
    DbWriter(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()