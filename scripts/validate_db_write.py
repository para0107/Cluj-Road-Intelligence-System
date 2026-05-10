"""
scripts/validate_db_write.py
------------------------------
Validation script for Stage 8 — PostgreSQL / PostGIS database write.

Reads deduplicated.json (output of validate_deduplication.py) and runs the
DbWriter. All database credentials are read exclusively from the project
.env file. No passwords or connection strings are accepted as CLI arguments.

Environment variables used (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
    POSTGRES_USER, POSTGRES_PASSWORD
    DATABASE_URL  (logged for reference only, not used directly here)
    DEDUP_CLUSTER_RADIUS_M       — upsert proximity threshold
    SURROUNDING_DENSITY_RADIUS_M — density search radius

Run modes:
    Default (--dry_run, the safe default): parses all rows, logs what would
    be inserted or updated, produces db_write_summary.json — no DB connection
    is opened, no data is written.

    --live: executes real writes. Requires a running PostgreSQL + PostGIS
    instance (docker-compose up -d) and the tables created by setup_db.py.

Outputs
-------
  data/validation_nrdd_2024/db_write/
      db_write_summary.json   -- insert/update/skip/error counts

Usage
-----
    python scripts/validate_db_write.py              # dry run (safe default)
    python scripts/validate_db_write.py --live       # real writes from .env creds
    python scripts/validate_db_write.py --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_db_write")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db_writer import DbWriter, DbWriterConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEDUP_JSON = (
    PROJECT_ROOT
    / "data" / "validation_nrdd_2024" / "deduplicated" / "deduplicated.json"
)
OUT_DIR = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "db_write"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(live: bool = False, verbose: bool = False) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Log .env config (no passwords printed)
    # ------------------------------------------------------------------
    logger.info(
        "DB config from .env: host=%s  port=%s  db=%s  user=%s",
        os.environ.get("POSTGRES_HOST", "localhost"),
        os.environ.get("POSTGRES_PORT", "5432"),
        os.environ.get("POSTGRES_DB",   "cluj_monitor"),
        os.environ.get("POSTGRES_USER", "postgres"),
    )
    logger.info(
        "Radii from .env: DEDUP_CLUSTER_RADIUS_M=%s m  "
        "SURROUNDING_DENSITY_RADIUS_M=%s m",
        os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "2.0"),
        os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"),
    )

    # ------------------------------------------------------------------
    # Load deduplicated.json
    # ------------------------------------------------------------------
    if not DEDUP_JSON.exists():
        logger.error("deduplicated.json not found: %s", DEDUP_JSON)
        logger.error("Run scripts/validate_deduplication.py first.")
        raise SystemExit(1)

    logger.info("Loading: %s", DEDUP_JSON)
    with DEDUP_JSON.open("r", encoding="utf-8") as f:
        dedup_data = json.load(f)

    frames = dedup_data.get("frames", [])

    n_retained = sum(
        1 for fr in frames
        for box in fr.get("boxes", [])
        if not box.get("dedup", {}).get("is_duplicate", False)
    )
    n_gps = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )

    logger.info(
        "Loaded %d frames | %d retained detections | %d GPS-equipped frames",
        len(frames), n_retained, n_gps,
    )

    if not live:
        logger.info(
            "=== DRY RUN MODE === "
            "No DB connection will be opened. "
            "Pass --live to execute real writes."
        )

    if n_gps == 0:
        logger.warning(
            "No GPS coordinates — 0 rows will be inserted. "
            "PostGIS requires valid lat/lon for the POINT geometry. "
            "This is expected for the current Cluj session (no .gpx file)."
        )

    # ------------------------------------------------------------------
    # Run DB write — credentials from .env via DbWriterConfig.__post_init__
    # ------------------------------------------------------------------
    cfg    = DbWriterConfig(dry_run=not live)
    result = DbWriter(cfg).run(frames, output_dir=str(OUT_DIR))

    # ------------------------------------------------------------------
    # Final log
    # ------------------------------------------------------------------
    mode = "LIVE" if live else "DRY RUN"
    logger.info("=== DB Write Validation Complete (%s) ===", mode)
    logger.info("  Inserted : %d", result.n_inserted)
    logger.info("  Updated  : %d", result.n_updated)
    logger.info(
        "  Skipped  : %d  (no GPS: %d | duplicate: %d)",
        result.n_skipped_no_gps + result.n_skipped_duplicate,
        result.n_skipped_no_gps,
        result.n_skipped_duplicate,
    )
    logger.info("  Errors   : %d", result.n_errors)
    logger.info("  Output   : %s", OUT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 8 validation — PostgreSQL/PostGIS database write. "
            "All DB credentials are read from .env (POSTGRES_*)."
        )
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Execute real writes (default: dry run, no DB connection opened)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(live=args.live, verbose=args.verbose)