"""
scripts/validate_db_write.py
------------------------------
Validation script for Stage 8 — PostgreSQL / PostGIS database write.

Reads deduplicated.json (output of validate_deduplication.py) and runs the
DbWriter in dry_run mode by default (no data is actually committed to the DB).
Pass --live to execute real writes against a running PostgreSQL + PostGIS instance.

In dry_run mode, the script:
  - Parses every retained detection into a row dict
  - Computes the priority_score formula for each row
  - Logs what would be inserted or upserted
  - Produces db_write_summary.json with counts and the parsed rows

In live mode, the script:
  - Connects to PostgreSQL (requires a running instance with the detections
    table created by scripts/setup_db.py)
  - Inserts new detections and upserts existing ones (within eps=2m, same class)
  - Updates surrounding_density for all records within 50m of each write
  - Commits the transaction and saves db_write_summary.json

Detections without GPS coordinates (latitude=None) are always skipped —
PostGIS requires a valid POINT geometry.

Setup prerequisites (live mode):
  1. Docker running PostgreSQL 15 + PostGIS:
       docker-compose up -d
  2. Table created:
       python scripts/setup_db.py
  3. psycopg2 installed:
       pip install psycopg2-binary

Outputs
-------
  data/validation_nrdd_2024/db_write/
      db_write_summary.json   -- insert/update/skip/error counts

Usage
-----
    python scripts/validate_db_write.py                   # dry run (default)
    python scripts/validate_db_write.py --live            # real writes
    python scripts/validate_db_write.py --live --host localhost --port 5432
    python scripts/validate_db_write.py --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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

def main(
    live:     bool = False,
    host:     str  = "localhost",
    port:     int  = 5432,
    dbname:   str  = "rids",
    user:     str  = "postgres",
    password: str  = "postgres",
    verbose:  bool = False,
) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load deduplicated.json — real data, no hardcoded values
    # ------------------------------------------------------------------
    if not DEDUP_JSON.exists():
        logger.error("deduplicated.json not found: %s", DEDUP_JSON)
        logger.error(
            "Run scripts/validate_deduplication.py first to generate it."
        )
        raise SystemExit(1)

    logger.info("Loading deduplicated results: %s", DEDUP_JSON)
    with DEDUP_JSON.open("r", encoding="utf-8") as f:
        dedup_data = json.load(f)

    frames = dedup_data.get("frames", [])

    n_retained_total = sum(
        1 for fr in frames
        for box in fr.get("boxes", [])
        if not box.get("dedup", {}).get("is_duplicate", False)
    )
    n_gps_equipped = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )

    logger.info(
        "Loaded %d frames | %d retained detections | %d GPS-equipped frames",
        len(frames), n_retained_total, n_gps_equipped,
    )

    if not live:
        logger.info(
            "=== DRY RUN MODE === "
            "No data will be written to the database. "
            "Pass --live to execute real writes."
        )

    # ------------------------------------------------------------------
    # GPS warning
    # ------------------------------------------------------------------
    if n_gps_equipped == 0:
        logger.warning(
            "No GPS coordinates found in deduplicated.json. "
            "All detections will be skipped (PostGIS requires valid lat/lon). "
            "Use a GPS-synchronised survey session to populate the database."
        )
        if not live:
            logger.info(
                "Dry run complete — 0 rows would be inserted. "
                "No db_write_summary.json written."
            )
            return

    # ------------------------------------------------------------------
    # Run DB writer
    # ------------------------------------------------------------------
    cfg = DbWriterConfig(
        host     = host,
        port     = port,
        dbname   = dbname,
        user     = user,
        password = password,
        dry_run  = not live,
    )

    writer = DbWriter(cfg)
    result = writer.run(frames, output_dir=str(OUT_DIR))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    mode = "LIVE" if live else "DRY RUN"
    logger.info("=== DB Write Validation Complete (%s) ===", mode)
    logger.info("  Inserted : %d", result.n_inserted)
    logger.info("  Updated  : %d", result.n_updated)
    logger.info("  Skipped  : %d (no GPS: %d, duplicate: %d)",
                result.n_skipped_no_gps + result.n_skipped_duplicate,
                result.n_skipped_no_gps,
                result.n_skipped_duplicate)
    logger.info("  Errors   : %d", result.n_errors)
    logger.info("  Output   : %s", OUT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 8 validation — PostgreSQL/PostGIS database write."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Execute real writes (default: dry run, no DB commits)",
    )
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=5432)
    parser.add_argument("--dbname",   default="rids")
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    main(
        live     = args.live,
        host     = args.host,
        port     = args.port,
        dbname   = args.dbname,
        user     = args.user,
        password = args.password,
        verbose  = args.verbose,
    )