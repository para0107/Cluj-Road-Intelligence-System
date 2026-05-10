"""
scripts/validate_deduplication.py
-----------------------------------
Validation script for Stage 7 — DBSCAN spatial deduplication.

Reads enriched.json (output of validate_enrichment.py) and runs the
Deduplicator. The DBSCAN epsilon is read from DEDUP_CLUSTER_RADIUS_M in .env.
If no GPS coordinates are present, DBSCAN is skipped gracefully.

Environment variables used (from .env):
    DEDUP_CLUSTER_RADIUS_M       — DBSCAN epsilon in metres (default 2)
    SURROUNDING_DENSITY_RADIUS_M — density radius (used by db_writer, logged here)
    CITY_LAT, CITY_LON           — map centre fallback for the HTML report

Outputs
-------
  data/validation_nrdd_2024/deduplicated/
      deduplicated.json     -- all frames with dedup metadata
      dedup_report.html     -- interactive HTML: chart + Leaflet map

Usage
-----
    python scripts/validate_deduplication.py
    python scripts/validate_deduplication.py --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
logger = logging.getLogger("validate_deduplication")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.deduplicator import Deduplicator, DeduplicatorConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENRICHED_JSON = (
    PROJECT_ROOT
    / "data" / "validation_nrdd_2024" / "enriched" / "enriched.json"
)
OUT_DIR = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "deduplicated"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(verbose: bool = False) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load enriched.json
    # ------------------------------------------------------------------
    if not ENRICHED_JSON.exists():
        logger.error("enriched.json not found: %s", ENRICHED_JSON)
        logger.error("Run scripts/validate_enrichment.py first.")
        raise SystemExit(1)

    logger.info("Loading: %s", ENRICHED_JSON)
    with ENRICHED_JSON.open("r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    frames = enriched_data.get("frames", [])
    n_boxes_total = sum(len(fr.get("boxes", [])) for fr in frames)
    logger.info("Loaded %d frames, %d boxes total", len(frames), n_boxes_total)

    # ------------------------------------------------------------------
    # .env parameters
    # ------------------------------------------------------------------
    eps_m    = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "2.0"))
    dens_m   = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))

    logger.info(
        "Config from .env: DEDUP_CLUSTER_RADIUS_M=%.1f m  "
        "SURROUNDING_DENSITY_RADIUS_M=%.1f m",
        eps_m, dens_m,
    )

    # ------------------------------------------------------------------
    # GPS check
    # ------------------------------------------------------------------
    n_gps = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )
    if n_gps == 0:
        logger.warning(
            "No GPS coordinates found in enriched.json — "
            "DBSCAN will be skipped. All detections forwarded as-is. "
            "Use a GPS-synchronised dataset (e.g. KITTI) for full validation."
        )
    else:
        logger.info("GPS-equipped frames: %d / %d — DBSCAN will run", n_gps, len(frames))

    # ------------------------------------------------------------------
    # Run deduplication
    # ------------------------------------------------------------------
    cfg = DeduplicatorConfig(
        eps_m            = eps_m,
        min_samples      = 1,
        selection_metric = "severity_score",
    )

    t_start = time.perf_counter()
    results = Deduplicator(cfg).run(frames, output_dir=str(OUT_DIR))
    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total    = sum(len(r.boxes) for r in results)
    n_retained = sum(len(r.retained) for r in results)
    n_removed  = sum(len(r.removed)  for r in results)
    pct        = n_removed / max(n_total, 1) * 100

    logger.info("=== Deduplication Validation Complete ===")
    logger.info("  Total detections  : %d", n_total)
    logger.info("  Retained          : %d  (%.1f%%)", n_retained, 100 - pct)
    logger.info("  Removed (dups)    : %d  (%.1f%%)", n_removed, pct)
    logger.info("  Elapsed           : %.1f s", elapsed)
    logger.info("  Output            : %s", OUT_DIR)

    if n_gps == 0:
        logger.info(
            "  HTML report: not generated (no GPS). "
            "Re-run with GPS-equipped data to produce the before/after map."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 7 validation — DBSCAN spatial deduplication."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)