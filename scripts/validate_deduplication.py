"""
scripts/validate_deduplication.py
-----------------------------------
Validation script for Stage 7 — DBSCAN spatial deduplication.

Reads enriched.json (output of validate_enrichment.py) and runs the
Deduplicator. If no GPS coordinates are present in the enriched data
(which is the case for the Cluj dashcam session without a .gpx file),
DBSCAN is skipped and all detections are forwarded unchanged — this is
the documented graceful fallback.

When GPS data IS available (e.g. KITTI dataset, or a future GPS-synchronised
Cluj survey), DBSCAN clusters detections within eps_m metres. Within each
cluster, the detection with the highest severity_score is retained; all
others are marked is_duplicate=True.

Outputs
-------
  data/validation_nrdd_2024/deduplicated/
      deduplicated.json     -- all frames with dedup metadata added
      dedup_report.html     -- interactive HTML report with:
                               - summary statistics table
                               - per-class stacked bar chart (before vs after)
                               - Leaflet.js map (green=retained, red=removed)

Usage
-----
    python scripts/validate_deduplication.py
    python scripts/validate_deduplication.py --eps_m 3.0
    python scripts/validate_deduplication.py --verbose

DBSCAN reference:
  Ester, Kriegel, Sander, Xu — KDD 1996.
  "A Density-Based Algorithm for Discovering Clusters in Large Spatial
  Databases with Noise." ACM KDD 1996. dl.acm.org/doi/10.5555/3001460.3001507

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

def main(
    eps_m:            float = 2.0,
    min_samples:      int   = 1,
    selection_metric: str   = "severity_score",
    verbose:          bool  = False,
) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load enriched.json — real data, no hardcoded values
    # ------------------------------------------------------------------
    if not ENRICHED_JSON.exists():
        logger.error("enriched.json not found: %s", ENRICHED_JSON)
        logger.error(
            "Run scripts/validate_enrichment.py first to generate it."
        )
        raise SystemExit(1)

    logger.info("Loading enriched results: %s", ENRICHED_JSON)
    with ENRICHED_JSON.open("r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    frames = enriched_data.get("frames", [])
    logger.info("Loaded %d frames (%d boxes total)",
                len(frames),
                sum(len(fr.get("boxes", [])) for fr in frames))

    # ------------------------------------------------------------------
    # GPS availability check
    # ------------------------------------------------------------------
    n_with_gps = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )

    if n_with_gps == 0:
        logger.warning(
            "No GPS coordinates found in enriched.json — "
            "DBSCAN spatial clustering will be skipped. "
            "All detections will be forwarded as-is (is_duplicate=False). "
            "To validate deduplication, use a GPS-synchronised dataset "
            "(e.g. KITTI, or a future Cluj survey with a .gpx file)."
        )
    else:
        logger.info(
            "GPS-equipped frames: %d / %d — DBSCAN will run",
            n_with_gps, len(frames),
        )

    # ------------------------------------------------------------------
    # Run deduplication
    # ------------------------------------------------------------------
    cfg = DeduplicatorConfig(
        eps_m            = eps_m,
        min_samples      = min_samples,
        selection_metric = selection_metric,
    )

    t_start = time.perf_counter()
    dedup   = Deduplicator(cfg)
    results = dedup.run(frames, output_dir=str(OUT_DIR))
    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total    = sum(len(r.boxes) for r in results)
    n_retained = sum(len(r.retained) for r in results)
    n_removed  = sum(len(r.removed)  for r in results)
    pct_removed = n_removed / max(n_total, 1) * 100

    logger.info("=== Deduplication Validation Complete ===")
    logger.info("  Total detections  : %d", n_total)
    logger.info("  Retained          : %d  (%.1f%%)", n_retained, 100 - pct_removed)
    logger.info("  Removed (dups)    : %d  (%.1f%%)", n_removed, pct_removed)
    logger.info("  Elapsed           : %.1f s", elapsed)
    logger.info("  Output directory  : %s", OUT_DIR)

    if n_with_gps == 0:
        logger.info(
            "  Note: Dedup report HTML was not generated because no GPS "
            "data was available. Re-run with GPS-equipped data to produce "
            "the before/after map and class-level removal chart."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 7 validation — DBSCAN spatial deduplication."
    )
    parser.add_argument(
        "--eps_m", type=float, default=2.0,
        help="DBSCAN epsilon in metres (default: 2.0)",
    )
    parser.add_argument(
        "--min_samples", type=int, default=1,
        help="DBSCAN min_samples (default: 1)",
    )
    parser.add_argument(
        "--selection_metric", default="severity_score",
        choices=["severity_score", "confidence"],
        help="Which field selects the representative in a cluster (default: severity_score)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    main(
        eps_m            = args.eps_m,
        min_samples      = args.min_samples,
        selection_metric = args.selection_metric,
        verbose          = args.verbose,
    )