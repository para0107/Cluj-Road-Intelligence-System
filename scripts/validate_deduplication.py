"""
scripts/validate_deduplication.py
-----------------------------------
Validation script for Stage 6 — DBSCAN spatial deduplication.
(Renumbered from Stage 7 after enrichment removal.)

Reads severity_estimates.json (output of validate_severity.py) directly.
The enrichment step has been permanently removed, so SeverityResult dicts
are fed into the Deduplicator without any intermediate enrichment.

Environment variables (from .env):
    DEDUP_CLUSTER_RADIUS_M       — DBSCAN epsilon in metres (default 0.5)
    SURROUNDING_DENSITY_RADIUS_M — density radius (used by db_writer)

Outputs
-------
  data/validation_nrdd_2024/deduplicated/
      deduplicated.json      — all frames with dedup metadata
      dedup_report.html      — Leaflet map + Chart.js before/after chart

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

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_deduplication")

PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.deduplicator import Deduplicator, DeduplicatorConfig  # noqa: E402

# Input is now severity_estimates.json — enrichment step removed
SEVERITY_JSON = (
    PROJECT_ROOT
    / "data" / "validation_nrdd_2024" / "severity" / "severity_estimates.json"
)
OUT_DIR = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "deduplicated"


def main(verbose: bool = False) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEVERITY_JSON.exists():
        logger.error("severity_estimates.json not found: %s", SEVERITY_JSON)
        logger.error("Run scripts/validate_severity.py first.")
        raise SystemExit(1)

    logger.info("Loading: %s", SEVERITY_JSON)
    with SEVERITY_JSON.open("r", encoding="utf-8") as f:
        severity_data = json.load(f)

    frames = severity_data.get("frames", [])
    n_boxes_total = sum(len(fr.get("boxes", [])) for fr in frames)
    logger.info("Loaded %d frames, %d boxes total", len(frames), n_boxes_total)

    eps_m  = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "0.5"))
    dens_m = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))
    logger.info(
        "Config from .env: DEDUP_CLUSTER_RADIUS_M=%.1f m  "
        "SURROUNDING_DENSITY_RADIUS_M=%.1f m",
        eps_m, dens_m,
    )

    n_gps = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )
    if n_gps == 0:
        logger.warning(
            "No GPS coordinates found — DBSCAN will be skipped. "
            "All detections forwarded as-is. "
            "Use a GPS-synchronised dataset (e.g. KITTI) for spatial deduplication."
        )
    else:
        logger.info("GPS-equipped frames: %d / %d", n_gps, len(frames))

    cfg = DeduplicatorConfig(
        eps_m            = eps_m,
        min_samples      = 1,
        selection_metric = "severity_score",
    )

    t_start = time.perf_counter()
    results = Deduplicator(cfg).run(frames, output_dir=str(OUT_DIR))
    elapsed = time.perf_counter() - t_start

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
            "  HTML report not generated (no GPS). "
            "Re-run with a GPS-equipped dataset to produce the before/after map."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 6 validation — DBSCAN spatial deduplication."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)