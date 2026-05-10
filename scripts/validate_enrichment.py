"""
scripts/validate_enrichment.py
--------------------------------
Validation script for Stage 6 — Spatial and weather enrichment.

Reads severity_results.json (output of validate_severity.py) and runs
the Enricher. All API URLs and the Nominatim User-Agent are read from
the project .env file — no secrets or URLs are hardcoded here.

Environment variables used (from .env):
    NOMINATIM_USER_AGENT   — required by Nominatim usage policy
    OSM_OVERPASS_URL       — Overpass endpoint
    OPEN_METEO_BASE_URL    — Open-Meteo endpoint

Outputs
-------
  data/validation_nrdd_2024/enriched/
      enriched.json           -- all frames with enrichment fields added
      enrichment_report.json  -- GPS rate, API success/fail counts, per-class stats

Usage
-----
    python scripts/validate_enrichment.py
    python scripts/validate_enrichment.py --skip_weather
    python scripts/validate_enrichment.py --skip_overpass
    python scripts/validate_enrichment.py --limit 10 --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env before any pipeline imports (pipeline modules also call load_dotenv,
# but loading here first ensures variables are available even if run standalone)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_enrichment")

# ---------------------------------------------------------------------------
# Project root — add to sys.path so pipeline modules can be imported
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.enricher import Enricher, EnricherConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Paths — all absolute, derived from PROJECT_ROOT
# ---------------------------------------------------------------------------
SEVERITY_JSON = (
    PROJECT_ROOT
    / "data" / "validation_nrdd_2024" / "severity" / "severity_results.json"
)
OUT_DIR = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "enriched"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    limit:        Optional[int] = None,
    skip_weather: bool = False,
    skip_overpass: bool = False,
    verbose:      bool = False,
) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load severity_results.json
    # ------------------------------------------------------------------
    if not SEVERITY_JSON.exists():
        logger.error("severity_results.json not found: %s", SEVERITY_JSON)
        logger.error("Run scripts/validate_severity.py first.")
        raise SystemExit(1)

    logger.info("Loading: %s", SEVERITY_JSON)
    with SEVERITY_JSON.open("r", encoding="utf-8") as f:
        severity_data = json.load(f)

    frames = severity_data.get("frames", [])
    logger.info(
        "Loaded %d frames | model=%s | device=%s",
        len(frames),
        severity_data.get("model", "?"),
        severity_data.get("device", "?"),
    )

    if limit is not None:
        frames = frames[:limit]
        logger.info("Limit applied: first %d frames", limit)

    # ------------------------------------------------------------------
    # GPS check
    # ------------------------------------------------------------------
    n_with_gps = sum(
        1 for fr in frames
        if fr.get("latitude") is not None and fr.get("longitude") is not None
    )
    logger.info(
        "GPS: %d frames with coordinates, %d without",
        n_with_gps, len(frames) - n_with_gps,
    )
    if n_with_gps == 0:
        logger.warning(
            "No GPS coordinates found in severity_results.json. "
            "All enrichment fields will be null. "
            "This is expected for the current Cluj dashcam session (no .gpx file). "
            "Run again with a GPS-synchronised dataset to get meaningful enrichment."
        )

    # ------------------------------------------------------------------
    # Run enrichment — API credentials come from .env via pipeline/enricher.py
    # ------------------------------------------------------------------
    cfg = EnricherConfig(
        delay_nominatim_s = 1.1,   # Nominatim policy: max 1 req/s
        delay_overpass_s  = 0.5,
        timeout_s         = 10.0,
        skip_weather      = skip_weather,
        skip_overpass     = skip_overpass,
    )

    t_start  = time.perf_counter()
    results  = Enricher(cfg).run(frames, output_dir=str(OUT_DIR))
    elapsed  = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Build and save enrichment_report.json
    # ------------------------------------------------------------------
    n_boxes_total = sum(r.n_detections for r in results)
    n_enriched    = sum(
        1 for r in results for b in r.boxes if b.street_name is not None
    )
    n_weather_ok  = sum(
        1 for r in results for b in r.boxes if b.weather is not None
    )

    class_counts:   dict = defaultdict(int)
    class_enriched: dict = defaultdict(int)
    for r in results:
        for b in r.boxes:
            cls = b.raw.get("class_name", "unknown")
            class_counts[cls] += 1
            if b.street_name is not None:
                class_enriched[cls] += 1

    report = {
        "source_severity_json":  str(SEVERITY_JSON),
        "n_frames":              len(results),
        "n_frames_with_gps":     n_with_gps,
        "n_frames_without_gps":  len(results) - n_with_gps,
        "n_boxes_total":         n_boxes_total,
        "n_boxes_enriched":      n_enriched,
        "n_boxes_with_weather":  n_weather_ok,
        "elapsed_seconds":       round(elapsed, 1),
        "config": {
            "skip_weather":  skip_weather,
            "skip_overpass": skip_overpass,
        },
        "per_class": {
            cls: {"total": class_counts[cls], "enriched": class_enriched[cls]}
            for cls in sorted(class_counts)
        },
    }

    report_path = OUT_DIR / "enrichment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Enrichment report saved: %s", report_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=== Enrichment Validation Complete ===")
    logger.info("  Frames processed          : %d", len(results))
    logger.info("  Frames with GPS           : %d", n_with_gps)
    logger.info("  Boxes enriched (Nominatim): %d / %d", n_enriched, n_boxes_total)
    logger.info("  Boxes with weather        : %d / %d", n_weather_ok, n_boxes_total)
    logger.info("  Elapsed                   : %.1f s", elapsed)
    logger.info("  Output                    : %s", OUT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 6 validation — spatial and weather enrichment."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames")
    parser.add_argument("--skip_weather",  action="store_true",
                        help="Skip Open-Meteo weather lookups")
    parser.add_argument("--skip_overpass", action="store_true",
                        help="Skip OSM Overpass lookups")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(
        limit         = args.limit,
        skip_weather  = args.skip_weather,
        skip_overpass = args.skip_overpass,
        verbose       = args.verbose,
    )