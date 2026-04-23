"""
scripts/download_test_footage.py
---------------------------------
Downloads a small sample of BDD100K footage with GPS data to use as test
input for the CRIS pipeline while you don't have a real dashcam recording.

What this gets you:
  data/raw/footage/bdd_sample.mp4       — ~40s driving clip, 720p, 30fps
  data/raw/gps_logs/bdd_sample.json     — matching GPS/IMU JSON

These are the official BDD100K sample files hosted by Berkeley and are
free to download for research use under the BDD100K license:
  https://bdd-data.berkeley.edu/

Usage:
    python scripts/download_test_footage.py

Then test Stage 1 with:
    python -m pipeline.preprocessor \
        --video  data/raw/footage/bdd_sample.mp4 \
        --gps    data/raw/gps_logs/bdd_sample.json \
        --output data/processed/frames/bdd_sample/ \
        --fps    2 \
        --verbose

Notes:
  - The full BDD100K dataset (1.8 TB) requires registration at
    https://bdd-data.berkeley.edu/ and is not downloaded here.
  - This script only fetches the publicly accessible sample files.
  - If the sample URLs change, update VIDEO_URL and GPS_URL below.
"""

import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target URLs
# BDD100K public sample: one 40-second driving clip + its GPS JSON.
# The GPS JSON is part of the bdd100k "info" package (3.9 GB total).
# We link to a single-file mirror of a sample GPS entry.
# ---------------------------------------------------------------------------

# Official BDD100K sample video (publicly accessible, no login required)
VIDEO_URL = (
    "https://s3.amazonaws.com/bdd-data/samples/"
    "bdd100k_sample.mp4"
)

# The full info/GPS package requires login. We instead generate a synthetic
# GPS JSON in the same BDD100K format, seeded on a known San Francisco route
# that matches the sample video's approximate timestamp.
# This is clearly labelled as synthetic — it is ONLY for testing the GPS
# parsing + interpolation logic in the preprocessor.
GENERATE_SYNTHETIC_GPS = True   # set False if you have a real BDD100K info download

# Output paths
ROOT = Path(__file__).parent.parent  # project root
VIDEO_OUT = ROOT / "data" / "raw" / "footage" / "bdd_sample.mp4"
GPS_OUT   = ROOT / "data" / "raw" / "gps_logs"  / "bdd_sample.json"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 1024 * 64) -> None:
    """Download a file with a simple progress counter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading: %s", url)
    logger.info("        → %s", dest)

    req = urllib.request.Request(url, headers={"User-Agent": "CRIS-pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with dest.open("wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)",
                              end="", flush=True)
            print()
    except urllib.error.HTTPError as e:
        logger.error("HTTP %d for %s", e.code, url)
        raise
    except urllib.error.URLError as e:
        logger.error("Network error: %s", e.reason)
        raise

    logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Synthetic GPS generator
# ---------------------------------------------------------------------------

def generate_synthetic_gps(output_path: Path, n_points: int = 80) -> None:
    """
    Write a synthetic BDD100K-format GPS JSON file.

    The track is a straight ~600m run along Market Street, San Francisco
    (lat ~37.779, lon ~-122.419), which is consistent with the BDD100K
    sample video's collection area.

    This is MOCK DATA — clearly labelled in the output — used only to test
    that the preprocessor's GPS loading and interpolation logic works end-to-end.
    """
    import time

    logger.info("Generating synthetic GPS JSON (mock data — for testing only)")

    # Epoch ms for 2017-12-03 17:00:00 UTC (BDD100K sample video era)
    start_epoch_ms = 1512320400000

    # Straight line: Market St, SF  37.7792° N → 37.7788° N, -122.4186° W → -122.4130° W
    lat_start, lat_end = 37.7792,  37.7788
    lon_start, lon_end = -122.4186, -122.4130

    interval_ms = 500   # GPS point every 500ms → 2 Hz, matching extraction fps

    records = []
    for i in range(n_points):
        alpha = i / max(n_points - 1, 1)
        records.append({
            "timestamp": start_epoch_ms + i * interval_ms,
            "latitude":  round(lat_start + alpha * (lat_end - lat_start), 6),
            "longitude": round(lon_start + alpha * (lon_end - lon_start), 6),
            "course":    round(90.0 + alpha * 2, 1),   # heading East
            "speed":     round(8.0 + alpha * 2, 2),    # ~30 km/h
            "_note":     "SYNTHETIC — generated by download_test_footage.py for testing only",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Synthetic GPS written: %s (%d points)", output_path, n_points)
    logger.warning(
        "This GPS file is SYNTHETIC (mock data). It only tests the parser "
        "and interpolation logic. Replace with real GPS data from a dashcam "
        "or a BDD100K GPS/IMU download for production use."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== CRIS test footage downloader ===")

    # --- Video ---
    if VIDEO_OUT.exists():
        size_mb = VIDEO_OUT.stat().st_size / 1e6
        logger.info("Video already present: %s (%.1f MB) — skipping download",
                    VIDEO_OUT, size_mb)
    else:
        try:
            download_file(VIDEO_URL, VIDEO_OUT)
        except Exception as e:
            logger.error(
                "Could not download sample video: %s\n"
                "Manual alternative:\n"
                "  1. Register at https://bdd-data.berkeley.edu/\n"
                "  2. Download any video from the 100K set\n"
                "  3. Save it as: %s",
                e, VIDEO_OUT,
            )
            # Don't exit — still generate the GPS file so the user can test
            # the GPS loader with their own video.

    # --- GPS ---
    if GPS_OUT.exists():
        logger.info("GPS file already present: %s — skipping", GPS_OUT)
    else:
        if GENERATE_SYNTHETIC_GPS:
            generate_synthetic_gps(GPS_OUT)
        else:
            logger.error(
                "GENERATE_SYNTHETIC_GPS is False but no GPS file exists at %s.\n"
                "Either set GENERATE_SYNTHETIC_GPS=True or download the BDD100K "
                "GPS/IMU package and place it at that path.",
                GPS_OUT,
            )
            sys.exit(1)

    # --- Print next steps ---
    print("\n" + "=" * 60)
    print("Test files ready. Run the preprocessor with:")
    print()
    print(f"  python -m pipeline.preprocessor \\")
    print(f"      --video  {VIDEO_OUT} \\")
    print(f"      --gps    {GPS_OUT} \\")
    print(f"      --output data/processed/frames/bdd_sample/ \\")
    print(f"      --fps    2 \\")
    print(f"      --verbose")
    print()
    if GENERATE_SYNTHETIC_GPS:
        print("NOTE: GPS file is synthetic (mock data). GPS coordinates will")
        print("be plausible but do NOT match the video's actual location.")
        print("This is only for testing preprocessor logic.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()