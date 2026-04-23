"""
scripts/download_validation_footage.py
----------------------------------------
Downloads the validation video for the CRIS pipeline.

Video: "4K City Drive Cluj-Napoca Part 1 - Cluj Transilvania Romania 4K"
Channel: RC Scenic Drives
URL: https://www.youtube.com/watch?v=yYO9iT6glnU
Published: November 2025

Why this video:
  - Shot from inside a car driving through Cluj-Napoca streets
  - Correct windshield-mounted dashcam perspective
  - Your exact target city — same road surfaces, markings, lighting
  - Daytime, multiple street types (centre, residential, arterial)
  - Recent (2025), so road condition is current

The video is 4K — we download at 720p to keep file size and
preprocessing speed reasonable. The detector runs at 640x640
regardless of input resolution, so quality is not lost.

Usage:
    python scripts/download_validation_footage.py

Then run the full pipeline:

  Step 1 - Preprocess:
    python -m pipeline.preprocessor
        --video  data/raw/footage/validation_cluj.mp4
        --gps    data/raw/gps_logs/bdd_sample.json
        --output data/processed/frames/validation/
        --fps 2 --min-brightness 15 --verbose

  Step 2 - Detect (requires ml/weights/rtdetr_l_rdd2022.pt):
    python pipeline/detector.py
        --manifest data/processed/frames/validation/manifest.json
        --weights  ml/weights/rtdetr_l_rdd2022.pt
        --output   data/processed/detections/validation/
        --conf 0.35 --device cpu --verbose

  Step 3 - Inspect detections:
    python scripts/inspect_detector.py
        --detections data/processed/detections/validation/detections.json
        --output-dir data/processed/inspection/detector/
        --no-display

Requirements:
    yt-dlp  : pip install yt-dlp
    Node.js : https://nodejs.org  (needed by yt-dlp for YouTube)
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
VIDEO_OUT = ROOT / "data" / "raw" / "footage" / "validation_cluj.mp4"

VIDEO_URL = "https://www.youtube.com/watch?v=yYO9iT6glnU"

# Fallback: search for any recent Cluj driving video
FALLBACK_QUERY = "ytsearch1:Cluj-Napoca city drive car 2024 2025"


def check_ytdlp() -> bool:
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        logger.info("yt-dlp version: %s", result.stdout.strip())
        return True
    except FileNotFoundError:
        logger.error(
            "yt-dlp not found.\n"
            "Install with: pip install yt-dlp\n"
            "Also install Node.js from https://nodejs.org for YouTube support."
        )
        return False


def download_video(url: str, out_path: Path) -> bool:
    """Download at 720p max, falling back to best available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in ["best[ext=mp4][height<=720]", "best[height<=720]", "best"]:
        cmd = [
            "yt-dlp",
            "-f", fmt,
            url,
            "-o", str(out_path),
            "--force-overwrites",
            "--no-playlist",
            "--merge-output-format", "mp4",
        ]
        logger.info("Trying format '%s': %s", fmt, url)
        result = subprocess.run(cmd, text=True)
        if result.returncode == 0 and out_path.exists():
            return True

    return False


def main() -> None:
    logger.info("=== CRIS validation footage downloader ===")
    logger.info("Video : 4K City Drive Cluj-Napoca Part 1")
    logger.info("URL   : %s", VIDEO_URL)
    logger.info("Output: %s", VIDEO_OUT)

    if not check_ytdlp():
        sys.exit(1)

    if VIDEO_OUT.exists():
        size_mb = VIDEO_OUT.stat().st_size / 1e6
        logger.info(
            "Already present: %s (%.1f MB) -- delete to re-download",
            VIDEO_OUT.name, size_mb,
        )
    else:
        success = download_video(VIDEO_URL, VIDEO_OUT)

        if not success:
            logger.warning("Primary URL failed -- trying fallback search")
            success = download_video(FALLBACK_QUERY, VIDEO_OUT)

        if not success or not VIDEO_OUT.exists():
            logger.error(
                "Automatic download failed.\n\n"
                "Manual steps:\n"
                "  1. Open: %s\n"
                "  2. Download at 720p using any YouTube downloader\n"
                "  3. Save as: %s\n\n"
                "If yt-dlp gives a JS warning, install Node.js:\n"
                "  https://nodejs.org/en/download",
                VIDEO_URL, VIDEO_OUT,
            )
            sys.exit(1)

        size_mb = VIDEO_OUT.stat().st_size / 1e6
        logger.info("Downloaded: %s (%.1f MB)", VIDEO_OUT.name, size_mb)

    print("\n" + "=" * 66)
    print("  Video ready. Run the full validation pipeline:")
    print("=" * 66)
    print()
    print("  1) Preprocess:")
    print(f"     python -m pipeline.preprocessor \\")
    print(f"         --video  {VIDEO_OUT} \\")
    print(f"         --gps    data/raw/gps_logs/bdd_sample.json \\")
    print(f"         --output data/processed/frames/validation/ \\")
    print(f"         --fps 2 --min-brightness 15 --verbose")
    print()
    print("  2) Detect:")
    print(f"     python pipeline/detector.py \\")
    print(f"         --manifest data/processed/frames/validation/manifest.json \\")
    print(f"         --weights  ml/weights/rtdetr_l_rdd2022.pt \\")
    print(f"         --output   data/processed/detections/validation/ \\")
    print(f"         --conf 0.35 --device cpu --verbose")
    print()
    print("  3) Inspect:")
    print(f"     python scripts/inspect_detector.py \\")
    print(f"         --detections data/processed/detections/validation/detections.json \\")
    print(f"         --output-dir data/processed/inspection/detector/ \\")
    print(f"         --no-display")
    print()
    print("  NOTE: Step 2 requires ml/weights/rtdetr_l_rdd2022.pt")
    print("  Download from Kaggle dataset 'cluj-road-weights'.")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()