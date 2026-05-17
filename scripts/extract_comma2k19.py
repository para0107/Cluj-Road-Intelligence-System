"""
scripts/extract_comma2k19.py
-----------------------------
Road Infrastructure Detection System (RIDS)

Extracts video.hevc and global_pose/ files from Comma2k19 chunk zips.
Fully resumable — existing non-empty files are skipped.

Handles:
- Windows-illegal characters in paths (pipe | replaced with _)
- Zip directory entries (entries ending with / are skipped)
- Already-existing empty directories from previous broken runs
- FileExistsError on mkdir when directory already exists as empty folder

Usage:
    python scripts/extract_comma2k19.py
    python scripts/extract_comma2k19.py --verbose

Author: Paraschiv Tudor - Babes-Bolyai University, 2026
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMMA2K19_ROOT = Path(
    r"C:\Facultate\pothole-detection\Pothole-Detection"
    r"\data\datasets\Comma2k19"
)
ZIP_DIR  = COMMA2K19_ROOT   # zips are directly here: Chunk_1.zip ... Chunk_10.zip
DEST_DIR = COMMA2K19_ROOT   # extract into: Chunk_1/ ... Chunk_10/

CHUNK_ZIPS = [f"Chunk_{i}.zip" for i in range(1, 11)]
CHUNK_SIZE = 1024 * 1024    # 1 MB read buffer


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level    = level,
        format   = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
        force    = True,
    )


logger = logging.getLogger("rids.extract")


def _sanitise(part: str) -> str:
    """Replace Windows-illegal characters in a single path component."""
    for ch in '|*?"<>':
        part = part.replace(ch, "_")
    return part


def _safe_makedirs(path: Path) -> None:
    """
    Create directory and all parents.

    Handles three cases that occur with Comma2k19 on Windows:
    1. Directory already exists — fine, continue (exist_ok=True).
    2. A zero-byte FILE exists at the path — this happens when a previous
       broken extraction run wrote the zip directory entry ('global_pose/')
       as a zero-byte file instead of a directory. Remove it, then mkdir.
    3. Any other OSError — re-raise.
    """
    # Walk each component of the path and fix any zero-byte files
    # that are blocking directory creation
    check = Path(path.anchor)
    for part in path.parts[1:]:
        check = check / part
        if check.exists() and not check.is_dir():
            # It's a file (likely zero-byte from broken extraction)
            if check.stat().st_size == 0:
                check.unlink()
            else:
                raise OSError(
                    f"Cannot create directory: '{check}' exists as a non-empty file"
                )

    try:
        os.makedirs(str(path), exist_ok=True)
    except FileExistsError:
        pass
    except OSError as exc:
        if exc.errno != 17:  # errno 17 = EEXIST
            raise


def extract_all(verbose: bool) -> List[Path]:
    """
    Streams each Chunk_N.zip entry by entry.
    Writes only video.hevc and global_pose/* to DEST_DIR.
    Skips zip directory entries (ending with /).
    Skips files already on disk with size > 0.
    Returns list of segment directories containing video.hevc.
    DATA NOTE: reads actual zip files from disk — no mocks.
    """

    for zip_name in CHUNK_ZIPS:
        zip_path = ZIP_DIR / zip_name

        if not zip_path.exists():
            logger.warning("Zip not found, skipping: %s", zip_path)
            continue

        size_gb = zip_path.stat().st_size / 1024 ** 3
        logger.info("=" * 60)
        logger.info("Extracting %s  (%.2f GB)", zip_name, size_gb)
        t0      = time.time()
        n_video = 0
        n_gps   = 0
        n_skip  = 0
        n_dir   = 0

        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                entries = zf.namelist()
                logger.info("  %d entries in archive.", len(entries))

                for entry in entries:
                    entry_norm = entry.replace("\\", "/")

                    # --------------------------------------------------
                    # Skip directory entries — zip archives store folder
                    # markers as entries ending with '/'. These are not
                    # files and must not be written to disk.
                    # --------------------------------------------------
                    if entry_norm.endswith("/"):
                        n_dir += 1
                        continue

                    # --------------------------------------------------
                    # Filter: only video.hevc and global_pose/* files
                    # global_pose contains: frame_positions, frame_times,
                    # frame_orientations, frame_velocities, frame_gps_times
                    # --------------------------------------------------
                    want = (
                        entry_norm.endswith("video.hevc")
                        or "/global_pose/" in entry_norm
                    )
                    if not want:
                        continue

                    # --------------------------------------------------
                    # Sanitise path — replace Windows-illegal chars
                    # --------------------------------------------------
                    parts     = entry_norm.split("/")
                    safe_norm = "/".join(_sanitise(p) for p in parts)
                    out_path  = DEST_DIR / safe_norm

                    # --------------------------------------------------
                    # Skip already-extracted non-empty files (resumable)
                    # --------------------------------------------------
                    if out_path.exists() and out_path.stat().st_size > 0:
                        n_skip += 1
                        if verbose:
                            logger.debug("  SKIP: %s", safe_norm)
                        continue

                    # --------------------------------------------------
                    # Create parent directory — tolerates already-existing
                    # empty directories from previous broken runs
                    # --------------------------------------------------
                    _safe_makedirs(out_path.parent)

                    # --------------------------------------------------
                    # Stream entry to disk in 1 MB chunks
                    # --------------------------------------------------
                    with zf.open(entry) as src, out_path.open("wb") as dst:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)

                    if entry_norm.endswith("video.hevc"):
                        n_video += 1
                        if verbose:
                            logger.debug("  VIDEO: %s", safe_norm)
                    else:
                        n_gps += 1
                        if verbose:
                            logger.debug("  GPS:   %s", safe_norm)

        except zipfile.BadZipFile as exc:
            logger.error("Bad zip %s: %s — skipping.", zip_name, exc)
            continue
        except Exception as exc:
            logger.error("Error extracting %s: %s", zip_name, exc)
            logger.debug(traceback.format_exc())
            continue

        elapsed = time.time() - t0
        logger.info(
            "  Done in %.0f s | videos=%d | GPS files=%d | skipped=%d | dir entries=%d",
            elapsed, n_video, n_gps, n_skip, n_dir,
        )

    # Collect all segment dirs that now contain video.hevc
    logger.info("=" * 60)
    logger.info("Scanning for extracted segments ...")
    segments: List[Path] = []
    total_bytes = 0

    for chunk_dir in sorted(DEST_DIR.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("Chunk"):
            continue
        for route_dir in sorted(chunk_dir.iterdir()):
            if not route_dir.is_dir():
                continue
            for seg_dir in sorted(route_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                video = seg_dir / "video.hevc"
                if not video.exists():
                    continue
                total_bytes += video.stat().st_size
                segments.append(seg_dir)

    # Count GPS coverage
    n_with_gps = sum(
        1 for s in segments
        if (s / "global_pose" / "frame_positions").exists()
    )

    logger.info(
        "Extraction complete: %d segments | %.2f GB video | %d with GPS (%.1f%%)",
        len(segments),
        total_bytes / 1024 ** 3,
        n_with_gps,
        100 * n_with_gps / max(1, len(segments)),
    )
    return segments


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    _setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("RIDS — Comma2k19 extraction")
    logger.info("Zip source : %s", ZIP_DIR)
    logger.info("Destination: %s", DEST_DIR)
    logger.info("Extracting : video.hevc + global_pose/ only")
    logger.info("Resumable  : yes (existing non-empty files skipped)")
    logger.info("=" * 60)

    if not ZIP_DIR.exists():
        logger.error("Zip directory not found: %s", ZIP_DIR)
        sys.exit(1)

    found = [ZIP_DIR / z for z in CHUNK_ZIPS if (ZIP_DIR / z).exists()]
    logger.info("Found %d / 10 zip files.", len(found))

    if not found:
        logger.error("No Chunk_N.zip files found in %s", ZIP_DIR)
        sys.exit(1)

    segments = extract_all(verbose)

    n_with_gps = sum(
        1 for s in segments
        if (s / "global_pose" / "frame_positions").exists()
    )

    if n_with_gps == 0:
        logger.error(
            "GPS extraction failed — 0 segments have global_pose/frame_positions. "
            "Check the zip entries with inspect_zip_entries.py."
        )
    elif n_with_gps < len(segments):
        logger.warning(
            "%d / %d segments are missing GPS. "
            "These will run pipeline stages 1-5 only (no DBSCAN, no DB write).",
            len(segments) - n_with_gps, len(segments),
        )
    else:
        logger.info("All %d segments have GPS. Ready.", len(segments))
        logger.info(
            "Run pipeline: python scripts/download_comma2k19_selective.py "
            "--skip_download --device cuda"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RIDS] Interrupted.", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[RIDS] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)