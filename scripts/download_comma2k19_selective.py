"""
scripts/download_comma2k19_selective.py
---------------------------------------
Road Infrastructure Detection System (RIDS)

Downloads the Comma2k19 dataset from HuggingFace over plain HTTPS,
then extracts only video.hevc and global_pos/ from each chunk zip,
then runs the full RIDS inference pipeline on each segment.

=======================================================================
SOURCE
=======================================================================

HuggingFace repository (official, MIT licensed):
    https://huggingface.co/datasets/commaai/comma2k19

Raw chunk zips are at:
    raw_data/Chunk_1.zip  ...  raw_data/Chunk_10.zip

Total: ~94.6 GB. Extraction keeps only video.hevc + global_pos/,
reducing on-disk size to approximately 80-85 GB.

=======================================================================
WHY HUGGINGFACE INSTEAD OF BITTORRENT
=======================================================================

The Academic Torrents version uses UDP/BitTorrent which is blocked on
most university networks. HuggingFace serves the same files over plain
HTTPS (port 443) which is never blocked.

=======================================================================
DATASET STRUCTURE (inside each Chunk_N.zip)
=======================================================================

    Chunk_N/
        <dongle_id|start_time>/
            <segment_number>/
                video.hevc          <- EXTRACTED  (~30-50 MB each)
                global_pos/         <- EXTRACTED  (KB, GPS arrays)
                    frame_positions <- ECEF (m), shape (N,3)
                    frame_times     <- boot timestamps (s), shape (N,)
                raw_log.bz2         <- SKIPPED
                processed_log/      <- SKIPPED
                preview.png         <- SKIPPED

=======================================================================
GPS  —  ECEF to WGS84
=======================================================================

Comma2k19 stores GPS as ECEF coordinates in metres. Converted to
WGS84 (lat/lon) using the Zhu (1994) iterative formula.

Reference: Zhu, J. (1994). IEEE Transactions on Aerospace and
Electronic Systems, 30(3), 957-961.
https://doi.org/10.1109/7.303772

=======================================================================
INSTALL
=======================================================================

    pip install huggingface_hub numpy python-dotenv

=======================================================================
USAGE
=======================================================================

    # Download all chunks + extract (no pipeline)
    python scripts/download_comma2k19_selective.py --skip_pipeline

    # Download + extract + run RIDS pipeline
    python scripts/download_comma2k19_selective.py --device cuda

    # Already downloaded and extracted — pipeline only
    python scripts/download_comma2k19_selective.py --skip_download --device cuda

    # No DB writes
    python scripts/download_comma2k19_selective.py --dry_run_db --device cuda

    # Verbose logging
    python scripts/download_comma2k19_selective.py --verbose

=======================================================================
REFERENCE
=======================================================================

Schafer H., Santana E., Haden A., Biasini R. (2018).
"A Commute in Data: The comma2k19 Dataset."
arXiv:1812.05752.  https://arxiv.org/abs/1812.05752

Author: Paraschiv Tudor - Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# dotenv — soft import; non-fatal if missing
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging — identical format to orchestrator so log files are comparable
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
_LOG_FILE   = os.environ.get("LOG_FILE",  "")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else getattr(logging, _LOG_LEVEL, logging.INFO)
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if _LOG_FILE:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))
    logging.basicConfig(
        level    = level,
        format   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = handlers,
        force    = True,
    )


logger = logging.getLogger("rids.comma2k19_dl")

# ---------------------------------------------------------------------------
# HuggingFace dataset details
# Confirmed from: https://huggingface.co/datasets/commaai/comma2k19/tree/main/raw_data
# All 10 chunk zips are present, sizes match Academic Torrents exactly.
# ---------------------------------------------------------------------------
HF_REPO_ID  = "commaai/comma2k19"
HF_REPO_TYPE = "dataset"

# Confirmed file list and sizes from HuggingFace (actual, not estimated):
#   Chunk_1.zip   8.73 GB
#   Chunk_2.zip   9.05 GB
#   Chunk_3.zip   9.41 GB
#   Chunk_4.zip   9.49 GB
#   Chunk_5.zip   9.81 GB
#   Chunk_6.zip   9.53 GB
#   Chunk_7.zip   9.29 GB
#   Chunk_8.zip   9.63 GB
#   Chunk_9.zip   9.77 GB
#   Chunk_10.zip  9.90 GB
CHUNK_NAMES = [f"raw_data/Chunk_{i}.zip" for i in range(1, 11)]

# ---------------------------------------------------------------------------
# WGS84 constants — standard geodetic values.
# Reference: Zhu (1994) https://doi.org/10.1109/7.303772
# ---------------------------------------------------------------------------
_WGS84_A  = 6_378_137.0
_WGS84_F  = 1.0 / 298.257_223_563
_WGS84_B  = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2


# ---------------------------------------------------------------------------
# ECEF -> WGS84  (Zhu 1994 iterative formula)
# ---------------------------------------------------------------------------
def _ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF (m) to WGS84 (lat_deg, lon_deg, alt_m).
    Reference: Zhu (1994) https://doi.org/10.1109/7.303772
    DATA NOTE: x, y, z are actual frame_positions values read from disk.
    """
    lon = math.atan2(y, x)
    p   = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N       = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat_new = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = (
        (p / cos_lat - N)
        if abs(cos_lat) > 1e-10
        else (abs(z) / abs(sin_lat) - N * (1.0 - _WGS84_E2))
    )
    return math.degrees(lat), math.degrees(lon), alt


# ---------------------------------------------------------------------------
# GPX writer — reads actual global_pos/ numpy files from disk
# ---------------------------------------------------------------------------
def _write_gpx_from_global_pos(
    pose_dir: Path, out_gpx: Path, route_name: str = ""
) -> "Tuple[bool, Optional[datetime]]":
    """
    Reads global_pose/frame_positions (ECEF, shape N x 3) and
    global_pose/frame_times (boot-time s, shape N) from actual extracted
    files, converts ECEF to WGS84, writes a GPX 1.1 file.

    frame_times in Comma2k19 are boot-time seconds (seconds since device
    boot), NOT Unix timestamps. Values are typically in the range 0-86400.
    Using datetime.fromtimestamp() on them produces 1970-01-01 dates which
    causes the preprocessor to compute a negative video duration.

    Fix: extract the date from the route name (e.g. 2018-07-27--06-03-57)
    and use it as the base date. The boot-time seconds then become the
    intra-day offset, producing realistic GPX timestamps that the
    preprocessor can use correctly.

    Returns (True, gpx_start_datetime) on success,
            (False, None) if files are missing or malformed.
    DATA NOTE: reads actual files from disk — no mocks.
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy not installed. Run: pip install numpy")
        sys.exit(1)

    def _load(name: str) -> Optional[object]:
        for candidate in (
            pose_dir / name,
            Path(str(pose_dir / name) + ".npy"),
        ):
            if candidate.exists():
                try:
                    return np.load(str(candidate), allow_pickle=False)
                except Exception as exc:
                    logger.warning("Cannot load %s: %s", candidate, exc)
        return None

    positions = _load("frame_positions")   # ECEF metres  (N, 3)
    times     = _load("frame_times")       # boot-time s  (N,)

    if positions is None:
        logger.warning(
            "global_pose/frame_positions not found in %s — no GPS for this segment",
            pose_dir,
        )
        return False, None

    if positions.ndim != 2 or positions.shape[1] < 3:
        logger.warning(
            "frame_positions unexpected shape %s in %s", positions.shape, pose_dir
        )
        return False, None

    # ------------------------------------------------------------------
    # Derive base date from route name: e.g. "99c94dc769b5d96e_2018-07-27--06-03-57"
    # The date part is after the first underscore.
    # Boot-time seconds are used as intra-day offsets from midnight UTC
    # on that date, producing valid absolute timestamps the preprocessor
    # can use to compute video duration correctly.
    # ------------------------------------------------------------------
    base_ts = 0.0   # Unix timestamp of midnight UTC on the recording date
    try:
        # Route name format: <dongle_id>_<YYYY-MM-DD>--<HH-MM-SS>
        date_part = route_name.split("_", 1)[-1]   # "2018-07-27--06-03-57"
        date_str  = date_part.split("--")[0]         # "2018-07-27"
        base_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        base_ts   = base_date.timestamp()
    except Exception:
        # Fallback: use 2018-01-01 as a reasonable Comma2k19 base date
        base_ts = datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp()

    # Normalise boot times so they start from 0 within the segment
    if times is not None and len(times) > 0:
        t0 = float(times[0])
        times_normalised = [float(t) - t0 for t in times]
    else:
        times_normalised = [float(i) / 10.0 for i in range(len(positions))]

    out_gpx.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="RIDS-Comma2k19"',
        '     xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk><trkseg>",
    ]
    n = len(positions)
    gpx_start: Optional[datetime] = None
    for i in range(n):
        x, y, z       = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        lat, lon, alt = _ecef_to_wgs84(x, y, z)
        abs_ts = base_ts + times_normalised[i]
        dt     = datetime.fromtimestamp(abs_ts, tz=timezone.utc)
        if i == 0:
            gpx_start = dt
        iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        lines.append(
            f'    <trkpt lat="{lat:.8f}" lon="{lon:.8f}">'
            f"<ele>{alt:.2f}</ele><time>{iso}</time></trkpt>"
        )
    lines += ["  </trkseg></trk>", "</gpx>"]
    with out_gpx.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("GPX written: %s (%d trackpoints, start=%s)", out_gpx, n, gpx_start)
    return True, gpx_start


# ---------------------------------------------------------------------------
# HuggingFace downloader
# ---------------------------------------------------------------------------
def _download_zips(dest: Path) -> List[Path]:
    """
    Downloads all 10 Comma2k19 chunk zips from HuggingFace over HTTPS.
    Uses huggingface_hub.hf_hub_download() which handles resumable
    downloads, progress bars, and caching automatically.

    Files are saved directly to dest/Chunk_N.zip.
    Returns list of zip file paths on disk.

    DATA NOTE: downloads actual files from HuggingFace CDN — no mocks.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed.\n"
            "  Run: pip install huggingface_hub"
        )
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    zip_paths: List[Path] = []

    logger.info(
        "Downloading %d chunk zips from HuggingFace (HTTPS, ~94.6 GB total).",
        len(CHUNK_NAMES),
    )
    logger.info("Repository: https://huggingface.co/datasets/%s", HF_REPO_ID)
    logger.info(
        "Each zip is ~9 GB. Progress is shown per file by huggingface_hub."
    )

    for i, hf_path in enumerate(CHUNK_NAMES, start=1):
        chunk_name = Path(hf_path).name   # e.g. Chunk_1.zip
        local_path = dest / chunk_name

        if local_path.exists():
            size_gb = local_path.stat().st_size / 1024 ** 3
            logger.info(
                "[%d/%d] %s already exists (%.2f GB) — skipping download.",
                i, len(CHUNK_NAMES), chunk_name, size_gb,
            )
            zip_paths.append(local_path)
            continue

        logger.info(
            "[%d/%d] Downloading %s ...", i, len(CHUNK_NAMES), chunk_name
        )
        t0 = time.time()
        try:
            # hf_hub_download downloads to HF cache by default.
            # local_dir overrides this to save directly into dest/.
            downloaded = hf_hub_download(
                repo_id   = HF_REPO_ID,
                repo_type = HF_REPO_TYPE,
                filename  = hf_path,
                local_dir = str(dest),
            )
            elapsed  = time.time() - t0
            size_gb  = Path(downloaded).stat().st_size / 1024 ** 3
            speed_mb = (size_gb * 1024) / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %s downloaded: %.2f GB in %.0f s (%.1f MB/s)",
                i, len(CHUNK_NAMES), chunk_name, size_gb, elapsed, speed_mb,
            )
            # hf_hub_download with local_dir saves to dest/raw_data/Chunk_N.zip
            # Normalise: move to dest/Chunk_N.zip if nested
            downloaded_path = Path(downloaded)
            if downloaded_path.parent != dest:
                target = dest / chunk_name
                downloaded_path.rename(target)
                downloaded_path = target
            zip_paths.append(downloaded_path)
        except Exception as exc:
            logger.error(
                "Failed to download %s: %s\n"
                "Check your internet connection and try again. "
                "The script will re-use any partially completed downloads.",
                chunk_name, exc,
            )
            logger.debug(traceback.format_exc())
            # Continue with other chunks — do not abort entire run
            continue

    logger.info(
        "Download phase complete: %d / %d zips on disk.",
        len(zip_paths), len(CHUNK_NAMES),
    )
    return zip_paths


# ---------------------------------------------------------------------------
# Selective zip extractor
# ---------------------------------------------------------------------------
def _extract_zips(zip_paths: List[Path], dest: Path) -> List[Path]:
    """
    Extracts only video.hevc and global_pos/* from each Chunk_N.zip,
    streaming entry by entry so unwanted files (raw_log.bz2, processed_log/,
    preview.png) are never written to disk.

    Skips entries that already exist on disk — fully resumable.

    Returns list of all segment directories that contain video.hevc.
    DATA NOTE: reads actual zip files — no mocks.
    """
    CHUNK_SIZE = 1024 * 1024   # 1 MB read buffer

    for zip_path in zip_paths:
        logger.info("Extracting %s ...", zip_path.name)
        t0 = time.time()
        n_video = 0
        n_gps   = 0
        n_skip  = 0

        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                entries = zf.namelist()
                logger.info("  %s contains %d entries.", zip_path.name, len(entries))

                for entry in entries:
                    entry_norm = entry.replace("\\", "/")

                    want = (
                        entry_norm.endswith("video.hevc")
                        or "/global_pose/" in entry_norm
                    )
                    if not want:
                        continue

                    out_path = dest / entry_norm

                    # Skip already-extracted files (resumable)
                    if out_path.exists() and out_path.stat().st_size > 0:
                        n_skip += 1
                        continue

                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(entry) as src, out_path.open("wb") as dst:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)

                    if entry_norm.endswith("video.hevc"):
                        n_video += 1
                    else:
                        n_gps += 1

        except zipfile.BadZipFile as exc:
            logger.error("Bad zip %s: %s — skipping.", zip_path.name, exc)
            continue
        except Exception as exc:
            logger.error("Error extracting %s: %s", zip_path.name, exc)
            logger.debug(traceback.format_exc())
            continue

        logger.info(
            "  %s done in %.0f s — %d videos, %d GPS files extracted, %d already existed.",
            zip_path.name, time.time() - t0, n_video, n_gps, n_skip,
        )

    # Collect all segment dirs that contain video.hevc
    segments: List[Path] = []
    for chunk_dir in sorted(dest.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("Chunk"):
            continue
        for route_dir in sorted(chunk_dir.iterdir()):
            if not route_dir.is_dir():
                continue
            for seg_dir in sorted(route_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                if (seg_dir / "video.hevc").exists():
                    segments.append(seg_dir)

    logger.info("Total segments with video.hevc extracted: %d", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Discover already-extracted segments  (--skip_download path)
# ---------------------------------------------------------------------------
def _discover_segments(dest: Path) -> List[Path]:
    """
    Walks dest collecting every segment directory that contains video.hevc.
    Used when data is already on disk (--skip_download).
    DATA NOTE: reads actual filesystem — no mocks.
    """
    if not dest.exists():
        logger.error("Destination directory does not exist: %s", dest)
        sys.exit(1)

    segments: List[Path] = []
    total_bytes = 0

    for chunk_dir in sorted(dest.iterdir()):
        if not chunk_dir.is_dir():
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

    logger.info(
        "Discovered %d segments (%.2f GB video on disk)",
        len(segments), total_bytes / 1024 ** 3,
    )
    return segments


# ---------------------------------------------------------------------------
# Per-segment RIDS pipeline runner
# ---------------------------------------------------------------------------
def _run_segment(
    segment_dir: Path,
    work_dir:    Path,
    device:      str,
    fps:         float,
    dry_run_db:  bool,
) -> dict:
    """
    1. Converts global_pose/frame_positions (ECEF) -> WGS84 -> GPX.
    2. Runs the full RIDS orchestrator (Stages 1-7) on video.hevc.
    Returns a summary dict of real pipeline metrics — nothing mocked.
    """
    seg_name   = segment_dir.name
    video_path = segment_dir / "video.hevc"
    pose_dir   = segment_dir / "global_pose"

    if not video_path.exists():
        return {
            "segment": seg_name,
            "route":   segment_dir.parent.name,
            "status":  "skipped",
            "reason":  "no_video.hevc",
        }

    tmp_gpx = (
        work_dir / "tmp_gpx"
        / f"{segment_dir.parent.name}_{seg_name}.gpx"
    )
    has_gps, gpx_start = _write_gpx_from_global_pos(
        pose_dir, tmp_gpx, route_name=segment_dir.parent.name
    )

    # Pass empty string (not None) when no GPS — OrchestratorConfig
    # coerces None to "" internally which then fails the format check.
    gps_arg = str(tmp_gpx) if has_gps else ""

    if not has_gps:
        logger.warning(
            "Segment %s | no GPS — DBSCAN and DB write skipped by orchestrator",
            seg_name,
        )

    try:
        from pipeline.orchestrator import Orchestrator, OrchestratorConfig
    except ImportError as exc:
        logger.error(
            "Cannot import pipeline.orchestrator: %s\n"
            "Run from the RIDS project root with your venv active.",
            exc,
        )
        sys.exit(1)

    session_id = (
        f"comma2k19_{segment_dir.parent.name}_{seg_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cfg = OrchestratorConfig(
        video_path       = str(video_path),
        gps_path         = gps_arg,
        work_dir         = str(work_dir / "sessions"),
        session_id       = session_id,
        device           = device,
        fps              = fps,
        dry_run_db       = dry_run_db,
        resume           = False,
        video_start_time = gpx_start,   # tells preprocessor when frame 0 is
    )

    t0 = time.perf_counter()
    try:
        result  = Orchestrator(cfg).run()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Segment %s | %s | frames=%d | dets=%d | %.1f s",
            seg_name, result.status,
            result.n_frames, result.n_detections, elapsed,
        )
        return {
            "segment":      seg_name,
            "route":        segment_dir.parent.name,
            "video":        str(video_path),
            "has_gps":      has_gps,
            "status":       result.status,
            "n_frames":     result.n_frames,
            "n_detections": result.n_detections,
            "n_inserted":   result.n_inserted,
            "n_updated":    result.n_updated,
            "elapsed_s":    round(elapsed, 2),
            "session_id":   result.session_id,
            "error":        result.error_message,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Segment %s FAILED: %s", seg_name, exc)
        logger.debug(traceback.format_exc())
        return {
            "segment":   seg_name,
            "route":     segment_dir.parent.name,
            "video":     str(video_path),
            "has_gps":   has_gps,
            "status":    "failed",
            "elapsed_s": round(elapsed, 2),
            "error":     str(exc),
        }


# ---------------------------------------------------------------------------
# Run summary writer
# ---------------------------------------------------------------------------
def _write_run_summary(summaries: List[dict], out_dir: Path) -> Path:
    """
    Writes a JSON run summary with per-segment rows and aggregate stats.
    Primary artefact for thesis plots and cross-dataset comparison.
    DATA NOTE: all values are real pipeline outputs — nothing mocked.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comma2k19_run_summary_{ts}.json"

    total_frames = sum(s.get("n_frames",     0)   for s in summaries)
    total_dets   = sum(s.get("n_detections", 0)   for s in summaries)
    total_ins    = sum(s.get("n_inserted",   0)   for s in summaries)
    total_upd    = sum(s.get("n_updated",    0)   for s in summaries)
    elapsed_all  = sum(s.get("elapsed_s",    0.0) for s in summaries)
    n_ok         = sum(1 for s in summaries if s.get("status") == "complete")
    n_fail       = sum(1 for s in summaries if s.get("status") == "failed")
    n_skip       = sum(1 for s in summaries if s.get("status") == "skipped")
    n_with_gps   = sum(1 for s in summaries if s.get("has_gps"))

    segs_with_frames = [s for s in summaries if s.get("n_frames", 0) > 0]
    det_rate = (
        sum(1 for s in segs_with_frames if s.get("n_detections", 0) > 0)
        / max(1, len(segs_with_frames))
    )

    aggregate = {
        "dataset":            "Comma2k19",
        "reference":          "Schafer et al. 2018 — arXiv:1812.05752",
        "source":             f"https://huggingface.co/datasets/{HF_REPO_ID}",
        "run_timestamp":      ts,
        "n_segments_total":   len(summaries),
        "n_complete":         n_ok,
        "n_failed":           n_fail,
        "n_skipped":          n_skip,
        "n_with_gps":         n_with_gps,
        "total_frames":       total_frames,
        "total_detections":   total_dets,
        "total_inserted":     total_ins,
        "total_updated":      total_upd,
        "total_elapsed_s":    round(elapsed_all, 2),
        "detection_rate_pct": round(det_rate * 100, 2),
        "interpretation_note": (
            "Comma2k19 is an unlabelled dashcam dataset (California I-280 highway). "
            "No road-damage ground truth exists. Detection rate reflects raw RIDS "
            "inference on unannotated footage — directly comparable to Cluj Run 1 "
            "(~4%) and Tokyo validation runs. Low rates expected due to domain gap "
            "(N-RDD2024 model trained on Japan/India/China/Norway/Czech/USA roads)."
        ),
        "gps_note": (
            "GPS converted from ECEF frame_positions to WGS84 using "
            "Zhu (1994) iterative formula. "
            "Reference: https://doi.org/10.1109/7.303772"
        ),
    }

    payload = {"aggregate": aggregate, "segments": summaries}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Run summary -> %s", out_path)
    logger.info(
        "AGGREGATE | segs=%d | ok=%d | fail=%d | skip=%d | gps=%d | "
        "frames=%d | dets=%d | det_rate=%.1f%%",
        len(summaries), n_ok, n_fail, n_skip,
        n_with_gps, total_frames, total_dets, det_rate * 100,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RIDS — Comma2k19 downloader (HuggingFace HTTPS) + pipeline runner.\n"
            "Downloads all 10 Chunk_N.zip files (~94.6 GB) from HuggingFace,\n"
            "extracts only video.hevc + global_pos/, then runs RIDS Stages 1-7.\n\n"
            "Source : https://huggingface.co/datasets/commaai/comma2k19\n"
            "Paper  : Schafer et al. 2018, arXiv:1812.05752"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default=(
            r"C:\Facultate\pothole-detection\Pothole-Detection"
            r"\data\datasets\Comma2k19"
        ),
        help="Root directory for downloaded zips and extracted data.",
    )
    parser.add_argument(
        "--work_dir",
        default=r"data\processed\comma2k19",
        help="Working directory for session outputs and temporary GPX files.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and extraction. Discovers already-extracted segments.",
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Download and extract only — do not run the RIDS pipeline.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto | cuda | cpu  (default: auto).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate for Stage 1 preprocessor (default: 2.0).",
    )
    parser.add_argument(
        "--dry_run_db",
        action="store_true",
        help="Do not write to PostgreSQL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args     = _parse_args()
    _setup_logging(args.verbose)

    dest     = Path(args.dest)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RIDS — Comma2k19 download (HuggingFace) + inference pipeline")
    logger.info("Dataset destination : %s", dest)
    logger.info("Work directory      : %s", work_dir)
    logger.info("HuggingFace repo    : https://huggingface.co/datasets/%s", HF_REPO_ID)
    logger.info("Skip download       : %s", args.skip_download)
    logger.info("Skip pipeline       : %s", args.skip_pipeline)
    logger.info("Device              : %s", args.device)
    logger.info("FPS                 : %.1f", args.fps)
    logger.info("Dry-run DB          : %s", args.dry_run_db)
    logger.info("Reference           : Schafer et al. 2018 — arXiv:1812.05752")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 0 — dependency check
    # ------------------------------------------------------------------
    logger.info("[Step 0] Checking dependencies ...")
    try:
        import numpy  # noqa: F401
        logger.info("[Step 0] numpy OK.")
    except ImportError:
        logger.error("numpy not installed. Run: pip install numpy")
        sys.exit(1)

    if not args.skip_download:
        try:
            import huggingface_hub  # noqa: F401
            logger.info("[Step 0] huggingface_hub OK.")
        except ImportError:
            logger.error(
                "huggingface_hub not installed.\n"
                "  Run: pip install huggingface_hub"
            )
            sys.exit(1)

    logger.info("[Step 0] All dependencies OK.")

    # ------------------------------------------------------------------
    # Step 1 — Download + extract, or discover existing
    # ------------------------------------------------------------------
    if args.skip_download:
        logger.info("[Step 1] Skipping download — discovering existing segments ...")
        segments = _discover_segments(dest)
    else:
        logger.info("[Step 1a] Downloading chunk zips from HuggingFace ...")
        zip_paths = _download_zips(dest)

        if not zip_paths:
            logger.error("No zip files downloaded. Check your internet connection.")
            sys.exit(1)

        logger.info("[Step 1b] Extracting video.hevc + global_pos/ from zips ...")
        segments = _extract_zips(zip_paths, dest)

    if not segments:
        logger.error("No segments found. Exiting.")
        sys.exit(1)

    logger.info("[Step 1] %d segments ready.", len(segments))

    if args.skip_pipeline:
        logger.info("[Step 2] --skip_pipeline set — download/extract-only mode. Done.")
        return

    # ------------------------------------------------------------------
    # Step 2 — Run RIDS pipeline segment by segment
    # ------------------------------------------------------------------
    logger.info(
        "[Step 2] Running RIDS inference pipeline on %d segments ...",
        len(segments),
    )
    summaries: List[dict] = []

    for idx, seg_dir in enumerate(segments, start=1):
        logger.info(
            "[Step 2] Segment %d / %d | route=%s | seg=%s",
            idx, len(segments), seg_dir.parent.name, seg_dir.name,
        )
        summary = _run_segment(
            segment_dir = seg_dir,
            work_dir    = work_dir,
            device      = args.device,
            fps         = args.fps,
            dry_run_db  = args.dry_run_db,
        )
        summaries.append(summary)

        # Incremental save after every segment — crash-safe
        partial = work_dir / "comma2k19_partial.json"
        with partial.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        logger.debug("Partial summary updated -> %s", partial)

    # ------------------------------------------------------------------
    # Step 3 — Final summary
    # ------------------------------------------------------------------
    logger.info("[Step 3] Writing final run summary ...")
    summary_path = _write_run_summary(summaries, work_dir / "reports")

    logger.info("=" * 70)
    logger.info("Done. Summary: %s", summary_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RIDS] Interrupted by user (Ctrl+C).", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[RIDS] FATAL ERROR: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
