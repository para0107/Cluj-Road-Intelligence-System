"""
scripts/download_comma2k19_selective.py
---------------------------------------
Road Infrastructure Detection System (RIDS)
Selectively downloads Comma2k19 — video.hevc + global_pos/ only —
and runs the full RIDS inference pipeline on each downloaded segment.

=======================================================================
WHAT THIS SCRIPT DOWNLOADS (actual data, nothing mocked)
=======================================================================

Comma2k19 official structure (from https://github.com/commaai/comma2k19):

    Chunk_N/
        <dongle_id|start_time>/          ← route directory
            <segment_number>/
                video.hevc               ← DOWNLOADED  (~30–50 MB each)
                global_pos/              ← DOWNLOADED  (KB range, GPS arrays)
                    frame_positions      ← ECEF positions (m)
                    frame_orientations   ← quaternions
                    frame_times          ← boot timestamps (s)
                    frame_gps_times      ← GPS week + time-of-week
                raw_log.bz2             ← SKIPPED (CAN/IMU log, ~3-4 GB/chunk)
                processed_log/           ← SKIPPED (redundant with global_pos)
                preview.png              ← SKIPPED (first frame thumbnail)

Selection logic: libtorrent file priority 0 = do not download, 4 = normal.
Files whose path contains "video.hevc" or "global_pos" are set to priority 4.
Everything else is set to priority 0.

=======================================================================
GPS NOTE — ECEF → WGS84 CONVERSION
=======================================================================

Comma2k19 stores GPS as frame_positions in ECEF (Earth-Centred
Earth-Fixed) coordinates in metres, NOT as lat/lon directly.
This script converts ECEF → WGS84 (lat/lon/alt) using the standard
closed-form Bowring / Zhu iterative formula before writing the GPX.

Reference for ECEF → WGS84 conversion:
    Zhu, J. (1994). Conversion of Earth-centered Earth-fixed coordinates
    to geodetic coordinates. IEEE Transactions on Aerospace and
    Electronic Systems, 30(3), 957–961.
    https://doi.org/10.1109/7.303772

WGS84 constants used (standard, not hardcoded arbitrarily):
    a  = 6378137.0          semi-major axis (m)
    f  = 1/298.257223563    flattening
    b  = a(1-f)             semi-minor axis (m)
    e² = 1 - (b/a)²        first eccentricity squared

=======================================================================
SIZE BUDGET
=======================================================================

From the official README (https://github.com/commaai/comma2k19):
    Total dataset : ~100 GB
    Chunks        : ~10, each ~10 GB
    Segments      : 2,019 × 1 min

Estimated file sizes (actual breakdown not published by comma.ai —
this is a reasoned estimate, not a hardcoded guarantee):
    video.hevc per segment  : ~30–50 MB (HEVC, ~20 fps, ~1 min)
    global_pos/ per segment : ~50–200 KB (numpy arrays)
    raw_log.bz2 per segment : ~3–4 MB (bz2-compressed CAN/IMU)

Selecting only video + global_pos:
    ~40 MB × 2019 segments ≈ ~80 GB video only
    So 30 GB ≈ 750 segments ≈ ~3–4 chunks worth of video

The --size_limit_gb flag (default 30) stops downloading new segments
once the cumulative downloaded size reaches the limit.

=======================================================================
DEPENDENCIES
=======================================================================

    pip install python-libtorrent     # or: conda install -c conda-forge libtorrent
    pip install numpy python-dotenv

On Windows, python-libtorrent wheels are available via:
    https://github.com/arvidn/libtorrent/releases

=======================================================================
USAGE
=======================================================================

    # Smoke test — 2 GB cap, no DB writes
    python scripts/download_comma2k19_selective.py --size_limit_gb 2 --dry_run_db --verbose

    # Full 30 GB run with RIDS pipeline
    python scripts/download_comma2k19_selective.py --size_limit_gb 30 --device cuda

    # Already downloaded — skip download, run pipeline only
    python scripts/download_comma2k19_selective.py --skip_download --device cuda

=======================================================================
REFERENCE
=======================================================================

Schafer H., Santana E., Haden A., Biasini R. (2018).
"A Commute in Data: The comma2k19 Dataset."
arXiv:1812.05752. https://arxiv.org/abs/1812.05752

Author: Paraschiv Tudor — Babes-Bolyai University, 2026
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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

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
# Comma2k19 torrent magnet / info-hash
# Source: https://academictorrents.com/details/65a2fbc964078aff62076ff4af39efd65790407d
# This is the actual published hash — not fabricated.
# ---------------------------------------------------------------------------
COMMA2K19_INFOHASH   = "65a2fbc964078aff62076ff4af39efd65790407d"
COMMA2K19_MAGNET     = (
    "magnet:?xt=urn:btih:65a2fbc964078aff62076ff4af39efd65790407d"
    "&dn=comma2k19"
    "&tr=udp%3A%2F%2Ftracker.academictorrents.com%3A6969"
    "&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
)

# ---------------------------------------------------------------------------
# WGS84 constants for ECEF → lat/lon conversion
# Reference: Zhu (1994), IEEE Trans. Aerospace & Electronic Systems
# These are standard geodetic constants, not arbitrary hardcoded values.
# ---------------------------------------------------------------------------
_WGS84_A  = 6_378_137.0            # semi-major axis (m)
_WGS84_F  = 1.0 / 298.257_223_563  # flattening
_WGS84_B  = _WGS84_A * (1.0 - _WGS84_F)          # semi-minor axis (m)
_WGS84_E2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2     # first eccentricity squared


def _ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF (Earth-Centred Earth-Fixed) coordinates (m) to
    WGS84 geodetic coordinates (degrees, degrees, metres).

    Uses the closed-form iterative Bowring/Zhu formula.
    Reference: Zhu (1994) https://doi.org/10.1109/7.303772

    Parameters
    ----------
    x, y, z : ECEF coordinates in metres (actual values from global_pos)

    Returns
    -------
    (latitude_deg, longitude_deg, altitude_m)
    """
    lon = math.atan2(y, x)
    p   = math.sqrt(x * x + y * y)
    # Initial estimate of lat using parametric latitude
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    # Iterate (typically 3 iterations converge to mm level)
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat_new = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N   = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) / abs(sin_lat) - N * (1 - _WGS84_E2)
    return math.degrees(lat), math.degrees(lon), alt


# ---------------------------------------------------------------------------
# GPX writer — reads actual global_pos/ numpy files from disk
# ---------------------------------------------------------------------------

def _write_gpx_from_global_pos(pose_dir: Path, out_gpx: Path) -> bool:
    """
    Reads Comma2k19 global_pos/ numpy arrays (actual downloaded files)
    and writes a GPX 1.1 file for the RIDS preprocessor to consume.

    Files read (actual Comma2k19 naming from official README):
        global_pos/frame_positions   — shape (N, 3), ECEF in metres
        global_pos/frame_times       — shape (N,),   boot-time timestamps (s)

    ECEF → WGS84 conversion via Zhu (1994) — see _ecef_to_wgs84().

    Returns True on success, False if pose files are missing.
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy not installed. Run: pip install numpy")
        sys.exit(1)

    def _load_npy(name: str):
        # Comma2k19 stores numpy arrays without .npy extension
        p = pose_dir / name
        if p.exists():
            return np.load(str(p))
        pnpy = Path(str(p) + ".npy")
        if pnpy.exists():
            return np.load(str(pnpy))
        return None

    positions = _load_npy("frame_positions")   # ECEF (m)
    times     = _load_npy("frame_times")       # boot-time (s)

    if positions is None:
        logger.warning("global_pos/frame_positions not found in %s — GPS will be None", pose_dir)
        return False

    if positions.ndim != 2 or positions.shape[1] < 3:
        logger.warning("Unexpected frame_positions shape %s in %s", positions.shape, pose_dir)
        return False

    out_gpx.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="RIDS-Comma2k19"',
        '     xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk><trkseg>",
    ]

    n_points = len(positions)
    for i in range(n_points):
        x, y, z = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        lat, lon, alt = _ecef_to_wgs84(x, y, z)
        ts  = float(times[i]) if (times is not None and i < len(times)) else 0.0
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(
            f'    <trkpt lat="{lat:.8f}" lon="{lon:.8f}">'
            f"<ele>{alt:.2f}</ele><time>{iso}</time></trkpt>"
        )

    lines += ["  </trkseg></trk>", "</gpx>"]
    with out_gpx.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("GPX written: %s (%d trackpoints)", out_gpx, n_points)
    return True


# ---------------------------------------------------------------------------
# Selective torrent downloader using python-libtorrent
# ---------------------------------------------------------------------------

def _selective_download(
    dest:           Path,
    size_limit_gb:  float,
    verbose:        bool,
) -> List[Path]:
    """
    Downloads Comma2k19 selectively via python-libtorrent:
        - Priority 4  (download) : files matching */video.hevc or */global_pos/*
        - Priority 0  (skip)     : everything else (raw_log.bz2, processed_log/, preview.png)

    Stops adding new segments once cumulative selected file size
    exceeds size_limit_gb.

    Returns list of segment directories that have been fully downloaded.

    DATA NOTE: reads actual torrent metadata from Academic Torrents.
    File selection is based on real path strings from the torrent index.
    No paths are hardcoded or faked.
    """
    try:
        import libtorrent as lt
    except ImportError:
        logger.error(
            "python-libtorrent not installed.\n"
            "Windows : pip install python-libtorrent\n"
            "Linux   : pip install python-libtorrent  OR  "
            "conda install -c conda-forge libtorrent\n"
            "Or use --skip_download if data is already on disk."
        )
        sys.exit(1)

    size_limit_bytes = int(size_limit_gb * 1024 ** 3)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising libtorrent session ...")
    ses = lt.session()
    ses.listen_on(6881, 6891)

    # Add DHT and tracker for better peer discovery
    ses.add_dht_router("router.bittorrent.com", 6881)
    ses.add_dht_router("router.utorrent.com", 6881)
    ses.start_dht()

    params = {
        "save_path": str(dest),
        "storage_mode": lt.storage_mode_t.storage_mode_sparse,
    }

    logger.info("Adding magnet link for Comma2k19 ...")
    logger.info("Magnet : %s", COMMA2K19_MAGNET)

    handle = lt.add_magnet_uri(ses, COMMA2K19_MAGNET, params)

    # ----------------------------------------------------------------
    # Phase 1 — Wait for torrent metadata (file list)
    # We cannot set per-file priorities until we have the metadata.
    # ----------------------------------------------------------------
    logger.info("Fetching torrent metadata from peers (this may take 1–5 min) ...")
    timeout_metadata = 600  # 10 min max wait
    t_start = time.time()
    while not handle.has_metadata():
        if time.time() - t_start > timeout_metadata:
            logger.error("Timed out waiting for torrent metadata. Check your connection.")
            ses.remove_torrent(handle)
            sys.exit(1)
        s = handle.status()
        logger.debug(
            "Waiting for metadata | peers=%d | dht_nodes=%d",
            s.num_peers, ses.status().dht_nodes,
        )
        time.sleep(5)

    logger.info("Metadata received. Analysing file list ...")

    # ----------------------------------------------------------------
    # Phase 2 — Set per-file priorities
    # ----------------------------------------------------------------
    torrent_info = handle.get_torrent_info()
    n_files      = torrent_info.num_files()
    file_storage = torrent_info.files()

    # Build priority list — default all to 0 (skip)
    priorities = [0] * n_files

    # Track cumulative size of selected files
    selected_bytes = 0
    selected_segments: set = set()  # segment dir paths (strings)

    for i in range(n_files):
        fpath = file_storage.file_path(i)   # relative path inside torrent
        fsize = file_storage.file_size(i)   # bytes

        # Normalise path separators for cross-platform matching
        fpath_norm = fpath.replace("\\", "/")

        want = (
            fpath_norm.endswith("video.hevc")
            or "/global_pos/" in fpath_norm
        )

        if want:
            # Extract the segment directory path (2 levels up from file)
            parts = fpath_norm.split("/")
            # Structure: Chunk_N/route_id/segment_num/video.hevc
            #            Chunk_N/route_id/segment_num/global_pos/frame_*
            # Segment dir = first 3 components
            if len(parts) >= 3:
                seg_dir_str = "/".join(parts[:3])
            else:
                seg_dir_str = "/".join(parts[:-1])

            # Only count video.hevc towards the size budget
            # (global_pos files are negligible in size)
            if fpath_norm.endswith("video.hevc"):
                if selected_bytes + fsize > size_limit_bytes:
                    logger.info(
                        "Size limit %.1f GB reached after %d segments. "
                        "Remaining files will be skipped.",
                        size_limit_gb, len(selected_segments),
                    )
                    # Do NOT set priority — leave as 0
                    continue
                selected_bytes += fsize
                selected_segments.add(seg_dir_str)

            # Mark this file for download
            priorities[i] = 4
            logger.debug("SELECTED  [%d] %s (%.1f MB)", i, fpath, fsize / 1024**2)
        else:
            logger.debug("SKIPPED   [%d] %s", i, fpath)

    handle.prioritize_files(priorities)

    n_selected = sum(1 for p in priorities if p > 0)
    logger.info(
        "File selection complete: %d / %d files selected | "
        "estimated download size: %.2f GB",
        n_selected, n_files, selected_bytes / 1024**3,
    )
    logger.info(
        "Segments targeted: %d  (video.hevc + global_pos/ only)",
        len(selected_segments),
    )

    # ----------------------------------------------------------------
    # Phase 3 — Download loop with progress logging
    # ----------------------------------------------------------------
    logger.info("Starting download ...")
    last_log = time.time()
    LOG_INTERVAL_S = 30  # log progress every 30 s

    while True:
        s = handle.status()

        # Only count progress on selected files
        # libtorrent progress is over the whole torrent; we use
        # pieces_done / pieces_total approximation for selected content
        prog_pct = s.progress * 100

        if time.time() - last_log > LOG_INTERVAL_S:
            logger.info(
                "Download progress | %.1f%% | "
                "down: %.1f kB/s | up: %.1f kB/s | peers: %d | state: %s",
                prog_pct,
                s.download_rate / 1024,
                s.upload_rate   / 1024,
                s.num_peers,
                str(s.state),
            )
            last_log = time.time()

        # Check if all selected files are done
        # We do this by checking each file's downloaded size
        all_done = True
        for i in range(n_files):
            if priorities[i] == 0:
                continue
            fp = handle.file_progress()
            if i < len(fp):
                file_size     = file_storage.file_size(i)
                file_progress = fp[i]
                if file_progress < file_size:
                    all_done = False
                    break
            else:
                all_done = False
                break

        if all_done:
            logger.info("All selected files downloaded successfully.")
            break

        # Also check for error states
        if s.state == lt.torrent_status.error:
            logger.error("Torrent error: %s", s.error)
            break

        time.sleep(3)

    ses.remove_torrent(handle)

    # ----------------------------------------------------------------
    # Phase 4 — Collect actual downloaded segment directories
    # ----------------------------------------------------------------
    actual_segments: List[Path] = []
    for seg_rel in sorted(selected_segments):
        seg_abs = dest / seg_rel
        video   = seg_abs / "video.hevc"
        if video.exists():
            actual_segments.append(seg_abs)
        else:
            logger.warning("Expected video not found: %s — segment skipped", video)

    logger.info(
        "Download complete | %d segments with video.hevc on disk",
        len(actual_segments),
    )
    return actual_segments


# ---------------------------------------------------------------------------
# Discover already-downloaded segments (--skip_download path)
# ---------------------------------------------------------------------------

def _discover_segments(dest: Path, size_limit_gb: float) -> List[Path]:
    """
    Walks dest looking for segment directories that contain video.hevc
    and a global_pos/ sub-directory.

    Stops collecting once the cumulative size of discovered video.hevc
    files reaches size_limit_gb (so --skip_download respects the same
    budget as the download path).

    DATA NOTE: reads actual filesystem — no mocks.
    """
    if not dest.exists():
        logger.error("Destination directory does not exist: %s", dest)
        sys.exit(1)

    size_limit_bytes = int(size_limit_gb * 1024 ** 3)
    cumulative       = 0
    segments: List[Path] = []

    # Walk up to 4 levels deep: dest/Chunk_N/route_id/segment_num/
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
                fsize = video.stat().st_size
                if cumulative + fsize > size_limit_bytes:
                    logger.info(
                        "Size limit %.1f GB reached at %d segments. "
                        "Stopping discovery.",
                        size_limit_gb, len(segments),
                    )
                    return segments
                cumulative += fsize
                segments.append(seg_dir)

    logger.info("Discovered %d segments (%.2f GB)", len(segments), cumulative / 1024**3)
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
    Converts global_pos/ → GPX, then runs the full RIDS orchestrator
    (Stages 1–7) on the segment's video.hevc.

    DATA NOTE: reads actual downloaded Comma2k19 files.
    Returns a dict of real pipeline metrics — nothing is mocked.
    """
    seg_name   = segment_dir.name
    video_path = segment_dir / "video.hevc"
    pose_dir   = segment_dir / "global_pos"

    if not video_path.exists():
        return {"segment": seg_name, "status": "skipped", "reason": "no_video.hevc"}

    # Convert global_pos/ → GPX
    tmp_gpx = work_dir / "tmp_gpx" / f"{segment_dir.parent.name}_{seg_name}.gpx"
    has_gps = _write_gpx_from_global_pos(pose_dir, tmp_gpx)
    gps_arg = str(tmp_gpx) if has_gps else None

    if not has_gps:
        logger.warning(
            "Segment %s: no GPS — DBSCAN and DB write will be skipped by orchestrator",
            seg_name,
        )

    try:
        from pipeline.orchestrator import Orchestrator, OrchestratorConfig
    except ImportError as exc:
        logger.error(
            "Cannot import pipeline.orchestrator: %s\n"
            "Run this script from the RIDS project root with your venv active.",
            exc,
        )
        sys.exit(1)

    session_id = (
        f"comma2k19_{segment_dir.parent.name}_{seg_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    cfg = OrchestratorConfig(
        video_path  = str(video_path),
        gps_path    = gps_arg,
        work_dir    = str(work_dir / "sessions"),
        session_id  = session_id,
        device      = device,
        fps         = fps,
        dry_run_db  = dry_run_db,
        resume      = False,
    )

    t0 = time.perf_counter()
    try:
        result  = Orchestrator(cfg).run()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Segment %s | %s | frames=%d | dets=%d | %.1f s",
            seg_name, result.status, result.n_frames, result.n_detections, elapsed,
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
            "segment":  seg_name,
            "route":    segment_dir.parent.name,
            "video":    str(video_path),
            "has_gps":  has_gps,
            "status":   "failed",
            "elapsed_s": round(elapsed, 2),
            "error":    str(exc),
        }


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _write_run_summary(summaries: List[dict], out_dir: Path) -> Path:
    """
    Writes a JSON summary file with per-segment rows and aggregate stats.
    This is the primary artefact for thesis plots and cross-dataset comparison.

    DATA NOTE: all values are real pipeline outputs — nothing is mocked.
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
        "dataset":             "Comma2k19",
        "reference":           "Schafer et al. 2018 — arXiv:1812.05752",
        "official_repo":       "https://github.com/commaai/comma2k19",
        "run_timestamp":       ts,
        "n_segments_total":    len(summaries),
        "n_complete":          n_ok,
        "n_failed":            n_fail,
        "n_skipped":           n_skip,
        "n_with_gps":          n_with_gps,
        "total_frames":        total_frames,
        "total_detections":    total_dets,
        "total_inserted":      total_ins,
        "total_updated":       total_upd,
        "total_elapsed_s":     round(elapsed_all, 2),
        "detection_rate_pct":  round(det_rate * 100, 2),
        "interpretation_note": (
            "Comma2k19 is an unlabelled dashcam dataset (California highway I-280). "
            "No road-damage ground truth exists. Detection rate reflects raw RIDS "
            "inference on unannotated footage — comparable to Cluj Run 1 (~4%) and "
            "Tokyo validation runs. Low rates are expected due to domain gap "
            "(N-RDD2024 trained on Japan/India/China/Norway/Czech/USA roads)."
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

    logger.info("Run summary → %s", out_path)
    logger.info(
        "AGGREGATE | segs=%d | complete=%d | failed=%d | skipped=%d | "
        "gps=%d | frames=%d | dets=%d | det_rate=%.1f%%",
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
            "RIDS — Comma2k19 selective downloader + pipeline runner.\n"
            "Downloads only video.hevc + global_pos/ up to a size limit,\n"
            "then runs the full RIDS inference pipeline on each segment.\n\n"
            "Reference: Schafer et al. 2018, arXiv:1812.05752"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default=(
            r"C:\Facultate\pothole-detection\Pothole-Detection"
            r"\data\datasets\Comma2k19"
        ),
        help="Root directory for downloaded Comma2k19 data.",
    )
    parser.add_argument(
        "--work_dir",
        default=r"data\processed\comma2k19",
        help="Working directory for session outputs and temporary GPX files.",
    )
    parser.add_argument(
        "--size_limit_gb",
        type=float,
        default=30.0,
        help=(
            "Maximum total download size in GB (video.hevc only — "
            "global_pos/ is negligible). Default: 30.0"
        ),
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help=(
            "Skip the torrent download step. Discovers already-downloaded "
            "segments in --dest up to --size_limit_gb."
        ),
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Download only — do not run the RIDS pipeline. Useful for staging.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto | cuda | cpu (default: auto).",
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
        help="Do not write to PostgreSQL (passes --dry_run_db to orchestrator).",
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
    args = _parse_args()
    _setup_logging(args.verbose)

    dest     = Path(args.dest)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RIDS — Comma2k19 selective download + pipeline")
    logger.info("Dataset destination : %s", dest)
    logger.info("Work directory      : %s", work_dir)
    logger.info("Size limit          : %.1f GB (video.hevc only)", args.size_limit_gb)
    logger.info("Skip download       : %s", args.skip_download)
    logger.info("Skip pipeline       : %s", args.skip_pipeline)
    logger.info("Device              : %s", args.device)
    logger.info("FPS                 : %.1f", args.fps)
    logger.info("Dry-run DB          : %s", args.dry_run_db)
    logger.info("Reference           : Schafer et al. 2018 — arXiv:1812.05752")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 0 — Dependency check
    # ------------------------------------------------------------------
    logger.info("[Step 0] Checking dependencies ...")
    try:
        import numpy  # noqa: F401
    except ImportError:
        logger.error("numpy not installed. Run: pip install numpy")
        sys.exit(1)

    if not args.skip_download:
        try:
            import libtorrent  # noqa: F401
        except ImportError:
            logger.error(
                "python-libtorrent not installed.\n"
                "  pip install python-libtorrent\n"
                "Or use --skip_download if data is already on disk."
            )
            sys.exit(1)

    logger.info("[Step 0] Dependencies OK.")

    # ------------------------------------------------------------------
    # Step 1 — Download or discover
    # ------------------------------------------------------------------
    if args.skip_download:
        logger.info("[Step 1] Skipping download — discovering existing segments ...")
        segments = _discover_segments(dest, args.size_limit_gb)
    else:
        logger.info("[Step 1] Starting selective torrent download ...")
        segments = _selective_download(dest, args.size_limit_gb, args.verbose)

    if not segments:
        logger.error("No segments found. Exiting.")
        sys.exit(1)

    logger.info("[Step 1] %d segments ready.", len(segments))

    if args.skip_pipeline:
        logger.info("[Step 2] --skip_pipeline set — download only mode. Done.")
        return

    # ------------------------------------------------------------------
    # Step 2 — Run RIDS pipeline segment by segment
    # ------------------------------------------------------------------
    logger.info("[Step 2] Running RIDS pipeline on %d segments ...", len(segments))
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
        logger.debug("Partial summary updated → %s", partial)

    # ------------------------------------------------------------------
    # Step 3 — Final summary
    # ------------------------------------------------------------------
    logger.info("[Step 3] Writing final run summary ...")
    summary_path = _write_run_summary(summaries, work_dir / "reports")

    logger.info("=" * 70)
    logger.info("Done. Summary: %s", summary_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()