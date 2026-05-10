"""
scripts/run_kitti_pipeline.py
------------------------------
End-to-end pipeline test on the KITTI raw dataset.

Purpose
-------
Validates the full RIDS pipeline (Stages 2–9) on KITTI imagery, which
provides real GPS coordinates and precise UTC timestamps — making it the
first dataset in this project where spatial deduplication (Stage 7) and
database enrichment (Stage 6) can be fully exercised.

Why Preprocessor.run() is NOT used
------------------------------------
Preprocessor.run() extracts frames from a .mp4 video. KITTI provides frames
as individual .png files on disk — there is no video to decode. Instead, this
script reuses the three internal functions from preprocessor.py directly:
  - classify_lighting()    — HSV-based lighting label + shadow score
  - compute_sun_elevation() — pysolar sun angle from GPS + UTC timestamp
  - FrameResult             — the exact same dataclass Stage 2 expects
This produces FrameResult objects that are 100% identical to what
Preprocessor.run() would produce, except frame_path points to the existing
.png file rather than a freshly written .jpg.

KITTI metadata parsing
----------------------
For each frame N (0-indexed), the following files are read:
  image_03/data/{N:010d}.png          — RGB frame (right colour camera)
  image_03/timestamps.txt             — line N gives the UTC wall-clock time
      Format: "YYYY-MM-DD HH:MM:SS.nanoseconds"
  oxts/data/{N:010d}.txt              — 30 space-separated floats
      Field 0 = latitude  (degrees, WGS84)
      Field 1 = longitude (degrees, WGS84)
      (see dataformat.txt for full field list)

Camera: image_03 is KITTI's right colour camera.
Focal length: 721 px (from KITTI calib_cam_to_cam.txt, fx of P_rect_03).

Drives processed
----------------
  2011_09_26_drive_0001_sync  (108 frames)
  2011_09_26_drive_0002_sync
  2011_09_26_drive_0018_sync
  2011_09_26_drive_0057_sync

All four drives are processed in sequence. Each drive becomes its own
session in data/processed/sessions/kitti_<drive>/. The survey_log table
receives one row per drive.

Usage
-----
    python scripts/run_kitti_pipeline.py
    python scripts/run_kitti_pipeline.py --drive 0001        # single drive
    python scripts/run_kitti_pipeline.py --limit 20          # first 20 frames per drive
    python scripts/run_kitti_pipeline.py --dry_run_db        # skip real DB writes
    python scripts/run_kitti_pipeline.py --skip_enrichment   # skip Nominatim/OSM/weather
    python scripts/run_kitti_pipeline.py --resume            # skip already-done stages
    python scripts/run_kitti_pipeline.py --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env first
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kitti_pipeline")

# ---------------------------------------------------------------------------
# Project root → sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import exactly the preprocessor functions we reuse — no Preprocessor.run()
# ---------------------------------------------------------------------------
from pipeline.preprocessor import (  # noqa: E402
    FrameResult,
    PreprocessorConfig,
    Preprocessor,
    classify_lighting,
    compute_sun_elevation,
)

# ---------------------------------------------------------------------------
# KITTI dataset root
# ---------------------------------------------------------------------------
KITTI_ROOT = (
    PROJECT_ROOT
    / "data" / "datasets" / "kitti" / "2011_09_26"
)

# KITTI image_03 right-colour-camera focal length (from calib_cam_to_cam.txt)
KITTI_FOCAL_LENGTH_PX = 721.0

# Four drives to process — (drive_id, has image_03 right cam)
KITTI_DRIVES: List[str] = ["0001", "0002", "0018", "0057"]

# Sessions output root
SESSIONS_DIR = PROJECT_ROOT / "data" / "processed" / "sessions"


# ---------------------------------------------------------------------------
# KITTI metadata parsing
# ---------------------------------------------------------------------------

def _parse_timestamp_line(line: str) -> datetime:
    """
    Parse one line from KITTI timestamps.txt.
    Format: "YYYY-MM-DD HH:MM:SS.nanoseconds"
    Python's datetime only handles microseconds (6 digits), so we truncate
    the nanosecond field to 6 digits before parsing.

    Example: "2011-09-26 13:02:25.961178112"
             truncated to "2011-09-26 13:02:25.961178"
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty timestamp line")

    # Split at the decimal point
    if "." in line:
        base, frac = line.rsplit(".", 1)
        # Truncate fractional seconds to 6 digits (microseconds)
        frac = frac[:6].ljust(6, "0")
        line = f"{base}.{frac}"

    dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc)


def _parse_oxts_file(oxts_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse one KITTI oxts .txt file and return (latitude, longitude).
    The file contains 30 space-separated floats on a single line.
    Field 0 = lat (degrees WGS84), Field 1 = lon (degrees WGS84).
    Returns (None, None) if the file is missing or malformed.
    """
    if not oxts_path.exists():
        logger.warning("oxts file missing: %s", oxts_path)
        return None, None

    try:
        with oxts_path.open("r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        lat = float(parts[0])
        lon = float(parts[1])
        return lat, lon
    except (IndexError, ValueError) as exc:
        logger.warning("oxts parse error for %s: %s", oxts_path, exc)
        return None, None


def build_kitti_frame_results(
    drive_id:   str,
    cfg:        PreprocessorConfig,
    limit:      Optional[int] = None,
) -> List[FrameResult]:
    """
    Build a list of FrameResult objects from KITTI files for one drive,
    reusing the exact same preprocessing functions (classify_lighting,
    compute_sun_elevation) that Preprocessor.run() uses internally.

    No video is opened. No frames are written to disk — the existing .png
    files are referenced directly.

    Parameters
    ----------
    drive_id : str
        Four-digit drive ID, e.g. "0001".
    cfg : PreprocessorConfig
        Used for lighting_thresholds, shadow_gradient_threshold, and
        focal_length_px (overridden to KITTI_FOCAL_LENGTH_PX).
    limit : int or None
        If given, process only the first N frames (useful for quick tests).

    Returns
    -------
    List[FrameResult]
        One per frame, in chronological order.
    """
    drive_dir  = KITTI_ROOT / f"2011_09_26_drive_{drive_id}_sync"
    images_dir = drive_dir / "image_03" / "data"
    ts_file    = drive_dir / "image_03" / "timestamps.txt"
    oxts_dir   = drive_dir / "oxts" / "data"

    # Validate paths
    if not images_dir.exists():
        raise FileNotFoundError(f"KITTI images dir not found: {images_dir}")
    if not ts_file.exists():
        raise FileNotFoundError(f"KITTI timestamps.txt not found: {ts_file}")
    if not oxts_dir.exists():
        raise FileNotFoundError(f"KITTI oxts data dir not found: {oxts_dir}")

    # Read timestamps.txt — one line per frame
    with ts_file.open("r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    # Collect sorted .png files
    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {images_dir}")

    n_frames = min(len(raw_lines), len(png_files))
    if limit is not None:
        n_frames = min(n_frames, limit)

    logger.info(
        "Drive %s | %d frames found | timestamps=%d | limit=%s",
        drive_id, len(png_files), len(raw_lines), limit or "none",
    )

    # Wall-clock time of the first frame — used for timestamp_s offset
    t0 = _parse_timestamp_line(raw_lines[0])

    results: List[FrameResult] = []
    n_dropped = 0

    for idx in range(n_frames):
        png_path = png_files[idx]
        frame_stem = png_path.stem  # e.g. "0000000000"

        # --- Parse timestamp ---
        try:
            wall_time = _parse_timestamp_line(raw_lines[idx])
        except (ValueError, IndexError) as exc:
            logger.warning("Drive %s frame %d: timestamp parse error: %s", drive_id, idx, exc)
            continue

        # timestamp_s = seconds since the first frame of this drive
        timestamp_s = (wall_time - t0).total_seconds()

        # --- Parse GPS from oxts ---
        oxts_path = oxts_dir / f"{frame_stem}.txt"
        lat, lon = _parse_oxts_file(oxts_path)

        # --- Load image with OpenCV ---
        frame_bgr = cv2.imread(str(png_path))
        if frame_bgr is None:
            logger.warning("Drive %s frame %d: cv2.imread failed for %s", drive_id, idx, png_path)
            n_dropped += 1
            continue

        # --- Brightness filter (same logic as Preprocessor.run step 5a) ---
        if cfg.min_mean_brightness > 0:
            mean_brightness = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))
            if mean_brightness < cfg.min_mean_brightness:
                logger.debug(
                    "Drive %s frame %d: dropped — brightness %.1f < %.1f",
                    drive_id, idx, mean_brightness, cfg.min_mean_brightness,
                )
                n_dropped += 1
                continue

        # --- Lighting classification — EXACT same function as Preprocessor ---
        lighting, shadow_score = classify_lighting(frame_bgr, cfg.lighting_thresholds)

        # --- Sun elevation — EXACT same function as Preprocessor ---
        sun_elev: Optional[float] = None
        if lat is not None and lon is not None:
            sun_elev = compute_sun_elevation(lat, lon, wall_time)

        result = FrameResult(
            frame_path            = str(png_path),   # existing .png, not copied
            frame_index           = idx,
            timestamp_s           = timestamp_s,
            wall_time             = wall_time,
            latitude              = lat,
            longitude             = lon,
            sun_elevation         = sun_elev,
            lighting              = lighting,
            shadow_geometry_score = shadow_score,
            focal_length_px       = KITTI_FOCAL_LENGTH_PX,
            gps_interpolated      = False,  # KITTI: each frame has its own oxts file
        )
        results.append(result)

        if idx % 20 == 0:
            logger.info(
                "Drive %s | frame %d/%d | t=%.2fs | lighting=%-10s | "
                "sun=%.1f° | lat=%.5f | lon=%.5f",
                drive_id, idx, n_frames - 1, timestamp_s,
                lighting,
                sun_elev if sun_elev is not None else float("nan"),
                lat if lat is not None else float("nan"),
                lon if lon is not None else float("nan"),
            )

    # Summary
    n_gps_ok   = sum(1 for r in results if r.latitude is not None)
    n_daylight = sum(1 for r in results if r.lighting == "daylight")
    n_overcast = sum(1 for r in results if r.lighting == "overcast")
    n_low      = sum(1 for r in results if r.lighting == "low_light")

    logger.info("=== Drive %s frame build complete ===", drive_id)
    logger.info("  Frames built       : %d", len(results))
    logger.info("  Frames dropped     : %d", n_dropped)
    logger.info("  Frames with GPS    : %d / %d", n_gps_ok, len(results))
    logger.info("  Lighting           : daylight=%d overcast=%d low_light=%d",
                n_daylight, n_overcast, n_low)

    return results


# ---------------------------------------------------------------------------
# Single-drive pipeline runner
# ---------------------------------------------------------------------------

def run_drive(
    drive_id:        str,
    device:          str,
    limit:           Optional[int],
    skip_enrichment: bool,
    skip_weather:    bool,
    skip_overpass:   bool,
    dry_run_db:      bool,
    resume:          bool,
) -> dict:
    """
    Run the full pipeline (Stages 1*–9) on one KITTI drive.

    Stage 1* is replaced by build_kitti_frame_results(), which produces
    identical FrameResult objects without extracting from a video.

    Returns a summary dict.
    """
    from pipeline.detector           import Detector, DetectorConfig
    from pipeline.segmentor          import Segmentor, SegmentorConfig
    from pipeline.depth_estimator    import DepthEstimator, DepthEstimatorConfig
    from pipeline.severity_classifier import SeverityClassifier, SeverityConfig
    from pipeline.enricher           import Enricher, EnricherConfig
    from pipeline.deduplicator       import Deduplicator, DeduplicatorConfig
    from pipeline.db_writer          import DbWriter, DbWriterConfig
    import os

    session_id  = f"kitti_{drive_id}"
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    weights_dir    = str(PROJECT_ROOT / "ml" / "weights")
    rtdetr_weights = os.environ.get("RTDETR_WEIGHTS", "rtdetr_l_nrdd2024.pt")
    weights_path   = str(Path(weights_dir) / rtdetr_weights)

    # survey_log
    started_at = datetime.now(tz=timezone.utc)
    log_id     = _survey_log_start(session_id, drive_id, started_at, dry_run_db)

    summary = {
        "drive_id":    drive_id,
        "session_id":  session_id,
        "started_at":  started_at.isoformat(),
        "status":      "running",
        "stages":      [],
        "n_frames":    0,
        "n_detections": 0,
        "n_inserted":  0,
        "n_updated":   0,
        "error":       None,
    }

    try:
        # ── Stage 1* — KITTI frame builder ────────────────────────────────
        manifest_path = session_dir / "01_manifest" / "manifest.json"
        frames = _stage_kitti_frames(
            summary, drive_id, manifest_path, limit, resume
        )
        summary["n_frames"] = len(frames)

        # ── Stage 2 — Detector ────────────────────────────────────────────
        det_path   = session_dir / "02_detections" / "detections.json"
        det_cfg    = DetectorConfig(weights=weights_path, device=device)
        det_results = _run_stage(
            summary, "detector",
            resume_path   = det_path,
            load_fn       = lambda: Detector.load_detections(str(det_path)),
            run_fn        = lambda: Detector(det_cfg).run(
                frames, output_dir=str(det_path.parent)
            ),
        )
        summary["n_detections"] = sum(r.n_detections for r in det_results)

        # ── Stage 3 — Segmentor ───────────────────────────────────────────
        seg_path  = session_dir / "03_segmentations" / "segmentations.json"
        sam_weights = str(Path(weights_dir) / "sam2.1_hiera_tiny.pt")
        seg_cfg   = SegmentorConfig(weights=sam_weights, device=device)
        seg_results = _run_stage(
            summary, "segmentor",
            resume_path = seg_path,
            load_fn     = lambda: Segmentor.load_segmentations(str(seg_path)),
            run_fn      = lambda: Segmentor(seg_cfg).run(
                det_results, output_dir=str(seg_path.parent)
            ),
        )

        # ── Stage 4 — DepthEstimator ──────────────────────────────────────
        depth_path = session_dir / "04_depth" / "depth_estimates.json"
        dep_cfg    = DepthEstimatorConfig(device=device)
        dep_results = _run_stage(
            summary, "depth_estimator",
            resume_path = depth_path,
            load_fn     = lambda: DepthEstimator.load_depth_estimates(str(depth_path)),
            run_fn      = lambda: DepthEstimator(dep_cfg).run(
                seg_results, output_dir=str(depth_path.parent)
            ),
        )

        # ── Stage 5 — SeverityClassifier ─────────────────────────────────
        sev_path  = session_dir / "05_severity" / "severity_estimates.json"
        sev_cfg   = SeverityConfig()
        sev_results = _run_stage(
            summary, "severity_classifier",
            resume_path = sev_path,
            load_fn     = lambda: SeverityClassifier.load_severity(str(sev_path)),
            run_fn      = lambda: SeverityClassifier(sev_cfg).run(
                dep_results, output_dir=str(sev_path.parent)
            ),
        )

        # ── Stage 6 — Enricher ────────────────────────────────────────────
        enr_path   = session_dir / "06_enriched" / "enriched.json"

        if skip_enrichment:
            logger.info("Drive %s Stage 6 — SKIPPED (--skip_enrichment)", drive_id)
            enr_frames = [r.to_dict() for r in sev_results]
            _write_json({"frames": enr_frames}, enr_path)
            summary["stages"].append(
                {"name": "enricher", "skipped": True, "elapsed_s": 0.0,
                 "output": str(enr_path), "error": None}
            )
        else:
            enr_cfg = EnricherConfig(
                skip_weather  = skip_weather,
                skip_overpass = skip_overpass,
            )
            sev_dicts = [r.to_dict() for r in sev_results]
            enr_results_obj = _run_stage(
                summary, "enricher",
                resume_path = enr_path,
                load_fn     = lambda: Enricher.load_enriched(str(enr_path)),
                run_fn      = lambda: Enricher(enr_cfg).run(
                    sev_dicts, output_dir=str(enr_path.parent)
                ),
            )
            # Enricher.run() returns List[EnrichmentResult]; convert to dicts
            # for Stage 7. If resume loaded from JSON it's already dicts.
            if enr_results_obj and hasattr(enr_results_obj[0], "to_dict"):
                enr_frames = [r.to_dict() for r in enr_results_obj]
            else:
                enr_frames = enr_results_obj

        # ── Stage 7 — Deduplicator ────────────────────────────────────────
        dedup_path = session_dir / "07_deduplicated" / "deduplicated.json"
        dedup_cfg  = DeduplicatorConfig()
        dedup_results_obj = _run_stage(
            summary, "deduplicator",
            resume_path = dedup_path,
            load_fn     = lambda: Deduplicator.load_deduplicated(str(dedup_path)),
            run_fn      = lambda: Deduplicator(dedup_cfg).run(
                enr_frames, output_dir=str(dedup_path.parent)
            ),
        )
        # Deduplicator.run() returns List[DeduplicationResult]; convert to dicts
        if dedup_results_obj and hasattr(dedup_results_obj[0], "to_dict"):
            dedup_frames = [r.to_dict() for r in dedup_results_obj]
        else:
            dedup_frames = dedup_results_obj

        # ── Stage 8 — DbWriter ────────────────────────────────────────────
        db_path  = session_dir / "08_db_write" / "db_write_summary.json"
        db_cfg   = DbWriterConfig(dry_run=dry_run_db)
        t0_db    = time.perf_counter()
        db_result = DbWriter(db_cfg).run(
            dedup_frames,
            output_dir=str(db_path.parent),
        )
        elapsed_db = time.perf_counter() - t0_db

        summary["n_inserted"] = db_result.n_inserted
        summary["n_updated"]  = db_result.n_updated
        summary["stages"].append({
            "name":      "db_writer",
            "skipped":   False,
            "elapsed_s": round(elapsed_db, 2),
            "output":    str(db_path),
            "error":     None,
        })

        summary["status"] = "complete"

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"]  = traceback.format_exc()
        logger.error("Drive %s pipeline failed: %s", drive_id, exc)

    finally:
        summary["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        elapsed_total = (
            datetime.fromisoformat(summary["finished_at"])
            - datetime.fromisoformat(summary["started_at"])
        ).total_seconds()
        logger.info(
            "=== Drive %s %s | %.1f s | frames=%d | dets=%d | "
            "inserted=%d | updated=%d ===",
            drive_id, summary["status"].upper(),
            elapsed_total,
            summary["n_frames"],
            summary["n_detections"],
            summary["n_inserted"],
            summary["n_updated"],
        )

        # Save session.json
        session_json = session_dir / "session.json"
        with session_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Session summary: %s", session_json)

        _survey_log_finish(log_id, summary, dry_run_db)

    return summary


# ---------------------------------------------------------------------------
# Stage helper: resume or run, log timing
# ---------------------------------------------------------------------------

def _stage_kitti_frames(
    summary:       dict,
    drive_id:      str,
    manifest_path: Path,
    limit:         Optional[int],
    resume:        bool,
) -> List[FrameResult]:
    """Build FrameResult list from KITTI files, or load from manifest on resume."""
    if resume and manifest_path.exists():
        logger.info("Drive %s Stage 1* — RESUME: manifest exists", drive_id)
        frames = Preprocessor.load_manifest(str(manifest_path))
        summary["stages"].append({
            "name": "kitti_frame_builder", "skipped": True,
            "elapsed_s": 0.0, "output": str(manifest_path), "error": None,
        })
        return frames

    t0  = time.perf_counter()
    cfg = PreprocessorConfig(focal_length_px=KITTI_FOCAL_LENGTH_PX)
    try:
        frames = build_kitti_frame_results(drive_id, cfg, limit=limit)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        Preprocessor.save_manifest(frames, str(manifest_path))
        elapsed = time.perf_counter() - t0
        summary["stages"].append({
            "name": "kitti_frame_builder", "skipped": False,
            "elapsed_s": round(elapsed, 2), "output": str(manifest_path), "error": None,
        })
        logger.info(
            "Drive %s Stage 1* complete: %d frames in %.1f s",
            drive_id, len(frames), elapsed,
        )
        return frames
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        summary["stages"].append({
            "name": "kitti_frame_builder", "skipped": False,
            "elapsed_s": round(elapsed, 2), "output": None, "error": str(exc),
        })
        raise


def _run_stage(
    summary:     dict,
    name:        str,
    resume_path: Path,
    load_fn,
    run_fn,
    resume:      bool = True,   # always honour resume in KITTI script
):
    """
    Generic stage runner: if resume_path exists and resume=True, load from disk.
    Otherwise execute run_fn, log timing, and append to summary["stages"].
    """
    if resume_path.exists():
        logger.info("Stage %-22s — RESUME: output exists, loading from disk", name)
        result = load_fn()
        summary["stages"].append({
            "name": name, "skipped": True,
            "elapsed_s": 0.0, "output": str(resume_path), "error": None,
        })
        return result

    t0 = time.perf_counter()
    try:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        result  = run_fn()
        elapsed = time.perf_counter() - t0
        summary["stages"].append({
            "name": name, "skipped": False,
            "elapsed_s": round(elapsed, 2), "output": str(resume_path), "error": None,
        })
        logger.info("Stage %-22s complete in %.1f s", name, elapsed)
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        summary["stages"].append({
            "name": name, "skipped": False,
            "elapsed_s": round(elapsed, 2), "output": None, "error": str(exc),
        })
        raise


# ---------------------------------------------------------------------------
# survey_log helpers
# ---------------------------------------------------------------------------

def _survey_log_start(
    session_id: str,
    drive_id:   str,
    started_at: datetime,
    dry_run:    bool,
) -> Optional[int]:
    if dry_run:
        logger.info("survey_log: DRY RUN — no row inserted")
        return None
    try:
        import psycopg2, os
        conn = psycopg2.connect(
            host     = os.environ.get("POSTGRES_HOST",     "localhost"),
            port     = int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname   = os.environ.get("POSTGRES_DB",       "cluj_monitor"),
            user     = os.environ.get("POSTGRES_USER",     "postgres"),
            password = os.environ.get("POSTGRES_PASSWORD", ""),
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO survey_log
                (survey_date, started_at, status, video_files)
            VALUES (%s, %s, 'running', %s)
            RETURNING id;
            """,
            (
                started_at.date(),
                started_at,
                json.dumps([f"kitti_drive_{drive_id}"]),
            ),
        )
        row_id = cur.fetchone()[0]
        cur.close()
        conn.close()
        logger.info("survey_log: started row id=%d (drive=%s)", row_id, drive_id)
        return row_id
    except Exception as exc:
        logger.warning("survey_log start failed: %s", exc)
        return None


def _survey_log_finish(
    log_id:  Optional[int],
    summary: dict,
    dry_run: bool,
) -> None:
    if log_id is None or dry_run:
        return
    try:
        import psycopg2, os
        conn = psycopg2.connect(
            host     = os.environ.get("POSTGRES_HOST",     "localhost"),
            port     = int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname   = os.environ.get("POSTGRES_DB",       "cluj_monitor"),
            user     = os.environ.get("POSTGRES_USER",     "postgres"),
            password = os.environ.get("POSTGRES_PASSWORD", ""),
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE survey_log
            SET    finished_at        = %s,
                   status             = %s,
                   frames_processed   = %s,
                   detections_found   = %s,
                   new_detections     = %s,
                   updated_detections = %s,
                   error_message      = %s
            WHERE  id = %s;
            """,
            (
                datetime.fromisoformat(summary.get("finished_at", datetime.now(tz=timezone.utc).isoformat())),
                summary["status"],
                summary["n_frames"],
                summary["n_detections"],
                summary["n_inserted"],
                summary["n_updated"],
                summary.get("error"),
                log_id,
            ),
        )
        cur.close()
        conn.close()
        logger.info("survey_log: row %d updated → %s", log_id, summary["status"])
    except Exception as exc:
        logger.warning("survey_log finish failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RIDS full pipeline test on KITTI dataset. "
            "Processes Stages 2–9 on pre-existing KITTI .png frames. "
            "All DB credentials are read from .env."
        )
    )
    parser.add_argument(
        "--drive", default=None,
        choices=KITTI_DRIVES,
        help="Process only a specific drive (default: all four drives)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N frames per drive (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device: cuda | cpu | cuda:0  (default: cuda)",
    )
    parser.add_argument(
        "--skip_enrichment", action="store_true",
        help="Skip Stage 6 — Nominatim / Overpass / Open-Meteo",
    )
    parser.add_argument(
        "--skip_weather", action="store_true",
        help="Skip Open-Meteo weather calls in Stage 6",
    )
    parser.add_argument(
        "--skip_overpass", action="store_true",
        help="Skip OSM Overpass calls in Stage 6",
    )
    parser.add_argument(
        "--dry_run_db", action="store_true",
        help="Stage 8: log what would be written but do not commit",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip stages whose output JSON already exists on disk",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    drives = [args.drive] if args.drive else KITTI_DRIVES

    logger.info("=== KITTI pipeline test ===")
    logger.info("  Drives    : %s", drives)
    logger.info("  Device    : %s", args.device)
    logger.info("  Limit     : %s", args.limit or "all frames")
    logger.info("  DB mode   : %s", "DRY RUN" if args.dry_run_db else "LIVE WRITES")
    logger.info("  Enrichment: %s", "SKIP" if args.skip_enrichment else "enabled")

    all_summaries = []
    any_failed    = False

    for drive_id in drives:
        logger.info("─" * 60)
        logger.info("Starting drive %s", drive_id)
        summary = run_drive(
            drive_id        = drive_id,
            device          = args.device,
            limit           = args.limit,
            skip_enrichment = args.skip_enrichment,
            skip_weather    = args.skip_weather,
            skip_overpass   = args.skip_overpass,
            dry_run_db      = args.dry_run_db,
            resume          = args.resume,
        )
        all_summaries.append(summary)
        if summary["status"] == "failed":
            any_failed = True

    # Overall summary across all drives
    logger.info("═" * 60)
    logger.info("=== KITTI pipeline test complete ===")
    total_frames = sum(s["n_frames"]     for s in all_summaries)
    total_dets   = sum(s["n_detections"] for s in all_summaries)
    total_ins    = sum(s["n_inserted"]   for s in all_summaries)
    total_upd    = sum(s["n_updated"]    for s in all_summaries)

    for s in all_summaries:
        status_icon = "✓" if s["status"] == "complete" else "✗"
        logger.info(
            "  %s Drive %s | frames=%d | dets=%d | inserted=%d | updated=%d",
            status_icon, s["drive_id"],
            s["n_frames"], s["n_detections"], s["n_inserted"], s["n_updated"],
        )

    logger.info("  ─────────────────────────────────────────")
    logger.info(
        "  TOTAL             | frames=%d | dets=%d | inserted=%d | updated=%d",
        total_frames, total_dets, total_ins, total_upd,
    )

    # Save combined summary
    combined_path = SESSIONS_DIR / "kitti_run_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Combined summary: %s", combined_path)

    if any_failed:
        logger.error("One or more drives failed. Check session.json files.")
        sys.exit(1)


if __name__ == "__main__":
    main()