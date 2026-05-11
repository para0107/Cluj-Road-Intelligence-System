"""
scripts/run_kitti_pipeline.py
------------------------------
KITTI pipeline test — enrichment stage permanently removed.

Stage sequence
--------------
  1* KITTI frame builder   — FrameResult from .png + oxts + timestamps
  2  Detector              — RT-DETR-L
  3  Segmentor             — SAM 2.1 Tiny
  4  DepthEstimator        — Monodepth2
  5  SeverityClassifier    — S1-S5
  6  Deduplicator          — DBSCAN  (renumbered from 7)
  7  DbWriter              — PostGIS upsert  (renumbered from 8)
  8  survey_log write

Output layout per drive
-----------------------
  data/processed/sessions/kitti_<drive>/
      01_manifest/         manifest.json
      02_detections/       detections.json
      03_segmentations/    segmentations.json
      04_depth/            depth_estimates.json
      05_severity/         severity_estimates.json
      06_deduplicated/     deduplicated.json + dedup_report.html
      07_db_write/         db_write_summary.json
      session.json

Usage
-----
    python scripts/run_kitti_pipeline.py
    python scripts/run_kitti_pipeline.py --drive 0001
    python scripts/run_kitti_pipeline.py --limit 20
    python scripts/run_kitti_pipeline.py --dry_run_db
    python scripts/run_kitti_pipeline.py --skip_dedup
    python scripts/run_kitti_pipeline.py --resume
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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kitti_pipeline")

PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessor import (
    FrameResult, PreprocessorConfig, Preprocessor,
    classify_lighting, compute_sun_elevation,
)

KITTI_ROOT            = PROJECT_ROOT / "data" / "datasets" / "kitti" / "2011_09_26"
SESSIONS_DIR          = PROJECT_ROOT / "data" / "processed" / "sessions"
KITTI_FOCAL_LENGTH_PX = 721.0
KITTI_DRIVES: List[str] = ["0001", "0002", "0018", "0057"]


def _parse_timestamp_line(line: str) -> datetime:
    line = line.strip()
    if "." in line:
        base, frac = line.rsplit(".", 1)
        frac = frac[:6].ljust(6, "0")
        line = f"{base}.{frac}"
    return datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def _parse_oxts_file(path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not path.exists():
        return None, None
    try:
        parts = path.read_text(encoding="utf-8").strip().split()
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None, None


def build_kitti_frame_results(
    drive_id: str,
    cfg:      PreprocessorConfig,
    limit:    Optional[int] = None,
) -> List[FrameResult]:
    drive_dir  = KITTI_ROOT / f"2011_09_26_drive_{drive_id}_sync"
    images_dir = drive_dir / "image_03" / "data"
    ts_file    = drive_dir / "image_03" / "timestamps.txt"
    oxts_dir   = drive_dir / "oxts" / "data"

    if not images_dir.exists():
        raise FileNotFoundError(f"KITTI images dir not found: {images_dir}")
    if not ts_file.exists():
        raise FileNotFoundError(f"KITTI timestamps.txt not found: {ts_file}")
    if not oxts_dir.exists():
        raise FileNotFoundError(f"KITTI oxts dir not found: {oxts_dir}")

    raw_lines = [l.strip() for l in ts_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files in {images_dir}")

    n_frames = min(len(raw_lines), len(png_files))
    if limit is not None:
        n_frames = min(n_frames, limit)

    logger.info(
        "Drive %s | %d frames found | timestamps=%d | limit=%s",
        drive_id, len(png_files), len(raw_lines), limit or "none",
    )

    t0 = _parse_timestamp_line(raw_lines[0])
    results: List[FrameResult] = []
    n_dropped = 0

    for idx in range(n_frames):
        png_path = png_files[idx]
        try:
            wall_time = _parse_timestamp_line(raw_lines[idx])
        except (ValueError, IndexError) as exc:
            logger.warning("Drive %s frame %d timestamp error: %s", drive_id, idx, exc)
            continue

        timestamp_s = (wall_time - t0).total_seconds()
        lat, lon    = _parse_oxts_file(oxts_dir / f"{png_path.stem}.txt")

        frame_bgr = cv2.imread(str(png_path))
        if frame_bgr is None:
            n_dropped += 1
            continue

        if cfg.min_mean_brightness > 0:
            mb = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))
            if mb < cfg.min_mean_brightness:
                n_dropped += 1
                continue

        lighting, shadow_score = classify_lighting(frame_bgr, cfg.lighting_thresholds)
        sun_elev = compute_sun_elevation(lat, lon, wall_time) if lat is not None else None

        results.append(FrameResult(
            frame_path            = str(png_path),
            frame_index           = idx,
            timestamp_s           = timestamp_s,
            wall_time             = wall_time,
            latitude              = lat,
            longitude             = lon,
            sun_elevation         = sun_elev,
            lighting              = lighting,
            shadow_geometry_score = shadow_score,
            focal_length_px       = KITTI_FOCAL_LENGTH_PX,
            gps_interpolated      = False,
        ))

        if idx % 20 == 0:
            logger.info(
                "Drive %s | frame %d/%d | t=%.2fs | lighting=%-10s | "
                "sun=%.1f° | lat=%.5f | lon=%.5f",
                drive_id, idx, n_frames - 1, timestamp_s, lighting,
                sun_elev if sun_elev is not None else float("nan"),
                lat if lat is not None else float("nan"),
                lon if lon is not None else float("nan"),
            )

    n_gps = sum(1 for r in results if r.latitude is not None)
    logger.info("=== Drive %s frame build complete ===", drive_id)
    logger.info("  Frames built   : %d", len(results))
    logger.info("  Frames dropped : %d", n_dropped)
    logger.info("  Frames with GPS: %d / %d", n_gps, len(results))
    logger.info(
        "  Lighting       : daylight=%d overcast=%d low_light=%d",
        sum(1 for r in results if r.lighting == "daylight"),
        sum(1 for r in results if r.lighting == "overcast"),
        sum(1 for r in results if r.lighting == "low_light"),
    )
    return results


def run_drive(
    drive_id:   str,
    device:     str,
    limit:      Optional[int],
    dry_run_db: bool,
    resume:     bool,
    skip_dedup: bool = False,
) -> dict:
    from pipeline.detector            import Detector, DetectorConfig
    from pipeline.segmentor           import Segmentor, SegmentorConfig
    from pipeline.depth_estimator     import DepthEstimator, DepthEstimatorConfig
    from pipeline.severity_classifier import SeverityClassifier, SeverityConfig
    from pipeline.deduplicator        import Deduplicator, DeduplicatorConfig
    from pipeline.db_writer           import DbWriter, DbWriterConfig
    import os

    session_id  = f"kitti_{drive_id}"
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    weights_dir  = str(PROJECT_ROOT / "ml" / "weights")
    weights_path = str(Path(weights_dir) / os.environ.get("RTDETR_WEIGHTS", "best.pt"))

    started_at = datetime.now(tz=timezone.utc)
    log_id     = _survey_log_start(session_id, drive_id, started_at, dry_run_db)

    summary = {
        "drive_id": drive_id, "session_id": session_id,
        "started_at": started_at.isoformat(), "status": "running",
        "stages": [], "n_frames": 0, "n_detections": 0,
        "n_inserted": 0, "n_updated": 0, "error": None,
    }

    try:
        # Stage 1*
        manifest_path = session_dir / "01_manifest" / "manifest.json"
        frames = _stage_kitti_frames(summary, drive_id, manifest_path, limit, resume)
        summary["n_frames"] = len(frames)

        # Stage 2
        det_path = session_dir / "02_detections" / "detections.json"
        det_cfg  = DetectorConfig(weights=weights_path, device=device)
        det_results = _run_stage(
            summary, "detector", det_path,
            lambda: Detector.load_detections(str(det_path)),
            lambda: Detector(det_cfg).run(frames, output_dir=str(det_path.parent)),
        )
        summary["n_detections"] = sum(r.n_detections for r in det_results)

        # Stage 3
        seg_path    = session_dir / "03_segmentations" / "segmentations.json"
        sam_weights = str(Path(weights_dir) / "sam2.1_hiera_tiny.pt")
        seg_cfg     = SegmentorConfig(weights=sam_weights, device=device)
        seg_results = _run_stage(
            summary, "segmentor", seg_path,
            lambda: Segmentor.load_segmentations(str(seg_path)),
            lambda: Segmentor(seg_cfg).run(det_results, output_dir=str(seg_path.parent)),
        )

        # Stage 4
        depth_path = session_dir / "04_depth" / "depth_estimates.json"
        dep_cfg    = DepthEstimatorConfig(
            monodepth_root = os.environ.get("MONODEPTH_ROOT",
                r"C:\Facultate\pothole-detection\Monodepth"),
            weights_dir    = os.environ.get("MONODEPTH_WEIGHTS_DIR",
                r"C:\Facultate\pothole-detection\Pothole-Detection\ml\weights\mono_640x192"),
            device         = device,
        )
        dep_results = _run_stage(
            summary, "depth_estimator", depth_path,
            lambda: DepthEstimator.load_depth_estimates(str(depth_path)),
            lambda: DepthEstimator(dep_cfg).run(seg_results, output_dir=str(depth_path.parent)),
        )

        # Stage 5
        sev_path    = session_dir / "05_severity" / "severity_estimates.json"
        sev_cfg     = SeverityConfig()
        sev_results = _run_stage(
            summary, "severity_classifier", sev_path,
            lambda: SeverityClassifier.load_severity(str(sev_path)),
            lambda: SeverityClassifier(sev_cfg).run(dep_results, output_dir=str(sev_path.parent)),
        )

        # Stage 6 — Deduplicator (renumbered from 7)
        dedup_path = session_dir / "06_deduplicated" / "deduplicated.json"
        sev_dicts  = (
            [r.to_dict() for r in sev_results]
            if sev_results and hasattr(sev_results[0], "to_dict")
            else sev_results
        )

        if skip_dedup:
            logger.info("Drive %s Stage 6 — Deduplication SKIPPED (--skip_dedup)", drive_id)
            dedup_frames = [
                dict(fr, boxes=[
                    dict(b, dedup={"cluster_id": -1, "is_duplicate": False, "cluster_size": 1})
                    for b in fr.get("boxes", [])
                ])
                for fr in sev_dicts
            ]
            summary["stages"].append({
                "name": "deduplicator", "skipped": True,
                "elapsed_s": 0.0, "output": None, "error": None,
            })
        else:
            dedup_cfg = DeduplicatorConfig()
            dedup_obj = _run_stage(
                summary, "deduplicator", dedup_path,
                lambda: Deduplicator.load_deduplicated(str(dedup_path)),
                lambda: Deduplicator(dedup_cfg).run(sev_dicts, output_dir=str(dedup_path.parent)),
            )
            dedup_frames = (
                [r.to_dict() for r in dedup_obj]
                if dedup_obj and hasattr(dedup_obj[0], "to_dict")
                else dedup_obj
            )

        # Stage 7 — DbWriter (renumbered from 8)
        db_path = session_dir / "07_db_write" / "db_write_summary.json"
        db_cfg  = DbWriterConfig(dry_run=dry_run_db)
        t0_db   = time.perf_counter()
        db_result = DbWriter(db_cfg).run(dedup_frames, output_dir=str(db_path.parent))
        elapsed_db = time.perf_counter() - t0_db

        summary["n_inserted"] = db_result.n_inserted
        summary["n_updated"]  = db_result.n_updated
        summary["stages"].append({
            "name": "db_writer", "skipped": False,
            "elapsed_s": round(elapsed_db, 2),
            "output": str(db_path), "error": None,
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
            "=== Drive %s %s | %.1f s | frames=%d | dets=%d | inserted=%d | updated=%d ===",
            drive_id, summary["status"].upper(), elapsed_total,
            summary["n_frames"], summary["n_detections"],
            summary["n_inserted"], summary["n_updated"],
        )
        session_json = session_dir / "session.json"
        with session_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Session summary: %s", session_json)
        _survey_log_finish(log_id, summary, dry_run_db)

    return summary


def _stage_kitti_frames(summary, drive_id, manifest_path, limit, resume):
    if resume and manifest_path.exists():
        logger.info("Drive %s Stage 1* — RESUME", drive_id)
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
        logger.info("Drive %s Stage 1* complete: %d frames in %.1f s",
                    drive_id, len(frames), elapsed)
        return frames
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        summary["stages"].append({
            "name": "kitti_frame_builder", "skipped": False,
            "elapsed_s": round(elapsed, 2), "output": None, "error": str(exc),
        })
        raise


def _run_stage(summary, name, resume_path, load_fn, run_fn):
    if resume_path.exists():
        logger.info("Stage %-22s — RESUME", name)
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


def _survey_log_start(session_id, drive_id, started_at, dry_run):
    if dry_run:
        return None
    try:
        import psycopg2, os
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST","localhost"),
            port=int(os.environ.get("POSTGRES_PORT","5432")),
            dbname=os.environ.get("POSTGRES_DB","cluj_monitor"),
            user=os.environ.get("POSTGRES_USER","postgres"),
            password=os.environ.get("POSTGRES_PASSWORD",""),
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO survey_log (survey_date, started_at, status, video_files)
            VALUES (%s, %s, 'running', %s)
            ON CONFLICT (survey_date) DO UPDATE
                SET started_at  = EXCLUDED.started_at,
                    status      = 'running',
                    video_files = survey_log.video_files || EXCLUDED.video_files
            RETURNING id;
            """,
            (started_at.date(), started_at, json.dumps([f"kitti_drive_{drive_id}"])),
        )
        row_id = cur.fetchone()[0]
        cur.close(); conn.close()
        logger.info("survey_log: started row id=%d (drive=%s)", row_id, drive_id)
        return row_id
    except Exception as exc:
        logger.warning("survey_log start failed: %s", exc)
        return None


def _survey_log_finish(log_id, summary, dry_run):
    if log_id is None or dry_run:
        return
    try:
        import psycopg2, os
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST","localhost"),
            port=int(os.environ.get("POSTGRES_PORT","5432")),
            dbname=os.environ.get("POSTGRES_DB","cluj_monitor"),
            user=os.environ.get("POSTGRES_USER","postgres"),
            password=os.environ.get("POSTGRES_PASSWORD",""),
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE survey_log
            SET finished_at=%s, status=%s, frames_processed=%s,
                detections_found=%s, new_detections=%s,
                updated_detections=%s, error_message=%s
            WHERE id=%s;
            """,
            (
                datetime.fromisoformat(
                    summary.get("finished_at", datetime.now(tz=timezone.utc).isoformat())
                ),
                summary["status"], summary["n_frames"], summary["n_detections"],
                summary["n_inserted"], summary["n_updated"],
                summary.get("error"), log_id,
            ),
        )
        cur.close(); conn.close()
        logger.info("survey_log: row %d → %s", log_id, summary["status"])
    except Exception as exc:
        logger.warning("survey_log finish failed: %s", exc)


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIDS KITTI pipeline test. Enrichment removed. DB creds from .env."
    )
    parser.add_argument("--drive",      default=None, choices=KITTI_DRIVES)
    parser.add_argument("--limit",      type=int,     default=None)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--dry_run_db", action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--skip_dedup", action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    drives = [args.drive] if args.drive else KITTI_DRIVES

    logger.info("=== KITTI pipeline test ===")
    logger.info("  Drives  : %s", drives)
    logger.info("  Device  : %s", args.device)
    logger.info("  Limit   : %s", args.limit or "all frames")
    logger.info("  DB mode : %s", "DRY RUN" if args.dry_run_db else "LIVE WRITES")

    all_summaries = []
    any_failed    = False

    for drive_id in drives:
        logger.info("─" * 60)
        logger.info("Starting drive %s", drive_id)
        s = run_drive(
            drive_id=drive_id, device=args.device, limit=args.limit,
            dry_run_db=args.dry_run_db, resume=args.resume, skip_dedup=args.skip_dedup,
        )
        all_summaries.append(s)
        if s["status"] == "failed":
            any_failed = True

    logger.info("═" * 60)
    logger.info("=== KITTI pipeline test complete ===")
    for s in all_summaries:
        icon = "✓" if s["status"] == "complete" else "✗"
        logger.info(
            "  %s Drive %s | frames=%d | dets=%d | inserted=%d | updated=%d",
            icon, s["drive_id"], s["n_frames"], s["n_detections"],
            s["n_inserted"], s["n_updated"],
        )
    logger.info("  ─────────────────────────────────────────")
    logger.info(
        "  TOTAL | frames=%d | dets=%d | inserted=%d | updated=%d",
        sum(s["n_frames"] for s in all_summaries),
        sum(s["n_detections"] for s in all_summaries),
        sum(s["n_inserted"] for s in all_summaries),
        sum(s["n_updated"] for s in all_summaries),
    )

    combined_path = SESSIONS_DIR / "kitti_run_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Combined summary: %s", combined_path)

    if any_failed:
        logger.error("One or more drives failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()