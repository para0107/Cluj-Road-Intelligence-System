"""
pipeline/orchestrator.py
------------------------
End-to-end coordinator for the Road Infrastructure Detection System (RIDS).

Stage sequence
--------------
  1  Preprocessor       — frame extraction from .mp4, GPS sync from .gpx,
                          sun angle (pysolar), lighting classification
  2  Detector           — RT-DETR-L inference, per-class confidence thresholds
  3  Segmentor          — SAM 2.1 Tiny, 4 geometry features per mask
  4  DepthEstimator     — Monodepth2 relative depth, proxy fallback
  5  SeverityClassifier — rule-based S1–S5 weighted multi-signal formula
  6  Deduplicator       — DBSCAN spatial clustering (eps from .env)
  7  DbWriter           — PostgreSQL/PostGIS upsert
  8  survey_log write   — record session outcome in survey_log table

Stage 6 (Enricher) has been permanently removed.
The pipeline no longer calls Nominatim, OSM Overpass, or Open-Meteo.
GPS coordinates (latitude, longitude) from the preprocessor/GPX are the
only location metadata stored. This is the designed and final behaviour.

Per-stage intermediate outputs:
    <work_dir>/<session_id>/
        01_manifest/         manifest.json
        02_detections/       detections.json
        03_segmentations/    segmentations.json
        04_depth/            depth_estimates.json
        05_severity/         severity_estimates.json
        06_deduplicated/     deduplicated.json + dedup_report.html
        07_db_write/         db_write_summary.json
        session.json         overall session summary

Environment variables (all from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    WEIGHTS_DIR              — base directory for model weights
    RTDETR_WEIGHTS           — RT-DETR checkpoint filename under WEIGHTS_DIR
    MONODEPTH_ROOT           — path to cloned Monodepth2 repo
    MONODEPTH_WEIGHTS_DIR    — path to mono_640x192/ directory
    DEDUP_CLUSTER_RADIUS_M   — DBSCAN epsilon in metres (default 2.0)
    LOG_LEVEL                — INFO / DEBUG
    LOG_FILE                 — optional log file path

Usage (CLI):
    python pipeline/orchestrator.py
        --video    data/raw/footage/survey_01.mp4
        --gps      data/raw/gps_logs/survey_01.gpx  (optional)
        [--device  cuda]
        [--fps     2.0]
        [--dry_run_db]
        [--resume]
        [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env constants
# ---------------------------------------------------------------------------
_WEIGHTS_DIR    = os.environ.get("WEIGHTS_DIR",    "ml/weights")
_RTDETR_WEIGHTS = os.environ.get("RTDETR_WEIGHTS", "best.pt")
_LOG_LEVEL      = os.environ.get("LOG_LEVEL",      "INFO")
_LOG_FILE       = os.environ.get("LOG_FILE",       "")
_DB_HOST        = os.environ.get("POSTGRES_HOST",     "localhost")
_DB_PORT        = int(os.environ.get("POSTGRES_PORT", "5432"))
_DB_NAME        = os.environ.get("POSTGRES_DB",       "cluj_monitor")
_DB_USER        = os.environ.get("POSTGRES_USER",     "postgres")
_DB_PASSWORD    = os.environ.get("POSTGRES_PASSWORD", "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorConfig:
    """
    Top-level configuration for one pipeline run.
    All API enrichment settings (skip_enrichment, skip_weather, skip_overpass)
    have been removed — enrichment is no longer part of the pipeline.
    """
    video_path:   str
    gps_path:     Optional[str] = None
    work_dir:     str           = "data/processed/sessions"
    session_id:   Optional[str] = None
    device:       str           = "auto"
    fps:          float         = 2.0
    dry_run_db:   bool          = False
    resume:       bool          = False
    save_debug:   bool          = False   # write per-stage debug images
    use_exact_mask_depth: bool  = False   # exact SAM mask in depth extraction
    keep_all_frames: bool       = False   # keep non-detection frames on disk


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name:      str
    skipped:   bool
    elapsed_s: float
    output:    Optional[str]
    error:     Optional[str]

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class SessionResult:
    session_id:    str
    video_path:    str
    gps_path:      Optional[str]
    started_at:    datetime
    finished_at:   Optional[datetime]
    status:        str
    stages:        List[StageResult] = field(default_factory=list)
    n_frames:      int = 0
    n_detections:  int = 0
    n_inserted:    int = 0
    n_updated:     int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "video_path":    self.video_path,
            "gps_path":      self.gps_path,
            "started_at":    self.started_at.isoformat(),
            "finished_at":   self.finished_at.isoformat() if self.finished_at else None,
            "status":        self.status,
            "n_frames":      self.n_frames,
            "n_detections":  self.n_detections,
            "n_inserted":    self.n_inserted,
            "n_updated":     self.n_updated,
            "error_message": self.error_message,
            "stages": [
                {
                    "name":      s.name,
                    "skipped":   s.skipped,
                    "elapsed_s": round(s.elapsed_s, 2),
                    "output":    s.output,
                    "error":     s.error,
                }
                for s in self.stages
            ],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """7-stage pipeline — enrichment permanently removed."""

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg

        if cfg.device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = cfg.device

        self.session_id = (
            cfg.session_id
            if cfg.session_id
            else datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir = Path(cfg.work_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Orchestrator | session=%s | device=%s | dir=%s",
            self.session_id, self._device, self.session_dir,
        )

    # ------------------------------------------------------------------
    # Public run
    # ------------------------------------------------------------------

    def run(self) -> SessionResult:
        started_at = datetime.now(tz=timezone.utc)
        result = SessionResult(
            session_id  = self.session_id,
            video_path  = self.cfg.video_path,
            gps_path    = self.cfg.gps_path,
            started_at  = started_at,
            finished_at = None,
            status      = "running",
        )
        log_id = self._survey_log_start(started_at)

        # Write session.json immediately (status="running", stages=[]) so the
        # status endpoint stops reporting "initialising" the moment the
        # orchestrator starts, and the frontend tracker shows Stage 1 running.
        self._flush_session(result)

        try:
            self._try_extract_gps()
            frames = self._run_stage1(
                result,
                self.session_dir / "01_manifest" / "manifest.json",
            )
            self._flush_session(result)
            det_results = self._run_stage2(
                result, frames,
                self.session_dir / "02_detections" / "detections.json",
            )
            self._flush_session(result)
            # Remove frames with no detections from disk (Stage 1 extracted all
            # of them; nothing downstream re-reads the non-detection frames).
            # Runs only after Stage 2 has fully completed.
            self._prune_nondetection_frames(det_results)
            seg_results = self._run_stage3(
                result, det_results,
                self.session_dir / "03_segmentations" / "segmentations.json",
            )
            self._flush_session(result)
            dep_results = self._run_stage4(
                result, seg_results,
                self.session_dir / "04_depth" / "depth_estimates.json",
            )
            self._flush_session(result)
            sev_results = self._run_stage5(
                result, dep_results,
                self.session_dir / "05_severity" / "severity_estimates.json",
            )
            self._flush_session(result)

            # Stages 6 and 7 need GPS coordinates (DBSCAN clustering + PostGIS
            # upsert). If no frame has a coordinate — no .gpx supplied and no
            # embedded GPS could be extracted — skip them and finish cleanly
            # rather than failing. Detection / segmentation / depth / severity
            # outputs (and their debug images) are still produced.
            gps_available = any(
                getattr(f, "latitude", None) is not None for f in frames
            )
            if gps_available:
                dedup_frames = self._run_stage6(
                    result, sev_results,
                    self.session_dir / "06_deduplicated" / "deduplicated.json",
                )
                self._flush_session(result)
                db_result = self._run_stage7(
                    result, dedup_frames,
                    self.session_dir / "07_db_write" / "db_write_summary.json",
                )
                result.n_inserted = db_result.n_inserted
                result.n_updated  = db_result.n_updated
                self._flush_session(result)
            else:
                logger.warning(
                    "No GPS coordinates in this run — skipping Stage 6 "
                    "(Deduplicator) and Stage 7 (DbWriter). Nothing is written "
                    "to the database."
                )
                result.stages.append(StageResult(
                    name="deduplicator", skipped=True, elapsed_s=0.0,
                    output=None, error=None,
                ))
                result.stages.append(StageResult(
                    name="db_writer", skipped=True, elapsed_s=0.0,
                    output=None, error=None,
                ))
                result.n_inserted = 0
                result.n_updated  = 0
            result.status     = "complete"

        except Exception as exc:
            result.status        = "failed"
            result.error_message = traceback.format_exc()
            logger.error("Pipeline failed: %s", exc)
            logger.debug(result.error_message)

        finally:
            result.finished_at = datetime.now(tz=timezone.utc)
            elapsed = (result.finished_at - result.started_at).total_seconds()
            logger.info(
                "=== Pipeline %s | %.1f s | frames=%d | dets=%d | "
                "inserted=%d | updated=%d ===",
                result.status.upper(), elapsed,
                result.n_frames, result.n_detections,
                result.n_inserted, result.n_updated,
            )
            session_json = self._flush_session(result)
            logger.info("Session summary: %s", session_json)
            self._survey_log_finish(log_id, result)

        return result

    # ------------------------------------------------------------------
    # Incremental progress write
    # ------------------------------------------------------------------

    def _flush_session(self, result: SessionResult) -> Path:
        """
        Write the current SessionResult snapshot to <session>/session.json.

        Called after every stage so the backend status endpoint (which reads
        this file) can report live, stage-by-stage progress to the frontend
        poller — not just the final state.

        The write is atomic: dump to a temporary file in the same directory,
        then os.replace() it over session.json. A concurrent reader therefore
        always sees either the previous complete document or the new complete
        document, never a half-written one.
        """
        session_json = self.session_dir / "session.json"
        tmp = session_json.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            os.replace(tmp, session_json)   # atomic on Windows and POSIX
        except OSError as exc:
            logger.warning("Could not flush session.json: %s", exc)
            tmp.unlink(missing_ok=True)
        return session_json

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_stage1(self, result: SessionResult, manifest_path: Path):
        from pipeline.preprocessor import Preprocessor, PreprocessorConfig
        if self.cfg.resume and manifest_path.exists():
            logger.info("Stage 1 — RESUME")
            frames = Preprocessor.load_manifest(str(manifest_path))
            result.n_frames = len(frames)
            result.stages.append(StageResult(
                name="preprocessor", skipped=True, elapsed_s=0.0,
                output=str(manifest_path), error=None,
            ))
            return frames
        t0 = time.perf_counter()
        try:
            gps_path = self.cfg.gps_path or ""
            if not gps_path or not Path(gps_path).exists():
                logger.warning(
                    "GPS file not found: '%s'. "
                    "Frames will have latitude=None / longitude=None. "
                    "DBSCAN and DB write will skip GPS-less detections.",
                    gps_path,
                )
            cfg = PreprocessorConfig(fps=self.cfg.fps)
            frames = Preprocessor(cfg).run(
                video_path = self.cfg.video_path,
                gps_path   = gps_path,
                output_dir = str(manifest_path.parent / "frames"),
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            Preprocessor.save_manifest(frames, str(manifest_path))
            result.n_frames = len(frames)
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="preprocessor", skipped=False, elapsed_s=elapsed,
                output=str(manifest_path), error=None,
            ))
            logger.info("Stage 1 complete: %d frames in %.1f s", len(frames), elapsed)
            return frames
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="preprocessor", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage2(self, result: SessionResult, frames: list, det_path: Path):
        from pipeline.detector import Detector, DetectorConfig
        if self.cfg.resume and det_path.exists():
            logger.info("Stage 2 — RESUME")
            det_results = Detector.load_detections(str(det_path))
            result.n_detections = sum(r.n_detections for r in det_results)
            result.stages.append(StageResult(
                name="detector", skipped=True, elapsed_s=0.0,
                output=str(det_path), error=None,
            ))
            return det_results
        t0 = time.perf_counter()
        try:
            weights = str(Path(_WEIGHTS_DIR) / _RTDETR_WEIGHTS)
            cfg = DetectorConfig(
                weights=weights, device=self._device,
                save_debug=self.cfg.save_debug,
            )
            det_results = Detector(cfg).run(frames, output_dir=str(det_path.parent))
            result.n_detections = sum(r.n_detections for r in det_results)
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="detector", skipped=False, elapsed_s=elapsed,
                output=str(det_path), error=None,
            ))
            logger.info(
                "Stage 2 complete: %d detections in %.1f s",
                result.n_detections, elapsed,
            )
            return det_results
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="detector", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise


    # In Orchestrator.__init__ or run() — NO change needed to existing code.
# Add this ONE method to the class:

    def _try_extract_gps(self) -> None:
        """
        If no GPX was provided, attempt to extract one from the MP4.
        On success:  self.cfg.gps_path is updated to the extracted .gpx path.
        On failure:  self.cfg.gps_path stays None — stages 6/7 will be skipped
                    by the existing GPS-guard already in _run_stage1().
        This method never raises — all errors are logged as warnings.
        """
        if self.cfg.gps_path and Path(self.cfg.gps_path).exists():
            return  # GPX already provided — nothing to do

        logger.info(
            "No GPX provided — attempting embedded GPS extraction from %s",
            self.cfg.video_path,
        )
        try:
            # Import BOTH helpers from the same module so a single GPSPoint
            # type flows from extract() into _build_gpx_xml(). The pipeline
            # copy is the canonical, fuller implementation (it also defines
            # _build_gpx_xml); sourcing them from different module roots
            # (pipeline vs scripts) risked a GPSPoint-type mismatch.
            from pipeline.extract_gpx_from_video import extract, _build_gpx_xml
            mp4_path = Path(self.cfg.video_path)
            points, strategy = extract(mp4_path)

            # Write the .gpx alongside the video file
            gpx_path = mp4_path.with_suffix(".gpx")
            gpx_xml = _build_gpx_xml(points, source_name=mp4_path.stem)
            gpx_path.write_text(gpx_xml, encoding="utf-8")

            self.cfg.gps_path = str(gpx_path)
            logger.info(
                "Embedded GPS extracted via [%s] — %d points → %s",
                strategy, len(points), gpx_path,
            )
        except Exception as exc:
            logger.warning(
                "Embedded GPS extraction failed (%s). "
                "Stages 6 and 7 will be skipped.",
                exc,
            )
            self.cfg.gps_path = None        

    def _prune_nondetection_frames(self, det_results: list) -> None:
        """
        Delete extracted frames that have no accepted detection.

        Stage 1 writes every extracted frame to <session>/01_manifest/frames/.
        Stages 3 and 4 only read frames that have detections, and the Stage 2
        debug overlays (when enabled) are already written by the time this is
        called, so the non-detection frames are safe to remove. The end state
        on disk is detection frames only. Disabled with keep_all_frames=True.

        Safety guards:
          - never runs when keep_all_frames is set
          - never runs when the whole session has zero detections (otherwise it
            would empty the entire frames directory)
          - only deletes image files inside the session frames directory
          - never deletes a frame referenced by a detection

        Note: manifest.json is not rewritten, so it still lists every extracted
        frame. A resume that re-runs Stage 2 from scratch would need the frames
        re-extracted; resuming from detections.json is unaffected.
        """
        if self.cfg.keep_all_frames:
            logger.info("Frame prune: disabled (keep_all_frames=True)")
            return

        frames_dir = self.session_dir / "01_manifest" / "frames"
        if not frames_dir.is_dir():
            logger.debug(
                "Frame prune: frames dir not found (%s) — nothing to do",
                frames_dir,
            )
            return

        keep = {
            Path(r.frame_path).name
            for r in det_results
            if r.has_detections
        }

        if not keep:
            logger.warning(
                "Frame prune: 0 frames have detections — skipping prune so the "
                "frames directory is not emptied. Inspect the run first."
            )
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        removed = kept = 0
        freed_bytes = 0
        for p in frames_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in image_exts:
                continue
            if p.name in keep:
                kept += 1
                continue
            try:
                freed_bytes += p.stat().st_size
                p.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Frame prune: could not delete %s (%s)", p, exc)

        logger.info(
            "Frame prune: kept %d detection frame(s), removed %d "
            "(%.1f MB freed). manifest.json still lists all extracted frames.",
            kept, removed, freed_bytes / 1e6,
        )

    def _run_stage3(self, result: SessionResult, det_results: list, seg_path: Path):
        from pipeline.segmentor import Segmentor, SegmentorConfig
        if self.cfg.resume and seg_path.exists():
            logger.info("Stage 3 — RESUME")
            seg_results = Segmentor.load_segmentations(str(seg_path))
            result.stages.append(StageResult(
                name="segmentor", skipped=True, elapsed_s=0.0,
                output=str(seg_path), error=None,
            ))
            return seg_results
        t0 = time.perf_counter()
        try:
            sam_weights = str(Path(_WEIGHTS_DIR) / "sam2.1_hiera_tiny.pt")
            cfg = SegmentorConfig(
                weights=sam_weights, device=self._device,
                save_debug=self.cfg.save_debug,
            )
            seg_results = Segmentor(cfg).run(
                det_results, output_dir=str(seg_path.parent)
            )
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="segmentor", skipped=False, elapsed_s=elapsed,
                output=str(seg_path), error=None,
            ))
            logger.info("Stage 3 complete in %.1f s", elapsed)
            return seg_results
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="segmentor", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage4(self, result: SessionResult, seg_results: list, depth_path: Path):
        from pipeline.depth_estimator import DepthEstimator, DepthEstimatorConfig
        if self.cfg.resume and depth_path.exists():
            logger.info("Stage 4 — RESUME")
            dep_results = DepthEstimator.load_depth_estimates(str(depth_path))
            result.stages.append(StageResult(
                name="depth_estimator", skipped=True, elapsed_s=0.0,
                output=str(depth_path), error=None,
            ))
            return dep_results
        t0 = time.perf_counter()
        try:
            cfg = DepthEstimatorConfig(
                monodepth_root = os.environ.get(
                    "MONODEPTH_ROOT",
                    r"C:\Facultate\pothole-detection\Monodepth",
                ),
                weights_dir = os.environ.get(
                    "MONODEPTH_WEIGHTS_DIR",
                    r"C:\Facultate\pothole-detection\Pothole-Detection\ml\weights\mono_640x192",
                ),
                device = self._device,
                save_debug = self.cfg.save_debug,
                use_exact_mask_depth = self.cfg.use_exact_mask_depth,
            )
            dep_results = DepthEstimator(cfg).run(
                seg_results, output_dir=str(depth_path.parent)
            )
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="depth_estimator", skipped=False, elapsed_s=elapsed,
                output=str(depth_path), error=None,
            ))
            logger.info("Stage 4 complete in %.1f s", elapsed)
            return dep_results
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="depth_estimator", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage5(self, result: SessionResult, dep_results: list, sev_path: Path):
        from pipeline.severity_classifier import SeverityClassifier, SeverityConfig
        if self.cfg.resume and sev_path.exists():
            logger.info("Stage 5 — RESUME")
            sev_results = SeverityClassifier.load_severity(str(sev_path))
            result.stages.append(StageResult(
                name="severity_classifier", skipped=True, elapsed_s=0.0,
                output=str(sev_path), error=None,
            ))
            return sev_results
        t0 = time.perf_counter()
        try:
            cfg = SeverityConfig()
            sev_results = SeverityClassifier(cfg).run(
                dep_results, output_dir=str(sev_path.parent)
            )
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="severity_classifier", skipped=False, elapsed_s=elapsed,
                output=str(sev_path), error=None,
            ))
            logger.info("Stage 5 complete in %.1f s", elapsed)
            return sev_results
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="severity_classifier", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage6(
        self,
        result:      SessionResult,
        sev_results: list,
        dedup_path:  Path,
    ) -> List[dict]:
        """
        Stage 6 — Deduplicator.
        Receives SeverityResult objects (from Stage 5) directly.
        No enricher passthrough needed — sev_results.to_dict() already
        contains latitude, longitude, and all required box fields.
        """
        from pipeline.deduplicator import Deduplicator, DeduplicatorConfig
        if self.cfg.resume and dedup_path.exists():
            logger.info("Stage 6 — RESUME")
            dedup_frames = Deduplicator.load_deduplicated(str(dedup_path))
            result.stages.append(StageResult(
                name="deduplicator", skipped=True, elapsed_s=0.0,
                output=str(dedup_path), error=None,
            ))
            return dedup_frames
        t0 = time.perf_counter()
        try:
            sev_dicts = [r.to_dict() for r in sev_results]
            cfg = DeduplicatorConfig()
            dedup_results = Deduplicator(cfg).run(
                sev_dicts, output_dir=str(dedup_path.parent)
            )
            dedup_frames = [r.to_dict() for r in dedup_results]
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="deduplicator", skipped=False, elapsed_s=elapsed,
                output=str(dedup_path), error=None,
            ))
            logger.info("Stage 6 complete in %.1f s", elapsed)
            return dedup_frames
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="deduplicator", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage7(
        self,
        result:       SessionResult,
        dedup_frames: List[dict],
        db_path:      Path,
    ):
        """Stage 7 — DbWriter."""
        from pipeline.db_writer import DbWriter, DbWriterConfig
        t0 = time.perf_counter()
        try:
            cfg = DbWriterConfig(dry_run=self.cfg.dry_run_db)
            db_result = DbWriter(cfg).run(
                dedup_frames, output_dir=str(db_path.parent)
            )
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="db_writer", skipped=False, elapsed_s=elapsed,
                output=str(db_path), error=None,
            ))
            logger.info(
                "Stage 7 complete: inserted=%d updated=%d in %.1f s",
                db_result.n_inserted, db_result.n_updated, elapsed,
            )
            return db_result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="db_writer", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    # ------------------------------------------------------------------
    # survey_log
    # ------------------------------------------------------------------

    def _survey_log_start(self, started_at: datetime) -> Optional[int]:
        if self.cfg.dry_run_db:
            logger.info("survey_log: DRY RUN — no row inserted")
            return None
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=_DB_HOST, port=_DB_PORT,
                dbname=_DB_NAME, user=_DB_USER, password=_DB_PASSWORD,
            )
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO survey_log
                    (survey_date, started_at, status, video_files)
                VALUES (%s, %s, 'running', %s)
                ON CONFLICT (survey_date) DO UPDATE
                    SET status      = 'running',
                        video_files = COALESCE(survey_log.video_files, '[]'::jsonb)
                                      || EXCLUDED.video_files
                RETURNING id;
                """,
                (
                    started_at.date(),
                    started_at,
                    json.dumps([self.cfg.video_path]),
                ),
            )
            row_id = cur.fetchone()[0]
            cur.close()
            conn.close()
            logger.info("survey_log: row id=%d", row_id)
            return row_id
        except Exception as exc:
            logger.warning("survey_log start failed: %s", exc)
            return None

    def _survey_log_finish(
        self,
        log_id: Optional[int],
        result: SessionResult,
    ) -> None:
        if log_id is None or self.cfg.dry_run_db:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=_DB_HOST, port=_DB_PORT,
                dbname=_DB_NAME, user=_DB_USER, password=_DB_PASSWORD,
            )
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE survey_log
                SET    finished_at        = %s,
                       status             = %s,
                       frames_processed   = COALESCE(frames_processed,   0) + %s,
                       detections_found   = COALESCE(detections_found,   0) + %s,
                       new_detections     = COALESCE(new_detections,     0) + %s,
                       updated_detections = COALESCE(updated_detections, 0) + %s,
                       error_message      = %s
                WHERE  id = %s;
                """,
                (
                    result.finished_at,
                    result.status,
                    result.n_frames,
                    result.n_detections,
                    result.n_inserted,
                    result.n_updated,
                    result.error_message,
                    log_id,
                ),
            )
            cur.close()
            conn.close()
            logger.info("survey_log: row %d → %s", log_id, result.status)
        except Exception as exc:
            logger.warning("survey_log finish failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers / logging / CLI
# ---------------------------------------------------------------------------

def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _setup_logging(verbose: bool) -> None:
    level    = logging.DEBUG if verbose else getattr(logging, _LOG_LEVEL, logging.INFO)
    handlers: list = [logging.StreamHandler()]
    if _LOG_FILE:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))
    logging.basicConfig(
        level    = level,
        format   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RIDS pipeline (Stages 1–8). "
            "Enrichment permanently removed. "
            "DB credentials from .env."
        )
    )
    parser.add_argument("--video",      required=True)
    parser.add_argument("--gps",        default=None)
    parser.add_argument("--work_dir",   default="data/processed/sessions")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--fps",        type=float, default=2.0)
    parser.add_argument("--dry_run_db", action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument(
        "--save_debug", action="store_true",
        help="Write per-stage debug images (boxes / mask overlay / disparity) "
             "to <stage>/debug/ for frames that have detections",
    )
    parser.add_argument(
        "--exact_mask_depth", action="store_true",
        help="Use the exact SAM mask in Stage 4 depth extraction (changes "
             "depth_norm / depth_confidence vs the validated default)",
    )
    parser.add_argument(
        "--keep_all_frames", action="store_true",
        help="Keep non-detection frames on disk (default: prune them after "
             "Stage 2 so only detection frames remain)",
    )
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    cfg = OrchestratorConfig(
        video_path  = args.video,
        gps_path    = args.gps,
        work_dir    = args.work_dir,
        session_id  = args.session_id,
        device      = args.device,
        fps         = args.fps,
        dry_run_db  = args.dry_run_db,
        resume      = args.resume,
        save_debug  = args.save_debug,
        use_exact_mask_depth = args.exact_mask_depth,
        keep_all_frames      = args.keep_all_frames,
    )

    result = Orchestrator(cfg).run()

    if result.status == "failed":
        logger.error("Pipeline failed. See session.json for details.")
        sys.exit(1)

    logger.info(
        "Pipeline complete | session=%s | frames=%d | dets=%d | "
        "inserted=%d | updated=%d",
        result.session_id, result.n_frames, result.n_detections,
        result.n_inserted, result.n_updated,
    )


if __name__ == "__main__":
    main()