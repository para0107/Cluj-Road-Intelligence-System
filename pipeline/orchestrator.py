"""
pipeline/orchestrator.py
------------------------
End-to-end coordinator for the Road Infrastructure Detection System (RIDS)
nine-stage inference pipeline.

Stage sequence
--------------
  1  Preprocessor       -- frame extraction, GPS sync, lighting
  2  Detector           -- RT-DETR-L inference, per-class thresholds
  3  Segmentor          -- SAM 2.1 Tiny, 4 geometry features
  4  DepthEstimator     -- Monodepth2 relative depth, proxy fallback
  5  SeverityClassifier -- rule-based S1–S5 weighted multi-signal
  6  Enricher           -- Nominatim + Overpass + Open-Meteo
  7  Deduplicator       -- DBSCAN spatial clustering, 2 m radius
  8  DbWriter           -- PostgreSQL/PostGIS upsert
  9  survey_log write   -- record session outcome in survey_log table

Design principles
-----------------
- Each stage's output is serialised to disk before the next stage begins.
  This means any stage can be re-run in isolation without repeating the
  expensive stages before it (e.g. re-running severity only, or DB write).
- All configuration is read from .env via python-dotenv. No secrets are
  hardcoded. Stage-specific config objects are constructed from .env values
  and defaults; no argument is passed directly from the CLI except paths
  and flags.
- The survey_log table is updated at the start (status='running') and end
  (status='complete' or 'failed') of every run. If the orchestrator crashes
  mid-run, the log row is left as 'running' and can be manually inspected.
- Per-stage intermediate outputs are saved under:
    <work_dir>/<session_id>/
        01_manifest/        manifest.json
        02_detections/      detections.json
        03_segmentations/   segmentations.json
        04_depth/           depth_estimates.json
        05_severity/        severity_estimates.json
        06_enriched/        enriched.json
        07_deduplicated/    deduplicated.json + dedup_report.html
        08_db_write/        db_write_summary.json
        session.json        overall session summary

Environment variables used (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    WEIGHTS_DIR           -- base directory for model weights
    RTDETR_WEIGHTS        -- RT-DETR checkpoint filename (under WEIGHTS_DIR)
    NOMINATIM_USER_AGENT  -- required by Nominatim usage policy
    OSM_OVERPASS_URL      -- Overpass endpoint
    OPEN_METEO_BASE_URL   -- Open-Meteo endpoint
    DEDUP_CLUSTER_RADIUS_M       -- DBSCAN epsilon
    SURROUNDING_DENSITY_RADIUS_M -- density search radius
    LOG_LEVEL             -- logging level (INFO / DEBUG)
    LOG_FILE              -- path to log file (optional)

Usage (module):
    from pipeline.orchestrator import Orchestrator, OrchestratorConfig
    cfg = OrchestratorConfig(
        video_path  = "data/raw/footage/survey_01.mp4",
        gps_path    = "data/raw/gps_logs/survey_01.gpx",
        work_dir    = "data/processed/sessions/",
        device      = "cuda",
    )
    result = Orchestrator(cfg).run()

Usage (CLI):
    python pipeline/orchestrator.py
        --video   data/raw/footage/survey_01.mp4
        --gps     data/raw/gps_logs/survey_01.gpx
        --work_dir data/processed/sessions/
        [--device cuda]
        [--skip_enrichment]
        [--skip_weather]
        [--skip_overpass]
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
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env first — all os.environ reads below depend on it
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env-derived constants
# ---------------------------------------------------------------------------
_WEIGHTS_DIR    = os.environ.get("WEIGHTS_DIR",     "ml/weights")
_RTDETR_WEIGHTS = os.environ.get("RTDETR_WEIGHTS",  "rtdetr_l_nrdd2024.pt")
_LOG_LEVEL      = os.environ.get("LOG_LEVEL",       "INFO")
_LOG_FILE       = os.environ.get("LOG_FILE",        "")

# DB connection — read by DbWriterConfig.__post_init__ automatically,
# but also needed here for survey_log writes
_DB_HOST     = os.environ.get("POSTGRES_HOST",     "localhost")
_DB_PORT     = int(os.environ.get("POSTGRES_PORT", "5432"))
_DB_NAME     = os.environ.get("POSTGRES_DB",       "cluj_monitor")
_DB_USER     = os.environ.get("POSTGRES_USER",     "postgres")
_DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorConfig:
    """
    Top-level configuration for one pipeline run.

    video_path:
        Absolute or relative path to the dashcam .mp4 file.

    gps_path:
        Absolute or relative path to the .gpx GPS telemetry file.
        Pass an empty string or None if GPS is unavailable; Stages 1–5
        will run without coordinates and Stages 6–8 will skip GPS-dependent
        processing gracefully.

    work_dir:
        Root directory for per-session working directories.
        Each run creates a subdirectory named by session_id.

    session_id:
        Unique identifier for this run. Auto-generated as
        YYYYMMDD_HHMMSS if not provided.

    device:
        PyTorch inference device for Stages 2, 3, 4.
        "auto" selects CUDA if available, otherwise CPU.
        Explicit values: "cpu", "cuda", "cuda:0".

    fps:
        Frame extraction rate for Stage 1 (default 2.0).

    skip_enrichment:
        If True, skip Stage 6 entirely and pass Stage 5 output directly
        to Stage 7. All enrichment fields will be None.

    skip_weather:
        Passed to Enricher — skip Open-Meteo calls.

    skip_overpass:
        Passed to Enricher — skip OSM Overpass calls.

    dry_run_db:
        If True, Stage 8 parses all rows and logs what would be written
        but does not open a database connection or execute any SQL.
        Safe to use without a running PostgreSQL instance.

    resume:
        If True and a previous session directory for the same video exists,
        skip stages whose output JSON already exists on disk. Allows
        continuing an interrupted run without repeating expensive stages.
    """
    video_path:       str
    gps_path:         Optional[str] = None
    work_dir:         str           = "data/processed/sessions"
    session_id:       Optional[str] = None
    device:           str           = "auto"
    fps:              float         = 2.0
    skip_enrichment:  bool          = False
    skip_weather:     bool          = False
    skip_overpass:    bool          = False
    dry_run_db:       bool          = False
    resume:           bool          = False


# ---------------------------------------------------------------------------
# Session result
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name:     str
    skipped:  bool
    elapsed_s: float
    output:   Optional[str]   # path to output JSON or None
    error:    Optional[str]   # exception message if stage failed

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class SessionResult:
    session_id:     str
    video_path:     str
    gps_path:       Optional[str]
    started_at:     datetime
    finished_at:    Optional[datetime]
    status:         str            # "complete" | "failed" | "running"
    stages:         List[StageResult] = field(default_factory=list)
    n_frames:       int = 0
    n_detections:   int = 0
    n_inserted:     int = 0
    n_updated:      int = 0
    error_message:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "video_path":   self.video_path,
            "gps_path":     self.gps_path,
            "started_at":   self.started_at.isoformat(),
            "finished_at":  self.finished_at.isoformat() if self.finished_at else None,
            "status":       self.status,
            "n_frames":     self.n_frames,
            "n_detections": self.n_detections,
            "n_inserted":   self.n_inserted,
            "n_updated":    self.n_updated,
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
    """
    Chains all nine pipeline stages into a single run.

    Each stage is called with the precise output type of the previous stage.
    Intermediate results are serialised to disk between stages so any stage
    can be resumed independently.
    """

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg

        # Resolve device
        if cfg.device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = cfg.device

        # Generate session_id if not provided
        if not cfg.session_id:
            self.session_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        else:
            self.session_id = cfg.session_id

        # Session working directory
        self.session_dir = Path(cfg.work_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Orchestrator initialised | session=%s | device=%s | session_dir=%s",
            self.session_id, self._device, self.session_dir,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> SessionResult:
        """
        Execute all nine pipeline stages in sequence.

        Returns
        -------
        SessionResult with per-stage timing, output paths, and final counts.
        """
        started_at = datetime.now(tz=timezone.utc)
        result = SessionResult(
            session_id  = self.session_id,
            video_path  = self.cfg.video_path,
            gps_path    = self.cfg.gps_path,
            started_at  = started_at,
            finished_at = None,
            status      = "running",
        )

        # Write survey_log row as "running" so the scheduler can track it
        log_id = self._survey_log_start(started_at)

        try:
            # ── Stage 1: Preprocessor ──────────────────────────────────
            manifest_path = self.session_dir / "01_manifest" / "manifest.json"
            frames = self._run_stage1(result, manifest_path)

            # ── Stage 2: Detector ──────────────────────────────────────
            detections_path = self.session_dir / "02_detections" / "detections.json"
            det_results = self._run_stage2(result, frames, detections_path)

            # ── Stage 3: Segmentor ─────────────────────────────────────
            seg_path = self.session_dir / "03_segmentations" / "segmentations.json"
            seg_results = self._run_stage3(result, det_results, seg_path)

            # ── Stage 4: DepthEstimator ────────────────────────────────
            depth_path = self.session_dir / "04_depth" / "depth_estimates.json"
            dep_results = self._run_stage4(result, seg_results, depth_path)

            # ── Stage 5: SeverityClassifier ────────────────────────────
            sev_path = self.session_dir / "05_severity" / "severity_estimates.json"
            sev_results = self._run_stage5(result, dep_results, sev_path)

            # ── Stage 6: Enricher (optional) ──────────────────────────
            enr_path = self.session_dir / "06_enriched" / "enriched.json"
            if self.cfg.skip_enrichment:
                logger.info("Stage 6 — Enrichment SKIPPED (--skip_enrichment)")
                # Convert severity frames to the flat dict format that
                # Enricher normally produces, so Stage 7 sees the same shape
                enr_frames = self._severity_to_enricher_passthrough(sev_results)
                _write_json(enr_frames, enr_path)
                result.stages.append(StageResult(
                    name="enricher", skipped=True, elapsed_s=0.0,
                    output=str(enr_path), error=None,
                ))
            else:
                enr_frames = self._run_stage6(result, sev_results, enr_path)

            # ── Stage 7: Deduplicator ──────────────────────────────────
            dedup_path = self.session_dir / "07_deduplicated" / "deduplicated.json"
            dedup_frames = self._run_stage7(result, enr_frames, dedup_path)

            # ── Stage 8: DbWriter ──────────────────────────────────────
            db_path = self.session_dir / "08_db_write" / "db_write_summary.json"
            db_result = self._run_stage8(result, dedup_frames, db_path)

            # Propagate DB counts into session result
            result.n_inserted = db_result.n_inserted
            result.n_updated  = db_result.n_updated

            result.status = "complete"

        except Exception as exc:
            result.status        = "failed"
            result.error_message = traceback.format_exc()
            logger.error("Pipeline failed: %s", exc)
            logger.debug(result.error_message)

        finally:
            result.finished_at = datetime.now(tz=timezone.utc)
            elapsed_total = (result.finished_at - result.started_at).total_seconds()
            logger.info(
                "=== Pipeline %s | elapsed=%.1f s | frames=%d | "
                "detections=%d | inserted=%d | updated=%d ===",
                result.status.upper(),
                elapsed_total,
                result.n_frames,
                result.n_detections,
                result.n_inserted,
                result.n_updated,
            )

            # Save session.json
            session_json = self.session_dir / "session.json"
            with session_json.open("w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Session summary: %s", session_json)

            # Update survey_log
            self._survey_log_finish(log_id, result)

        return result

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_stage1(
        self,
        result:        SessionResult,
        manifest_path: Path,
    ):
        """Stage 1 — Preprocessor."""
        from pipeline.preprocessor import Preprocessor, PreprocessorConfig

        # Resume: if manifest already exists, load and skip extraction
        if self.cfg.resume and manifest_path.exists():
            logger.info("Stage 1 — RESUME: manifest exists, loading from disk")
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
                    "GPS file not found or not provided: '%s'. "
                    "All frames will have latitude=None, longitude=None. "
                    "Stages 6–8 will skip GPS-dependent processing.",
                    gps_path,
                )

            cfg = PreprocessorConfig(fps=self.cfg.fps)
            pp  = Preprocessor(cfg)
            frames = pp.run(
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

    def _run_stage2(
        self,
        result:          SessionResult,
        frames:          list,
        detections_path: Path,
    ):
        """Stage 2 — Detector."""
        from pipeline.detector import Detector, DetectorConfig

        if self.cfg.resume and detections_path.exists():
            logger.info("Stage 2 — RESUME: detections.json exists, loading from disk")
            det_results = Detector.load_detections(str(detections_path))
            result.stages.append(StageResult(
                name="detector", skipped=True, elapsed_s=0.0,
                output=str(detections_path), error=None,
            ))
            return det_results

        t0 = time.perf_counter()
        try:
            weights = str(Path(_WEIGHTS_DIR) / _RTDETR_WEIGHTS)
            cfg = DetectorConfig(weights=weights, device=self._device)
            det_results = Detector(cfg).run(
                frames,
                output_dir=str(detections_path.parent),
            )
            result.n_detections = sum(r.n_detections for r in det_results)
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="detector", skipped=False, elapsed_s=elapsed,
                output=str(detections_path), error=None,
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

    def _run_stage3(
        self,
        result:      SessionResult,
        det_results: list,
        seg_path:    Path,
    ):
        """Stage 3 — Segmentor (SAM 2.1)."""
        from pipeline.segmentor import Segmentor, SegmentorConfig

        if self.cfg.resume and seg_path.exists():
            logger.info("Stage 3 — RESUME: segmentations.json exists, loading from disk")
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
                weights = sam_weights,
                device  = self._device,
            )
            seg_results = Segmentor(cfg).run(
                det_results,
                output_dir=str(seg_path.parent),
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

    def _run_stage4(
        self,
        result:      SessionResult,
        seg_results: list,
        depth_path:  Path,
    ):
        """Stage 4 — DepthEstimator (Monodepth2)."""
        from pipeline.depth_estimator import DepthEstimator, DepthEstimatorConfig

        if self.cfg.resume and depth_path.exists():
            logger.info("Stage 4 — RESUME: depth_estimates.json exists, loading from disk")
            dep_results = DepthEstimator.load_depth_estimates(str(depth_path))
            result.stages.append(StageResult(
                name="depth_estimator", skipped=True, elapsed_s=0.0,
                output=str(depth_path), error=None,
            ))
            return dep_results

        t0 = time.perf_counter()
        try:
            # Monodepth2 repo path and weights dir come from the existing
            # DepthEstimatorConfig defaults, which already encode the correct paths.
            # No .env override needed — the paths are static per-machine constants.
            cfg = DepthEstimatorConfig(
                monodepth_root = os.environ.get(
                    'MONODEPTH_ROOT',
                    r'C:\Facultate\pothole-detection\Monodepth',
                ),
                weights_dir    = os.environ.get(
                    'MONODEPTH_WEIGHTS_DIR',
                    r'C:\Facultate\pothole-detection\Pothole-Detection\ml\weights\mono_640x192',
                ),
                device         = self._device,
            )
            dep_results = DepthEstimator(cfg).run(
                seg_results,
                output_dir=str(depth_path.parent),
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

    def _run_stage5(
        self,
        result:      SessionResult,
        dep_results: list,
        sev_path:    Path,
    ):
        """Stage 5 — SeverityClassifier."""
        from pipeline.severity_classifier import SeverityClassifier, SeverityConfig

        if self.cfg.resume and sev_path.exists():
            logger.info("Stage 5 — RESUME: severity_estimates.json exists, loading from disk")
            from pipeline.severity_classifier import SeverityClassifier
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
                dep_results,
                output_dir=str(sev_path.parent),
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
        enr_path:    Path,
    ) -> List[dict]:
        """Stage 6 — Enricher."""
        from pipeline.enricher import Enricher, EnricherConfig

        if self.cfg.resume and enr_path.exists():
            logger.info("Stage 6 — RESUME: enriched.json exists, loading from disk")
            enr_frames = Enricher.load_enriched(str(enr_path))
            result.stages.append(StageResult(
                name="enricher", skipped=True, elapsed_s=0.0,
                output=str(enr_path), error=None,
            ))
            return enr_frames

        t0 = time.perf_counter()
        try:
            # Convert SeverityResult objects → flat dicts that Enricher expects
            frames_as_dicts = self._severity_to_enricher_passthrough(sev_results)

            cfg = EnricherConfig(
                skip_weather  = self.cfg.skip_weather,
                skip_overpass = self.cfg.skip_overpass,
            )
            enr_results = Enricher(cfg).run(
                frames_as_dicts,
                output_dir=str(enr_path.parent),
            )
            # Enricher.run returns List[EnrichmentResult]; convert to dicts
            enr_frames = [r.to_dict() for r in enr_results]
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="enricher", skipped=False, elapsed_s=elapsed,
                output=str(enr_path), error=None,
            ))
            logger.info("Stage 6 complete in %.1f s", elapsed)
            return enr_frames

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="enricher", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage7(
        self,
        result:     SessionResult,
        enr_frames: List[dict],
        dedup_path: Path,
    ) -> List[dict]:
        """Stage 7 — Deduplicator."""
        from pipeline.deduplicator import Deduplicator, DeduplicatorConfig

        if self.cfg.resume and dedup_path.exists():
            logger.info("Stage 7 — RESUME: deduplicated.json exists, loading from disk")
            dedup_frames = Deduplicator.load_deduplicated(str(dedup_path))
            result.stages.append(StageResult(
                name="deduplicator", skipped=True, elapsed_s=0.0,
                output=str(dedup_path), error=None,
            ))
            return dedup_frames

        t0 = time.perf_counter()
        try:
            cfg = DeduplicatorConfig()
            dedup_results = Deduplicator(cfg).run(
                enr_frames,
                output_dir=str(dedup_path.parent),
            )
            # Deduplicator.run returns List[DeduplicationResult]; convert to dicts
            dedup_frames = [r.to_dict() for r in dedup_results]
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="deduplicator", skipped=False, elapsed_s=elapsed,
                output=str(dedup_path), error=None,
            ))
            logger.info("Stage 7 complete in %.1f s", elapsed)
            return dedup_frames

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="deduplicator", skipped=False, elapsed_s=elapsed,
                output=None, error=str(exc),
            ))
            raise

    def _run_stage8(
        self,
        result:       SessionResult,
        dedup_frames: List[dict],
        db_path:      Path,
    ):
        """Stage 8 — DbWriter."""
        from pipeline.db_writer import DbWriter, DbWriterConfig

        t0 = time.perf_counter()
        try:
            cfg = DbWriterConfig(dry_run=self.cfg.dry_run_db)
            db_result = DbWriter(cfg).run(
                dedup_frames,
                output_dir=str(db_path.parent),
            )
            elapsed = time.perf_counter() - t0
            result.stages.append(StageResult(
                name="db_writer", skipped=False, elapsed_s=elapsed,
                output=str(db_path), error=None,
            ))
            logger.info(
                "Stage 8 complete: inserted=%d updated=%d in %.1f s",
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
    # survey_log writes (Stage 9)
    # ------------------------------------------------------------------

    def _survey_log_start(self, started_at: datetime) -> Optional[int]:
        """
        Insert a 'running' row into survey_log.
        Returns the row ID or None if the DB is unavailable / dry_run.
        """
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
            logger.info("survey_log: started row id=%d", row_id)
            return row_id

        except Exception as exc:
            logger.warning("survey_log: could not insert start row: %s", exc)
            return None

    def _survey_log_finish(
        self,
        log_id:  Optional[int],
        result:  SessionResult,
    ) -> None:
        """Update the survey_log row with final counts and status."""
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
                SET    finished_at         = %s,
                       status              = %s,
                       frames_processed    = %s,
                       detections_found    = %s,
                       new_detections      = %s,
                       updated_detections  = %s,
                       error_message       = %s
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
            logger.info(
                "survey_log: row %d updated → status=%s", log_id, result.status
            )

        except Exception as exc:
            logger.warning("survey_log: could not update row %d: %s", log_id, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _severity_to_enricher_passthrough(self, sev_results: list) -> List[dict]:
        """
        Convert a list of SeverityResult objects into the flat frame-dict format
        that Enricher.run() expects (same shape as severity_results.json["frames"]).

        This is needed because Enricher reads the same JSON format that
        SeverityClassifier.save_severity() writes, but the orchestrator passes
        live Python objects between stages to avoid redundant disk reads.
        """
        frames = []
        for r in sev_results:
            frames.append(r.to_dict())
        return frames


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else getattr(logging, _LOG_LEVEL, logging.INFO)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RIDS end-to-end pipeline orchestrator (Stages 1–9). "
            "All DB credentials and API settings are read from .env."
        )
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to dashcam .mp4 file",
    )
    parser.add_argument(
        "--gps", default=None,
        help="Path to GPS .gpx file (optional; enrichment/dedup skipped if absent)",
    )
    parser.add_argument(
        "--work_dir", default="data/processed/sessions",
        help="Root directory for per-session working directories",
    )
    parser.add_argument(
        "--session_id", default=None,
        help="Session identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device: auto | cpu | cuda | cuda:0  (default: auto)",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frame extraction rate in frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--skip_enrichment", action="store_true",
        help="Skip Stage 6 entirely (all enrichment fields will be null)",
    )
    parser.add_argument(
        "--skip_weather", action="store_true",
        help="Skip Open-Meteo weather lookups in Stage 6",
    )
    parser.add_argument(
        "--skip_overpass", action="store_true",
        help="Skip OSM Overpass lookups in Stage 6",
    )
    parser.add_argument(
        "--dry_run_db", action="store_true",
        help="Stage 8: parse and log without writing to the database",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run: skip stages whose output already exists",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    cfg = OrchestratorConfig(
        video_path       = args.video,
        gps_path         = args.gps,
        work_dir         = args.work_dir,
        session_id       = args.session_id,
        device           = args.device,
        fps              = args.fps,
        skip_enrichment  = args.skip_enrichment,
        skip_weather     = args.skip_weather,
        skip_overpass    = args.skip_overpass,
        dry_run_db       = args.dry_run_db,
        resume           = args.resume,
    )

    result = Orchestrator(cfg).run()

    # Exit with non-zero code if the pipeline failed
    if result.status == "failed":
        logger.error("Pipeline failed. See session.json for details.")
        sys.exit(1)

    logger.info(
        "Pipeline complete. Session: %s  "
        "Frames: %d  Detections: %d  Inserted: %d  Updated: %d",
        result.session_id,
        result.n_frames,
        result.n_detections,
        result.n_inserted,
        result.n_updated,
    )


if __name__ == "__main__":
    main()