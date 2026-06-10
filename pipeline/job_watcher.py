"""
pipeline/job_watcher.py
-----------------------
Runs on the Windows HOST machine (outside Docker).
Watches data/jobs/ for new job-request JSON files written by the backend
container and executes orchestrator.py for each one with --device cuda.

This is the correct execution boundary: the orchestrator requires the host
GPU (RTX 2050), the host Python environment (PyTorch + CUDA, Ultralytics,
SAM 2.1, Monodepth2), and Windows absolute paths for model weights.
The Docker backend container cannot satisfy any of these requirements.

Usage
-----
From the project root on the Windows host:

    python pipeline/job_watcher.py

Or as a background process (run once, keep alive):

    start /B pythonw pipeline\\job_watcher.py

Environment variables (loaded from .env automatically):
    PROJECT_ROOT          — absolute host path to project root
                            (default: parent of this file's directory)
    PIPELINE_DEVICE       — cuda | cpu | auto  (default: cuda, forced here)
    PIPELINE_FPS          — frame extraction rate  (default: 2.0)
    WATCHER_POLL_S        — seconds between directory scans  (default: 5)
    LOG_LEVEL             — INFO | DEBUG  (default: INFO)
    LOG_FILE              — optional log file path

Path translation
----------------
The backend container mounts the host's data/ directory at /app/data inside
the container. Job files contain container-side paths (/app/data/...).
This watcher replaces the /app/data prefix with the host-side DATA_DIR
before passing paths to the orchestrator.

    Container: /app/data/raw/footage/20260523_120000_survey.mp4
    Host:      C:\\Facultate\\pothole-detection\\Pothole-Detection\\data\\raw\\footage\\20260523_120000_survey.mp4

Author: Paraschiv Tudor — Babeș-Bolyai University, 2026
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Absolute path to the project root on the Windows host.
# Default: one level up from this file (pipeline/job_watcher.py → project root)
_PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).parent.parent))
).resolve()

# data/ directory on the HOST (physical disk path)
_DATA_DIR_HOST = _PROJECT_ROOT / "data"

# The prefix the backend container uses for the same data/ directory.
# Matches PROJECT_DATA_DIR in .env and the docker-compose bind mount target.
_DATA_DIR_CONTAINER_PREFIX = os.getenv("PROJECT_DATA_DIR", "/app/data")

_JOBS_DIR     = _DATA_DIR_HOST / "jobs"
_SESSIONS_DIR = _DATA_DIR_HOST / "processed" / "sessions"

# Always force CUDA — this watcher only runs on the GPU host.
# The env var is read as a fallback, but we override to "cuda" by default.
_DEVICE = os.getenv("PIPELINE_DEVICE", "cuda")

# ===========================================================================
# DEBUG IMAGE TOGGLE (hardcoded)  --  search for SAVE_DEBUG to find this
# ---------------------------------------------------------------------------
# When True, every frontend-triggered run passes --save_debug to the
# orchestrator, so each run writes debug folders + images:
#     <session>/02_detections/debug/<frame>.jpg
#     <session>/03_segmentations/debug/<frame>.jpg
#     <session>/04_depth/debug/<frame>.jpg
# (only for frames that have detections; depth is skipped on low-light frames).
#
# TO DISABLE for frontend runs in the future: set this to False (or delete the
# "--save_debug" line marked SAVE_DEBUG in _run_job below). No other change
# needed. CLI runs of orchestrator.py are unaffected — they only get debug
# images when you pass --save_debug yourself.
# ===========================================================================
_SAVE_DEBUG = True

_FPS          = float(os.getenv("PIPELINE_FPS", "2.0"))
_POLL_S       = float(os.getenv("WATCHER_POLL_S", "5"))
_LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
_LOG_FILE     = os.getenv("LOG_FILE", "")

# Path to orchestrator.py — must be the host absolute path.
_ORCHESTRATOR = _PROJECT_ROOT / "pipeline" / "orchestrator.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    level    = getattr(logging, _LOG_LEVEL.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if _LOG_FILE:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s | %(levelname)-8s | job_watcher | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= handlers,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path translation: container → host
# ---------------------------------------------------------------------------

def _container_to_host(container_path: Optional[str]) -> Optional[Path]:
    """
    Replace the container data dir prefix with the host data dir prefix.

    Example:
        /app/data/raw/footage/abc.mp4
        → C:\\...\\data\\raw\\footage\\abc.mp4
    """
    if not container_path:
        return None
    # Normalise forward-slashes from the container
    norm = container_path.replace("\\", "/")
    prefix = _DATA_DIR_CONTAINER_PREFIX.replace("\\", "/").rstrip("/")
    if norm.startswith(prefix):
        rel = norm[len(prefix):].lstrip("/")
        return _DATA_DIR_HOST / rel
    # Already a host path or unknown — return as-is
    logger.warning(
        "Could not translate container path %r (prefix %r not found). "
        "Passing to orchestrator unchanged.",
        container_path, _DATA_DIR_CONTAINER_PREFIX,
    )
    return Path(container_path)

# ---------------------------------------------------------------------------
# Job file helpers
# ---------------------------------------------------------------------------

def _load_job(job_file: Path) -> Optional[dict]:
    try:
        with job_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Cannot read job file %s: %s", job_file, exc)
        return None


def _update_job_status(job_file: Path, status: str) -> None:
    """Atomically update the 'status' field of a job file."""
    data = _load_job(job_file)
    if data is None:
        return
    data["status"] = status
    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    try:
        tmp = job_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(job_file)   # atomic rename on both Windows and Linux
    except OSError as exc:
        logger.error("Cannot update job file %s: %s", job_file, exc)

# ---------------------------------------------------------------------------
# Orchestrator runner
# ---------------------------------------------------------------------------

def _run_job(job_file: Path, job: dict) -> None:
    """
    Translate paths from container-space to host-space, then call orchestrator.py
    via subprocess with --device cuda forced.

    Blocks until the orchestrator completes (or fails).
    The orchestrator writes session.json; we do not duplicate its output here.
    """
    job_id = job["job_id"]

    video_host = _container_to_host(job.get("video_path_container"))
    gps_host   = _container_to_host(job.get("gps_path_container"))
    fps        = float(job.get("fps", _FPS))

    if video_host is None or not video_host.exists():
        logger.error(
            "Job %s: video file not found on host: %s  (container path: %s)",
            job_id, video_host, job.get("video_path_container"),
        )
        _update_job_status(job_file, "failed")
        return

    if gps_host is not None and not gps_host.exists():
        logger.warning(
            "Job %s: GPS file not found on host: %s — proceeding without GPS.",
            job_id, gps_host,
        )
        gps_host = None

    # Build CLI command — always force CUDA here
    cmd = [
        sys.executable,
        str(_ORCHESTRATOR),
        "--video",      str(video_host),
        "--session_id", job_id,
        "--work_dir",   str(_SESSIONS_DIR),
        "--device",     "cuda",          # forced — this watcher only runs on the GPU host
        "--fps",        str(fps),
    ]
    if gps_host is not None:
        cmd += ["--gps", str(gps_host)]

    # --- SAVE_DEBUG (hardcoded) -------------------------------------------
    # Adds --save_debug so every frontend run writes the per-stage debug
    # folders/images. Controlled by the _SAVE_DEBUG constant at the top of
    # this file. Set _SAVE_DEBUG = False (or remove these two lines) to turn
    # debug image saving off for frontend-triggered runs.
    if _SAVE_DEBUG:
        cmd += ["--save_debug"]
    # ----------------------------------------------------------------------

    logger.info(
        "Starting orchestrator | job_id=%s | device=cuda | save_debug=%s | "
        "video=%s | gps=%s",
        job_id, _SAVE_DEBUG, video_host.name, gps_host.name if gps_host else "none",
    )
    logger.debug("Command: %s", " ".join(cmd))

    _update_job_status(job_file, "running")

    try:
        # stdout/stderr go to the console (and LOG_FILE via basicConfig).
        # The orchestrator also logs to its own LOG_FILE internally.
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            check=False,    # we check returncode manually below
        )
        if result.returncode == 0:
            logger.info("Orchestrator completed successfully | job_id=%s", job_id)
            _update_job_status(job_file, "complete")
        else:
            logger.error(
                "Orchestrator exited with returncode=%d | job_id=%s",
                result.returncode, job_id,
            )
            _update_job_status(job_file, "failed")
    except Exception as exc:
        logger.error("Failed to run orchestrator for job %s: %s", job_id, exc)
        _update_job_status(job_file, "failed")

# ---------------------------------------------------------------------------
# Watcher loop
# ---------------------------------------------------------------------------

def _scan_and_dispatch() -> None:
    """
    Scan _JOBS_DIR for job files with status == 'pending'.
    Process them in chronological order (oldest first).
    Only one job runs at a time.
    """
    if not _JOBS_DIR.exists():
        return

    pending = sorted(
        (f for f in _JOBS_DIR.glob("*.json")),
        key=lambda p: p.stem   # stems are timestamps: YYYYMMDD_HHMMSS
    )

    for job_file in pending:
        job = _load_job(job_file)
        if job is None:
            continue
        if job.get("status") != "pending":
            continue

        # Run synchronously — blocks until the orchestrator finishes.
        # This guarantees only one GPU job at a time.
        _run_job(job_file, job)
        # After one job completes (or fails) re-scan from the top so we
        # always pick up the oldest pending job next.
        break


def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("RIDS job_watcher starting")
    logger.info("  Project root : %s", _PROJECT_ROOT)
    logger.info("  Data dir     : %s", _DATA_DIR_HOST)
    logger.info("  Jobs dir     : %s", _JOBS_DIR)
    logger.info("  Orchestrator : %s", _ORCHESTRATOR)
    logger.info("  Device       : cuda (forced)")
    logger.info("  Save debug   : %s", _SAVE_DEBUG)
    logger.info("  Poll interval: %.1f s", _POLL_S)
    logger.info("=" * 60)

    if not _ORCHESTRATOR.exists():
        logger.error(
            "orchestrator.py not found at %s. "
            "Set PROJECT_ROOT in .env to the correct project root.",
            _ORCHESTRATOR,
        )
        sys.exit(1)

    _JOBS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            _scan_and_dispatch()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down job_watcher.")
            sys.exit(0)
        except Exception as exc:
            # Never crash the watcher loop; log and keep polling.
            logger.exception("Unexpected error in watcher loop: %s", exc)

        time.sleep(_POLL_S)


if __name__ == "__main__":
    main()