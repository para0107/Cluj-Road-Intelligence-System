"""
backend/routes/ingest.py

POST /ingest/upload      — accept video (.mp4) + optional gps (.gpx) multipart
                           files, persist them inside the shared data/ volume,
                           write a job-request JSON to data/jobs/<job_id>.json.
                           Returns job_id immediately (HTTP 202).

GET  /ingest/status/{job_id} — read data/processed/sessions/<job_id>/session.json
                               written by the orchestrator on the host.
                               Returns live progress for the frontend poller.

──────────────────────────────────────────────────────────────────────────────
WHY NOT subprocess.Popen?
──────────────────────────────────────────────────────────────────────────────
The FastAPI backend runs inside a Linux Docker container (python:3.11-slim).
The orchestrator requires:
  • CUDA GPU access (RTX 2050 on the Windows host)
  • PyTorch, Ultralytics, SAM 2.1, Monodepth2 — heavy ML deps not present in
    the backend container image
  • Windows absolute paths (MONODEPTH_ROOT, MONODEPTH_WEIGHTS_DIR in .env)
  • Access to ml/weights/*.pt model checkpoints on the host filesystem

A subprocess from inside the container cannot reach the host GPU, the host
filesystem at C:\..., or execute the correct Python interpreter with those deps.

──────────────────────────────────────────────────────────────────────────────
JOB-FILE HANDOFF PATTERN
──────────────────────────────────────────────────────────────────────────────
The data/ directory is bind-mounted into the backend container as /app/data.
This shared directory is the communication channel:

  1. Backend container writes:
         /app/data/jobs/<job_id>.json        (job request)
         /app/data/raw/footage/<job_id>_*.mp4
         /app/data/raw/gps_logs/<job_id>_*.gpx  (optional)

  2. pipeline/job_watcher.py runs on the Windows HOST, watching
         C:\\...\\data\\jobs\\          for new *.json files.
     When it finds one it spawns orchestrator.py with --device cuda.

  3. Orchestrator writes:
         C:\\...\\data\\processed\\sessions\\<job_id>\\session.json
     after every stage (and on completion/failure).

  4. Backend container reads:
         /app/data/processed/sessions/<job_id>/session.json
     on every GET /ingest/status/{job_id} call.

The concurrency guard lives in the job file itself (status field), so it works
correctly even if the backend container restarts while a job is running.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths — all relative to /app/data inside the container.
# The entire data/ tree is bind-mounted from the Windows host, so every file
# written here is immediately visible to job_watcher.py on the host.
# ---------------------------------------------------------------------------

# /app/data inside the container  (set PROJECT_DATA_DIR in .env to override)
_DATA_DIR     = Path(os.getenv("PROJECT_DATA_DIR", "/app/data"))
_RAW_FOOTAGE  = _DATA_DIR / "raw" / "footage"
_RAW_GPS      = _DATA_DIR / "raw" / "gps_logs"
_SESSIONS_DIR = _DATA_DIR / "processed" / "sessions"
_JOBS_DIR     = _DATA_DIR / "jobs"          # job request drop folder

_FPS          = float(os.getenv("PIPELINE_FPS", "2.0"))


def _ensure_dirs() -> None:
    for d in (_RAW_FOOTAGE, _RAW_GPS, _SESSIONS_DIR, _JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _generate_job_id() -> str:
    """Timestamp-based session ID matching the orchestrator's own format."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_job_file(job_id: str) -> Optional[dict]:
    """Return the parsed job request JSON, or None if it does not exist."""
    path = _JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read job file for %s: %s", job_id, exc)
        return None


def _read_session_json(job_id: str) -> Optional[dict]:
    """
    Return the parsed session.json written by the orchestrator, or None.
    The orchestrator writes this file after every stage; reads are always
    of a complete JSON document (no partial writes).
    """
    path = _SESSIONS_DIR / job_id / "session.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read session.json for %s: %s", job_id, exc)
        return None


def _active_job_id() -> Optional[str]:
    """
    Scan the jobs directory for a job whose status is 'pending' or 'running'.
    Returns the first such job_id, or None if no active job exists.

    This replaces the in-process _RUNNING_JOBS set from the subprocess version.
    It is correct across container restarts because the state lives on disk.
    """
    if not _JOBS_DIR.exists():
        return None
    for job_file in sorted(_JOBS_DIR.glob("*.json")):
        try:
            with job_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") in ("pending", "running"):
                return data.get("job_id")
        except (json.JSONDecodeError, OSError):
            continue
    return None


# ---------------------------------------------------------------------------
# POST /ingest/upload
# ---------------------------------------------------------------------------

@router.post(
    "/ingest/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload dashcam footage and optional GPS log; queue a pipeline job.",
    tags=["Ingest"],
)
async def upload_survey(
    video: UploadFile = File(..., description="Dashcam footage (.mp4)"),
    gps:   UploadFile = File(None, description="GPS log (.gpx) — optional"),
):
    """
    Persists uploaded files to the shared data/ volume and writes a job-request
    JSON to data/jobs/<job_id>.json.  The pipeline/job_watcher.py process
    running on the Windows host picks this file up and executes the orchestrator
    with --device cuda.

    Returns HTTP 202 immediately with the job_id for polling.

    Raises 400 for wrong file types.
    Raises 409 if a job is already pending/running.
    Raises 500 if files cannot be saved.
    """
    # ── Validate file types ───────────────────────────────────────────────
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only .mp4 video files are accepted. Got: {video.filename!r}",
        )
    if gps is not None and not gps.filename.lower().endswith(".gpx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only .gpx GPS files are accepted. Got: {gps.filename!r}",
        )

    # ── Concurrency guard (disk-based, survives container restarts) ───────
    running = _active_job_id()
    if running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Pipeline job {running!r} is already pending or running. "
                "Wait for it to complete before submitting a new survey."
            ),
        )

    _ensure_dirs()
    job_id = _generate_job_id()

    # ── Save video file ───────────────────────────────────────────────────
    # Prefix with job_id to guarantee uniqueness across concurrent uploads.
    safe_video_name = f"{job_id}_{Path(video.filename).name}"
    video_path = _RAW_FOOTAGE / safe_video_name
    try:
        video_bytes = await video.read()
        video_path.write_bytes(video_bytes)
        logger.info("Saved video → %s (%d bytes)", video_path, len(video_bytes))
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {exc}",
        )

    # ── Save GPS file (optional) ──────────────────────────────────────────
    gps_container_path: Optional[str] = None
    if gps is not None:
        safe_gps_name = f"{job_id}_{Path(gps.filename).name}"
        gps_path = _RAW_GPS / safe_gps_name
        try:
            gps_bytes = await gps.read()
            gps_path.write_bytes(gps_bytes)
            gps_container_path = str(gps_path)
            logger.info("Saved GPS  → %s (%d bytes)", gps_path, len(gps_bytes))
        except OSError as exc:
            video_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save GPS file: {exc}",
            )

    # ── Write job-request file ────────────────────────────────────────────
    # job_watcher.py on the host translates container paths → host paths
    # before calling the orchestrator. The container_prefix / host_prefix
    # fields make this translation explicit and robust.
    job_payload = {
        "job_id":           job_id,
        "status":           "pending",       # watcher sets → "running" when it starts
        "created_at":       datetime.now(tz=timezone.utc).isoformat(),
        # Paths as seen inside the container (/app/data/...)
        "video_path_container": str(video_path),
        "gps_path_container":   gps_container_path,
        # fps forwarded to orchestrator --fps flag
        "fps":              _FPS,
    }

    job_file = _JOBS_DIR / f"{job_id}.json"
    try:
        with job_file.open("w", encoding="utf-8") as f:
            json.dump(job_payload, f, indent=2)
        logger.info("Job file written → %s", job_file)
    except OSError as exc:
        video_path.unlink(missing_ok=True)
        if gps_container_path:
            Path(gps_container_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write job file: {exc}",
        )

    logger.info(
        "Job queued | job_id=%s | video=%s | gps=%s",
        job_id, safe_video_name, Path(gps_container_path).name if gps_container_path else None,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id":     job_id,
            "status":     "pending",
            "video_file": safe_video_name,
            "gps_file":   Path(gps_container_path).name if gps_container_path else None,
            "message":    (
                "Job queued. pipeline/job_watcher.py on the host will pick it up. "
                "Poll /api/ingest/status/{job_id} for live updates."
            ),
        },
    )


# ---------------------------------------------------------------------------
# GET /ingest/status/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/ingest/status/{job_id}",
    summary="Poll the status of a queued, running, or completed pipeline job.",
    tags=["Ingest"],
)
def get_job_status(job_id: str):
    """
    Status resolution order:
      1. session.json exists and status == complete|failed  → return it
      2. session.json exists and status == running          → return it
      3. session.json does not exist, job file status == pending|running
             → return "initialising" (watcher received it, orchestrator starting)
      4. job file does not exist  → 404

    Status values the frontend understands:
        pending       — job file written, watcher has not picked it up yet
        initialising  — watcher started orchestrator, Stage 1 not done yet
        running       — orchestrator in progress (session.json exists)
        complete      — all stages succeeded
        failed        — a stage raised an exception
    """
    # Check session.json first — most authoritative source of truth once the
    # orchestrator has started writing it.
    session_data = _read_session_json(job_id)
    if session_data is not None:
        return {
            "job_id":        job_id,
            "status":        session_data.get("status", "unknown"),
            "stages":        session_data.get("stages", []),
            "n_frames":      session_data.get("n_frames", 0),
            "n_detections":  session_data.get("n_detections", 0),
            "n_inserted":    session_data.get("n_inserted", 0),
            "n_updated":     session_data.get("n_updated", 0),
            "error_message": session_data.get("error_message"),
            "started_at":    session_data.get("started_at"),
            "finished_at":   session_data.get("finished_at"),
            "video_path":    session_data.get("video_path"),
            "gps_path":      session_data.get("gps_path"),
        }

    # session.json not yet written — check job file
    job_data = _read_job_file(job_id)
    if job_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline job found for job_id={job_id!r}.",
        )

    job_status = job_data.get("status", "pending")
    return {
        "job_id":        job_id,
        "status":        "initialising" if job_status == "running" else job_status,
        "stages":        [],
        "n_frames":      0,
        "n_detections":  0,
        "n_inserted":    0,
        "n_updated":     0,
        "error_message": None,
        "started_at":    None,
        "finished_at":   None,
        "video_path":    job_data.get("video_path_container"),
        "gps_path":      job_data.get("gps_path_container"),
    }
