"""
backend/routes/media.py

Serve per-detection evidence photos written by the pipeline
(pipeline/db_writer.py stores a data-dir-relative path in detections.crop_path,
e.g. "processed/sessions/20260711_120000/07_db_write/evidence/ab12….jpg").

Path safety mirrors the ingest route's stance: the stored value is rejected
unless it resolves strictly inside PROJECT_DATA_DIR and ends in .jpg — a
tampered DB value can never read outside the data tree.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.auth import require_operator
from backend.database import get_db
from backend.models import Detection

router = APIRouter()

_DATA_DIR = Path(os.getenv("PROJECT_DATA_DIR", "/app/data"))


@router.get("/media/evidence/{detection_id}")
def get_evidence(
    detection_id: UUID,
    db: Session = Depends(get_db),
    _op=Depends(require_operator),
):
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found.")
    rel = detection.crop_path
    if not rel:
        raise HTTPException(status_code=404, detail="No evidence photo for this detection.")

    # Reject absolute paths, parent-escapes, and anything that is not a JPG.
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or rel_path.suffix.lower() != ".jpg":
        raise HTTPException(status_code=404, detail="No evidence photo for this detection.")

    data_root = _DATA_DIR.resolve()
    full = (data_root / rel_path).resolve()
    inside = full == data_root or data_root in full.parents
    if not inside or not full.is_file():
        raise HTTPException(status_code=404, detail="No evidence photo for this detection.")

    return FileResponse(
        full,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )
