"""
backend/routes/export.py

GET /export/csv — export detections as a CSV file
"""

import os
import sys
import csv
from io import StringIO
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.database import get_db
from backend.auth import require_operator
from backend.models import Detection

router = APIRouter()

_HEADERS = [
    "id", "damage_type", "confidence", "latitude", "longitude",
    "severity", "depth_estimate_cm", "surface_area_cm2",
    "detection_count", "first_detected", "last_detected", "priority_score", "is_fixed"
]

_BATCH_ROWS = 500


def _iter_csv(db: Session):
    """Stream the table in batches so a large export never sits whole in RAM.

    The request-scoped session stays open until the response finishes
    (FastAPI runs dependency teardown after the body is sent).
    """
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(_HEADERS)

    pending = 0
    for det in db.query(Detection).yield_per(1000):
        writer.writerow([
            det.id,
            det.damage_type,
            round(det.confidence, 4) if det.confidence is not None else "",
            round(det.latitude, 6) if det.latitude is not None else "",
            round(det.longitude, 6) if det.longitude is not None else "",
            det.severity,
            round(det.depth_estimate_cm, 2) if det.depth_estimate_cm is not None else "",
            round(det.surface_area_cm2, 2) if det.surface_area_cm2 is not None else "",
            det.detection_count,
            det.first_detected,
            det.last_detected,
            round(det.priority_score, 4) if det.priority_score is not None else "",
            det.is_fixed
        ])
        pending += 1
        if pending >= _BATCH_ROWS:
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
            pending = 0
    if buf.tell() or pending:
        yield buf.getvalue()


@router.get("/export/csv")
def export_csv(db: Session = Depends(get_db), _op=Depends(require_operator)):
    """
    Export all detections as a CSV file (streamed in batches).
    """
    response = StreamingResponse(_iter_csv(db), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=rids_detections_export.csv"
    return response
