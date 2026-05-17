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
from backend.models import Detection

router = APIRouter()

@router.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    """
    Export all detections as a CSV file.
    """
    detections = db.query(Detection).all()
    
    # Create an in-memory string buffer
    f = StringIO()
    writer = csv.writer(f)
    
    # Write the header
    headers = [
        "id", "damage_type", "confidence", "latitude", "longitude",
        "severity", "depth_estimate_cm", "surface_area_cm2",
        "detection_count", "first_detected", "last_detected", "priority_score", "is_fixed"
    ]
    writer.writerow(headers)
    
    # Write the rows
    for det in detections:
        writer.writerow([
            det.id,
            det.damage_type,
            round(det.confidence, 4) if det.confidence else "",
            round(det.latitude, 6) if det.latitude else "",
            round(det.longitude, 6) if det.longitude else "",
            det.severity,
            round(det.depth_estimate_cm, 2) if det.depth_estimate_cm else "",
            round(det.surface_area_cm2, 2) if det.surface_area_cm2 else "",
            det.detection_count,
            det.first_detected,
            det.last_detected,
            round(det.priority_score, 4) if det.priority_score else "",
            det.is_fixed
        ])
    
    # Return as a streaming response with appropriate headers
    f.seek(0)
    response = StreamingResponse(iter([f.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=rids_detections_export.csv"
    
    return response
