"""
backend/routes/heatmap.py

GET /heatmap — returns lat/lon/weight points for Leaflet.heat overlay
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import require_operator
from backend.models import Detection
from backend.schemas import HeatmapResponse, HeatmapPoint

router = APIRouter()


@router.get("/heatmap", response_model=HeatmapResponse)
def get_heatmap(db: Session = Depends(get_db), _op=Depends(require_operator)):
    detections = (
        db.query(
            Detection.latitude,
            Detection.longitude,
            Detection.severity,
            Detection.detection_count,
        )
        .filter(Detection.severity.isnot(None))
        .all()
    )

    points = []
    for lat, lon, severity, count in detections:
        # Weight = severity (1-5) amplified by how many times it's been seen
        import math
        weight = severity * math.log(count + 1)
        points.append(HeatmapPoint(latitude=lat, longitude=lon, weight=round(weight, 3)))

    return HeatmapResponse(points=points)