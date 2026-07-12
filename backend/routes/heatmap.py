"""
backend/routes/heatmap.py

GET /heatmap — returns lat/lon/weight points for Leaflet.heat overlay
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import APIRouter, Depends
from sqlalchemy import Numeric, cast, desc, func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import require_operator
from backend.models import Detection
from backend.schemas import HeatmapResponse, HeatmapPoint

router = APIRouter()

# One heat point per ~11 m bin (4 decimal places) instead of one per row keeps
# the payload bounded as the survey history grows; the blur radius on the map
# is far coarser than 11 m, so the rendered overlay is identical.
_MAX_POINTS = int(os.getenv("HEATMAP_MAX_POINTS", "20000"))


@router.get("/heatmap", response_model=HeatmapResponse)
def get_heatmap(db: Session = Depends(get_db), _op=Depends(require_operator)):
    lat_bin = func.round(cast(Detection.latitude, Numeric), 4)
    lon_bin = func.round(cast(Detection.longitude, Numeric), 4)
    weight = func.sum(
        Detection.severity * func.ln(Detection.detection_count + 1)
    ).label("weight")

    rows = (
        db.query(
            func.avg(Detection.latitude).label("lat"),
            func.avg(Detection.longitude).label("lon"),
            weight,
        )
        .filter(Detection.severity.isnot(None))
        .group_by(lat_bin, lon_bin)
        .order_by(desc("weight"))
        .limit(_MAX_POINTS)
        .all()
    )

    points = [
        HeatmapPoint(latitude=lat, longitude=lon, weight=round(float(w or 0.0), 3))
        for lat, lon, w in rows
    ]
    return HeatmapResponse(points=points)