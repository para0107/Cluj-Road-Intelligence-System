"""
backend/routes/quality.py

Road Quality Index (RQI): a sellable, street-name-free data product.

The map area is divided into ~cell_m × cell_m grid cells (plain lat/lon math,
no geocoding — the enrichment ban stands). Each cell aggregates its
detections into a penalty and the penalty maps to a 0–100 score:

    sev_w      severity 1..5 → 1, 2, 4, 7, 11        (non-linear: an S5 hurts
                                                       far more than five S1s)
    recency_w  exp(-days_since_last_detected / 180)  (old damage fades)
    fixed_w    0.2 when repaired, else 1.0           (fixed ≈ mostly healed)

    penalty = Σ sev_w · recency_w · fixed_w
    score   = round(100 · exp(-penalty / 8))         → bands A ≥85, B ≥70,
                                                       C ≥50, D ≥30, else E

Endpoints:
    GET /quality/grid    (any signed-in user)  cells for a bbox
    GET /quality/export  (operator)            CSV or GeoJSON download
"""

import csv
import io
import json
import math
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from backend.auth import get_current_user, require_operator
from backend.database import get_db
from backend.models import Detection

router = APIRouter()

# Tuning knobs, all in one place.
SEVERITY_WEIGHT = {1: 1.0, 2: 2.0, 3: 4.0, 4: 7.0, 5: 11.0}
RECENCY_HALF_DAYS = 180.0
FIXED_WEIGHT = 0.2
PENALTY_SCALE = 8.0
BANDS = (("A", 85), ("B", 70), ("C", 50), ("D", 30))   # else E

_MAX_CELLS = int(os.getenv("RQI_MAX_CELLS", "5000"))
_CACHE_TTL_S = float(os.getenv("RQI_CACHE_S", "30.0"))
_cache: dict = {}
_cache_lock = threading.Lock()


class QualityCell(BaseModel):
    lat: float                 # cell centroid (mean of member detections)
    lon: float
    score: int
    band: str
    n: int                     # detections aggregated in this cell
    worst: Optional[int] = None   # worst severity present


class QualityGridResponse(BaseModel):
    cell_m: int
    generated_at: datetime
    count: int
    cells: List[QualityCell] = Field(default_factory=list)


def _band(score: int) -> str:
    for name, floor in BANDS:
        if score >= floor:
            return name
    return "E"


def _parse_bbox(bbox: str | None):
    """bbox = 'minLon,minLat,maxLon,maxLat' (the GeoJSON convention)."""
    if not bbox:
        return None
    try:
        parts = [float(p) for p in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        min_lon, min_lat, max_lon, max_lat = parts
    except ValueError:
        raise HTTPException(400, "bbox must be 'minLon,minLat,maxLon,maxLat'.")
    if not (min_lon < max_lon and min_lat < max_lat):
        raise HTTPException(400, "bbox is empty or inverted.")
    if not (-180 <= min_lon <= 180 and -90 <= min_lat <= 90
            and -180 <= max_lon <= 180 and -90 <= max_lat <= 90):
        raise HTTPException(400, "bbox is out of range.")
    return (min_lon, min_lat, max_lon, max_lat)


def compute_grid(db: Session, bbox: str | None, cell_m: int) -> List[QualityCell]:
    """Shared by /quality/grid, /quality/export and the public API."""
    box = _parse_bbox(bbox)

    q = db.query(Detection).filter(Detection.severity.isnot(None))
    if box:
        min_lon, min_lat, max_lon, max_lat = box
        q = q.filter(
            Detection.longitude >= min_lon, Detection.longitude <= max_lon,
            Detection.latitude >= min_lat, Detection.latitude <= max_lat,
        )
        lat_ref = (min_lat + max_lat) / 2.0
    else:
        lat_ref = db.query(func.avg(Detection.latitude)).scalar() or 45.0

    # Cell size in degrees at the reference latitude.
    dlat = cell_m / 111_320.0
    dlon = cell_m / (111_320.0 * max(math.cos(math.radians(lat_ref)), 0.1))

    sev_w = case(
        *[(Detection.severity == s, w) for s, w in SEVERITY_WEIGHT.items()],
        else_=1.0,
    )
    # current_date - date → integer days in PostgreSQL; clamp future dates to 0.
    days_since = func.greatest(func.current_date() - Detection.last_detected, 0)
    recency_w = func.exp(-days_since / RECENCY_HALF_DAYS)
    fixed_w = case((Detection.is_fixed.is_(True), FIXED_WEIGHT), else_=1.0)
    penalty = func.sum(sev_w * recency_w * fixed_w).label("penalty")

    lat_bin = func.floor(Detection.latitude / dlat)
    lon_bin = func.floor(Detection.longitude / dlon)

    rows = (
        q.with_entities(
            func.avg(Detection.latitude).label("lat"),
            func.avg(Detection.longitude).label("lon"),
            penalty,
            func.count(Detection.id).label("n"),
            func.max(Detection.severity).label("worst"),
        )
        .group_by(lat_bin, lon_bin)
        .all()
    )

    if len(rows) > _MAX_CELLS:
        raise HTTPException(
            413,
            f"The requested area has {len(rows)} cells (limit {_MAX_CELLS}). "
            "Zoom in or pass a smaller bbox.",
        )

    cells = []
    for lat, lon, pen, n, worst in rows:
        score = int(round(100.0 * math.exp(-float(pen or 0.0) / PENALTY_SCALE)))
        score = max(0, min(100, score))
        cells.append(QualityCell(
            lat=round(float(lat), 6), lon=round(float(lon), 6),
            score=score, band=_band(score), n=int(n),
            worst=int(worst) if worst is not None else None,
        ))
    cells.sort(key=lambda c: c.score)
    return cells


@router.get("/quality/grid", response_model=QualityGridResponse)
def quality_grid(
    bbox: str | None = Query(None, max_length=120),
    cell_m: int = Query(120, ge=40, le=1000),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    key = f"{bbox}|{cell_m}"
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]

    cells = compute_grid(db, bbox, cell_m)
    resp = QualityGridResponse(
        cell_m=cell_m,
        generated_at=datetime.now(tz=timezone.utc),
        count=len(cells),
        cells=cells,
    )
    with _cache_lock:
        if len(_cache) > 200:
            _cache.clear()
        _cache[key] = (now + _CACHE_TTL_S, resp)
    return resp


@router.get("/quality/export")
def quality_export(
    format: str = Query("csv", pattern="^(csv|geojson)$"),
    bbox: str | None = Query(None, max_length=120),
    cell_m: int = Query(120, ge=40, le=1000),
    db: Session = Depends(get_db),
    _op=Depends(require_operator),
):
    cells = compute_grid(db, bbox, cell_m)

    if format == "csv":
        def _iter():
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["latitude", "longitude", "score", "band", "detections", "worst_severity"])
            for c in cells:
                writer.writerow([c.lat, c.lon, c.score, c.band, c.n, c.worst if c.worst is not None else ""])
                if buf.tell() > 64_000:
                    yield buf.getvalue()
                    buf.seek(0)
                    buf.truncate(0)
            yield buf.getvalue()
        resp = StreamingResponse(_iter(), media_type="text/csv")
        resp.headers["Content-Disposition"] = "attachment; filename=rdds_quality_index.csv"
        return resp

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
            "properties": {
                "score": c.score, "band": c.band,
                "detections": c.n, "worst_severity": c.worst,
                "cell_m": cell_m,
            },
        }
        for c in cells
    ]
    body = json.dumps({"type": "FeatureCollection", "features": features})
    resp = StreamingResponse(iter([body]), media_type="application/geo+json")
    resp.headers["Content-Disposition"] = "attachment; filename=rdds_quality_index.geojson"
    return resp
