"""
backend/routes/detections.py

GET /detections         — paginated list with filters
GET /detections/{id}    — single detection by UUID
GET /detections/nearby  — detections within radius of a point
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from uuid import UUID
from typing import Optional
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID

from backend.database import get_db
from backend.models import Detection
from backend.schemas import DetectionRead, DetectionListResponse

router = APIRouter()


@router.get("/detections", response_model=DetectionListResponse)
def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    damage_type: Optional[str] = Query(None),
    severity_min: Optional[int] = Query(None, ge=1, le=5),
    severity_max: Optional[int] = Query(None, ge=1, le=5),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(Detection)

    if damage_type:
        query = query.filter(Detection.damage_type == damage_type)
    if severity_min is not None:
        query = query.filter(Detection.severity >= severity_min)
    if severity_max is not None:
        query = query.filter(Detection.severity <= severity_max)
    if date_from:
        query = query.filter(Detection.last_detected >= date_from)
    if date_to:
        query = query.filter(Detection.last_detected <= date_to)

    total = query.count()
    items = (
        query.order_by(Detection.priority_score.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return DetectionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get("/detections/nearby", response_model=DetectionListResponse)
def detections_nearby(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(100, ge=1, le=5000),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    radius_deg = radius_m / 111000.0
    point = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)

    items = (
        db.query(Detection)
        .filter(ST_DWithin(Detection.geom, point, radius_deg))
        .order_by(Detection.priority_score.desc())
        .limit(limit)
        .all()
    )

    return DetectionListResponse(
        total=len(items),
        page=1,
        page_size=limit,
        items=items,
    )


@router.get("/detections/{detection_id}", response_model=DetectionRead)
def get_detection(detection_id: UUID, db: Session = Depends(get_db)):
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found.")
    return detection