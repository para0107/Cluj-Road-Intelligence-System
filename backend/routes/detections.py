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
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID

from backend.database import get_db
from backend.models import Detection, SurveyLog
from backend.schemas import (
    DetectionRead,
    DetectionListResponse,
    DetectionStatusUpdate,
    DetectionDeleteRequest,
    DetectionDeleteResponse,
)

router = APIRouter()


@router.get("/detections", response_model=DetectionListResponse)
def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=10000),
    damage_type: Optional[str] = Query(None),
    severity_min: Optional[int] = Query(None, ge=1, le=5),
    severity_max: Optional[int] = Query(None, ge=1, le=5),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    sort_by: Optional[str] = Query("priority_score"),
    sort_order: Optional[str] = Query("desc"),
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

    # Handle sorting safely
    valid_sort_columns = {
        "damage_type": Detection.damage_type,
        "severity": Detection.severity,
        "confidence": Detection.confidence,
        "priority_score": Detection.priority_score,
        "detection_count": Detection.detection_count,
        "latitude": Detection.latitude,
        "last_detected": Detection.last_detected,
    }
    order_col = valid_sort_columns.get(sort_by, Detection.priority_score)
    if sort_order == "asc":
        query = query.order_by(order_col.asc())
    else:
        query = query.order_by(order_col.desc())

    items = (
        query
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
    # Cast both operands to geography so ST_DWithin measures the radius in
    # true metres (matching db_writer's upsert). Using raw geometry degrees
    # would over-estimate the east–west extent at Cluj's latitude.
    point = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)

    items = (
        db.query(Detection)
        .filter(
            ST_DWithin(
                Detection.geom.cast(Geography),
                point.cast(Geography),
                radius_m,
            )
        )
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

@router.patch("/detections/{detection_id}/status", response_model=DetectionRead)
def update_detection_status(
    detection_id: UUID,
    status_update: DetectionStatusUpdate,
    db: Session = Depends(get_db)
):
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found.")
    
    detection.is_fixed = status_update.is_fixed
    db.commit()
    db.refresh(detection)
    return detection


@router.delete("/detections/bulk", response_model=DetectionDeleteResponse)
def delete_detections_bulk(
    payload: DetectionDeleteRequest,
    db: Session = Depends(get_db),
):
    if not payload.ids:
        raise HTTPException(status_code=400, detail="At least one detection id is required.")

    rows = (
        db.query(Detection.id, Detection.survey_date)
        .filter(Detection.id.in_(payload.ids))
        .all()
    )

    existing_ids = {row.id for row in rows}
    deleted_survey_dates = sorted({row.survey_date for row in rows if row.survey_date})
    missing_ids = [detection_id for detection_id in payload.ids if detection_id not in existing_ids]

    deleted_detections = (
        db.query(Detection)
        .filter(Detection.id.in_(existing_ids))
        .delete(synchronize_session=False)
        if existing_ids
        else 0
    )

    deleted_survey_logs = 0
    if payload.delete_survey_log and deleted_survey_dates:
        deleted_survey_logs = (
            db.query(SurveyLog)
            .filter(SurveyLog.survey_date.in_(deleted_survey_dates))
            .delete(synchronize_session=False)
        )

    db.commit()

    return DetectionDeleteResponse(
        requested_count=len(payload.ids),
        deleted_detections=deleted_detections,
        deleted_survey_logs=deleted_survey_logs,
        deleted_survey_dates=deleted_survey_dates,
        missing_ids=missing_ids,
    )