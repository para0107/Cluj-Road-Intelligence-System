"""
backend/routes/priority.py

GET /priority-list — top N detections ranked by priority_score
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Detection
from backend.schemas import PriorityListResponse, PriorityItem

router = APIRouter()

# backend/routes/priority.py

@router.get("/priority-list", response_model=PriorityListResponse)
def get_priority_list(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Detection)
        .filter(Detection.severity.isnot(None))
        .order_by(Detection.priority_score.desc())
        .limit(limit)
        .all()
    )

    items = [
        PriorityItem(
            id=row.id,
            rank=idx + 1,
            priority_score=row.priority_score or 0.0,
            severity=row.severity,
            damage_type=row.damage_type,
            latitude=row.latitude,
            longitude=row.longitude,
            detection_count=row.detection_count,
            last_detected=row.last_detected,
        )
        for idx, row in enumerate(rows)
    ]

    return PriorityListResponse(items=items)