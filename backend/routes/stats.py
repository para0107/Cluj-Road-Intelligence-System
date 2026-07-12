"""
backend/routes/stats.py

GET /stats — city-wide summary statistics
"""

import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import Detection
from backend.schemas import StatsResponse, DamageTypeCount, SeverityCount

router = APIRouter()

# Micro-cache: /stats is on the Command page for every signed-in user, and the
# aggregates only change when a survey lands or an operator edits. Ten seconds
# of staleness is invisible; the DB saves a full-table aggregate per request.
_CACHE_TTL_S = float(os.getenv("STATS_CACHE_S", "10.0"))
_cache: dict = {}          # key -> (expires_monotonic, value)
_cache_lock = threading.Lock()


def _cached(key: str, build):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = build()
    with _cache_lock:
        _cache[key] = (now + _CACHE_TTL_S, value)
    return value


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    # Any signed-in role may read the headline stats (the Command page shows
    # them to citizens too); the detailed survey pages are operator-only.
    return _cached("stats", lambda: _build_stats(db))


def _build_stats(db: Session) -> StatsResponse:
    total = db.query(func.count(Detection.id)).scalar() or 0
    last_survey = db.query(func.max(Detection.survey_date)).scalar()

    detections_today = 0
    if last_survey:
        detections_today = (
            db.query(func.count(Detection.id))
            .filter(Detection.survey_date == last_survey)
            .scalar()
            or 0
        )

    type_rows = (
        db.query(Detection.damage_type, func.count(Detection.id))
        .group_by(Detection.damage_type)
        .all()
    )
    damage_type_breakdown = [
        DamageTypeCount(damage_type=row[0], count=row[1]) for row in type_rows
    ]

    sev_rows = (
        db.query(Detection.severity, func.count(Detection.id))
        .filter(Detection.severity.isnot(None))
        .group_by(Detection.severity)
        .order_by(Detection.severity)
        .all()
    )
    severity_breakdown = [
        SeverityCount(severity=row[0], count=row[1]) for row in sev_rows
    ]

    avg_severity = db.query(func.avg(Detection.severity)).scalar()
    avg_severity = round(float(avg_severity), 2) if avg_severity is not None else None

    avg_confidence = db.query(func.avg(Detection.confidence)).scalar()
    avg_confidence = round(float(avg_confidence), 3) if avg_confidence is not None else None

    critical_count = (
        db.query(func.count(Detection.id))
        .filter(Detection.severity >= 4)
        .scalar()
        or 0
    )

    fixed_count = (
        db.query(func.count(Detection.id))
        .filter(Detection.is_fixed == True)
        .scalar()
        or 0
    )

    return StatsResponse(
        total_detections=total,
        last_survey_date=last_survey,
        detections_today=detections_today,
        damage_type_breakdown=damage_type_breakdown,
        severity_breakdown=severity_breakdown,
        avg_severity=avg_severity,
        avg_confidence=avg_confidence,
        critical_count=critical_count,
        fixed_count=fixed_count,
    )