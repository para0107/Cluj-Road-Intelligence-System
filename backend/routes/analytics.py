"""
backend/routes/analytics.py

GET /analytics/ops — operations dashboard numbers for municipalities:
time-to-repair, open backlog by severity, reopened repairs, a 12-week
new-vs-repaired trend, and the work-order budget rollup.

Everything is computed straight from PostGIS aggregates and served through a
10 s micro-cache (same pattern as /live's read cache): the numbers only move
when surveys land or operators act.
"""

import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import Date, cast, func
from sqlalchemy.orm import Session

from backend.auth import require_operator
from backend.database import get_db
from backend.models import Detection
from backend.models_work import WorkOrder

router = APIRouter()

_CACHE_TTL_S = float(os.getenv("OPS_ANALYTICS_CACHE_S", "10.0"))
_cache: dict = {}
_cache_lock = threading.Lock()


class WeeklyPoint(BaseModel):
    week: str            # ISO date of the Monday
    new: int
    repaired: int


class BudgetRollup(BaseModel):
    estimated_open_ron: float = 0.0
    estimated_all_ron: float = 0.0
    actual_spent_ron: float = 0.0
    orders_by_status: Dict[str, int] = Field(default_factory=dict)


class OpsAnalyticsResponse(BaseModel):
    avg_days_to_repair: Optional[float] = None
    open_by_severity: Dict[int, int] = Field(default_factory=dict)
    open_total: int = 0
    fixed_total: int = 0
    reopened_count: int = 0
    weekly: List[WeeklyPoint] = Field(default_factory=list)
    budget: BudgetRollup = Field(default_factory=BudgetRollup)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _build(db: Session) -> OpsAnalyticsResponse:
    # Average days from first sighting to the fixed_at stamp.
    avg_days = (
        db.query(func.avg(
            cast(Detection.fixed_at, Date) - Detection.first_detected
        ))
        .filter(Detection.is_fixed.is_(True), Detection.fixed_at.isnot(None))
        .scalar()
    )
    avg_days_to_repair = round(float(avg_days), 1) if avg_days is not None else None

    open_rows = (
        db.query(Detection.severity, func.count(Detection.id))
        .filter(Detection.is_fixed.is_(False), Detection.severity.isnot(None))
        .group_by(Detection.severity)
        .all()
    )
    open_by_severity = {int(sev): int(n) for sev, n in open_rows}

    open_total = (
        db.query(func.count(Detection.id))
        .filter(Detection.is_fixed.is_(False)).scalar() or 0
    )
    fixed_total = (
        db.query(func.count(Detection.id))
        .filter(Detection.is_fixed.is_(True)).scalar() or 0
    )
    reopened_count = (
        db.query(func.count(Detection.id))
        .filter(
            Detection.is_fixed.is_(True),
            Detection.fixed_at.isnot(None),
            Detection.last_detected > cast(Detection.fixed_at, Date),
        )
        .scalar() or 0
    )

    # 12-week trend: new damage (first_detected) vs repairs (fixed_at).
    horizon = (_now() - timedelta(weeks=12)).date()
    new_rows = dict(
        db.query(
            func.date_trunc("week", cast(Detection.first_detected, Date)).label("wk"),
            func.count(Detection.id),
        )
        .filter(Detection.first_detected >= horizon)
        .group_by("wk")
        .all()
    )
    fixed_rows = dict(
        db.query(
            func.date_trunc("week", Detection.fixed_at).label("wk"),
            func.count(Detection.id),
        )
        .filter(Detection.fixed_at.isnot(None), cast(Detection.fixed_at, Date) >= horizon)
        .group_by("wk")
        .all()
    )

    def _week_key(dt) -> str:
        return dt.date().isoformat() if hasattr(dt, "date") else str(dt)

    weeks = sorted({_week_key(w) for w in list(new_rows) + list(fixed_rows)})
    by_key_new = {_week_key(w): n for w, n in new_rows.items()}
    by_key_fixed = {_week_key(w): n for w, n in fixed_rows.items()}
    weekly = [
        WeeklyPoint(week=w, new=int(by_key_new.get(w, 0)), repaired=int(by_key_fixed.get(w, 0)))
        for w in weeks
    ]

    # Budget rollup over work orders.
    status_rows = (
        db.query(WorkOrder.status, func.count(WorkOrder.id))
        .group_by(WorkOrder.status)
        .all()
    )
    est_all = db.query(func.coalesce(func.sum(WorkOrder.cost_estimate_ron), 0.0)).scalar() or 0.0
    est_open = (
        db.query(func.coalesce(func.sum(WorkOrder.cost_estimate_ron), 0.0))
        .filter(WorkOrder.status.in_(("open", "scheduled", "in_progress")))
        .scalar() or 0.0
    )
    spent = (
        db.query(func.coalesce(func.sum(WorkOrder.cost_actual_ron), 0.0))
        .filter(WorkOrder.status.in_(("repaired", "verified")))
        .scalar() or 0.0
    )

    return OpsAnalyticsResponse(
        avg_days_to_repair=avg_days_to_repair,
        open_by_severity=open_by_severity,
        open_total=int(open_total),
        fixed_total=int(fixed_total),
        reopened_count=int(reopened_count),
        weekly=weekly,
        budget=BudgetRollup(
            estimated_open_ron=round(float(est_open), 2),
            estimated_all_ron=round(float(est_all), 2),
            actual_spent_ron=round(float(spent), 2),
            orders_by_status={s: int(n) for s, n in status_rows},
        ),
    )


@router.get("/analytics/ops", response_model=OpsAnalyticsResponse)
def ops_analytics(db: Session = Depends(get_db), _op=Depends(require_operator)):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get("ops")
        if hit and hit[0] > now:
            return hit[1]
    value = _build(db)
    with _cache_lock:
        _cache["ops"] = (now + _CACHE_TTL_S, value)
    return value
