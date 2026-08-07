"""
backend/routes/public_landing.py

GET /public/landing — a single unauthenticated payload for the public marketing
landing page. It serves three things the front door needs and nothing else:

  * headline city figures (how much damage is on record, how much is critical,
    how much has been repaired),
  * a short recent-activity feed (the latest de-duplicated detections, damage
    type and severity only),
  * a derived road-quality grade (A-E) for an at-a-glance badge.

Read-only aggregates only. No users, devices, coordinates, or evidence are
exposed. The route is rate-limited per IP and micro-cached, so a burst of
visitors costs one aggregate query, not one per request.
"""

import math
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import func, nullslast
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Detection
from backend.ratelimit import Limiter, client_ip

router = APIRouter()

# Public and unauthenticated: keep a generous but real per-IP budget.
_landing_limiter = Limiter(
    "public_landing", max_events=120, window_s=60.0,
    detail="Too many requests. Please slow down and try again shortly.",
)

_CACHE_TTL_S = float(os.getenv("LANDING_CACHE_S", "15.0"))
_cache: dict = {}
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


class LandingStats(BaseModel):
    total_detections: int
    open_count: int
    fixed_count: int
    critical_count: int
    avg_severity: Optional[float] = None
    last_survey_date: Optional[date] = None


class RecentItem(BaseModel):
    damage_type: str
    severity: Optional[int] = None
    detection_count: int = 1
    last_detected: Optional[date] = None


class QualityBadge(BaseModel):
    grade: str
    score: int


class LandingResponse(BaseModel):
    stats: LandingStats
    recent: List[RecentItem]
    quality: Optional[QualityBadge] = None


def _grade(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "E"


def _build(db: Session) -> LandingResponse:
    total = db.query(func.count(Detection.id)).scalar() or 0
    fixed = db.query(func.count(Detection.id)).filter(Detection.is_fixed.is_(True)).scalar() or 0
    critical = db.query(func.count(Detection.id)).filter(Detection.severity >= 4).scalar() or 0
    avg_sev = db.query(func.avg(Detection.severity)).scalar()
    avg_sev = round(float(avg_sev), 2) if avg_sev is not None else None
    last_survey = db.query(func.max(Detection.survey_date)).scalar()
    open_count = int(total - fixed)

    stats = LandingStats(
        total_detections=int(total),
        open_count=open_count,
        fixed_count=int(fixed),
        critical_count=int(critical),
        avg_severity=avg_sev,
        last_survey_date=last_survey,
    )

    recent_rows = (
        db.query(
            Detection.damage_type, Detection.severity,
            Detection.detection_count, Detection.last_detected,
        )
        .order_by(nullslast(Detection.last_detected.desc()), Detection.id.desc())
        .limit(8)
        .all()
    )
    recent = [
        RecentItem(
            damage_type=r[0], severity=r[1],
            detection_count=r[2] or 1, last_detected=r[3],
        )
        for r in recent_rows
    ]

    # A simple, auditable grade: heavier average severity and a larger share of
    # still-open damage push the score down. Repaired roads lift it back up.
    quality: Optional[QualityBadge] = None
    if total > 0:
        open_share = open_count / total if total else 0.0
        penalty = (avg_sev or 0.0) * (0.4 + 0.6 * open_share)
        score = int(round(100 * math.exp(-penalty / 3.2)))
        score = max(0, min(100, score))
        quality = QualityBadge(grade=_grade(score), score=score)

    return LandingResponse(stats=stats, recent=recent, quality=quality)


@router.get("/public/landing", response_model=LandingResponse)
def public_landing(request: Request, db: Session = Depends(get_db)):
    _landing_limiter.hit(f"landing|{client_ip(request)}")
    return _cached("landing", lambda: _build(db))
