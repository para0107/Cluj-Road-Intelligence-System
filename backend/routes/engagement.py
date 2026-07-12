"""
backend/routes/engagement.py

Citizen-facing gamification reads and the in-app notification feed.

  GET  /engagement/me            my stats, ranks, badges, points, reports
  GET  /engagement/leaderboard   top reporters (city-scoped or global)
  GET  /notifications            newest-first feed + unread count
  POST /notifications/read       mark some (or all) as read

Everything requires a signed-in user; nothing here mutates points (awards
happen inside the live-report flow, see backend/gamification.py).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.gamification import get_stats
from backend.models_auth import User
from backend.models_engagement import Notification, UserBadge, UserPoints, UserStats
from backend.models_live import LiveEvent, LiveReport
from backend.schemas_engagement import (
    BadgeRead, ImpactResponse, LeaderboardResponse, LeaderboardRow,
    MyReportRead, NotificationsMarkRequest, NotificationsMarkResponse,
    NotificationsResponse, PointsEntryRead, UserStatsRead,
)

router = APIRouter()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _rank(db: Session, stats: UserStats, city: str | None = None) -> int:
    q = db.query(func.count(UserStats.user_id)).filter(
        UserStats.points_total > (stats.points_total or 0)
    )
    if city:
        q = q.filter(UserStats.city == city)
    return (q.scalar() or 0) + 1


@router.get("/engagement/me", response_model=ImpactResponse)
def my_impact(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    stats = get_stats(db, user)

    badges = (
        db.query(UserBadge)
        .filter(UserBadge.user_id == user.id)
        .order_by(UserBadge.awarded_at.desc())
        .all()
    )
    recent_points = (
        db.query(UserPoints)
        .filter(UserPoints.user_id == user.id)
        .order_by(UserPoints.created_at.desc())
        .limit(20)
        .all()
    )

    # The 50 most recent events this account reported or confirmed.
    latest_signal = (
        db.query(
            LiveReport.event_id.label("event_id"),
            func.max(LiveReport.created_at).label("last_signal"),
        )
        .filter(
            LiveReport.user_id == user.id,
            LiveReport.kind.in_(("sighting", "confirm")),
        )
        .group_by(LiveReport.event_id)
        .order_by(func.max(LiveReport.created_at).desc())
        .limit(50)
        .subquery()
    )
    events = (
        db.query(LiveEvent)
        .join(latest_signal, LiveEvent.id == latest_signal.c.event_id)
        .order_by(latest_signal.c.last_signal.desc())
        .all()
    )
    my_reports = [
        MyReportRead(
            event_id=e.id,
            damage_type=e.damage_type,
            status=e.status or "unverified",
            is_active=bool(e.is_active),
            resolved=bool(e.resolved),
            promoted=e.promoted_detection_id is not None,
            latitude=e.latitude,
            longitude=e.longitude,
            last_reported=e.last_reported,
        )
        for e in events
    ]

    db.commit()   # persists the stats row if it was created on first visit
    return ImpactResponse(
        stats=UserStatsRead.model_validate(stats),
        rank_city=_rank(db, stats, city=user.city) if user.city else None,
        rank_global=_rank(db, stats),
        badges=[BadgeRead.model_validate(b) for b in badges],
        recent_points=[PointsEntryRead.model_validate(p) for p in recent_points],
        my_reports=my_reports,
    )


@router.get("/engagement/leaderboard", response_model=LeaderboardResponse)
def leaderboard(
    city: str | None = Query(None, max_length=80),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = (
        db.query(UserStats, User.username)
        .join(User, User.id == UserStats.user_id)
        .filter(UserStats.points_total > 0)
    )
    if city:
        q = q.filter(UserStats.city == city)
    rows = q.order_by(UserStats.points_total.desc()).limit(limit).all()

    return LeaderboardResponse(
        scope="city" if city else "global",
        city=city,
        items=[
            LeaderboardRow(
                rank=i + 1,
                username=username,
                points_total=stats.points_total or 0,
                confirmed_total=stats.confirmed_total or 0,
                city=stats.city,
            )
            for i, (stats, username) in enumerate(rows)
        ],
    )


@router.get("/notifications", response_model=NotificationsResponse)
def list_notifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    base = db.query(Notification).filter(Notification.user_id == user.id)
    total = base.count()
    unread = base.filter(Notification.read_at.is_(None)).count()
    items = (
        base.order_by(Notification.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return NotificationsResponse(unread=unread, total=total, items=items)


@router.post("/notifications/read", response_model=NotificationsMarkResponse)
def mark_notifications_read(
    payload: NotificationsMarkRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Notification).filter(
        Notification.user_id == user.id,
        Notification.read_at.is_(None),
    )
    if not payload.all:
        ids = payload.ids or []
        if not ids:
            return NotificationsMarkResponse(updated=0)
        q = q.filter(Notification.id.in_(ids))
    updated = q.update({Notification.read_at: _now()}, synchronize_session=False)
    db.commit()
    return NotificationsMarkResponse(updated=updated)
