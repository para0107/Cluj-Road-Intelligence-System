"""
backend/gamification.py

Points, badges, streaks, and in-app notifications for citizen reporters.

Anti-spam economy (the rules that make points farm-resistant):
  * A raw report earns NO points — only community-validated outcomes do:
        event reaches confirmed  → +10 to every distinct supporting account
        event reaches verified   → +15 more
        operator resolves (fixed)→ +25 more
        operator promotes        → +20 more
    A single account cannot reach any of these alone (distinct-DEVICE
    escalation in routes/live.py) and server-side report cooldowns apply.
  * Every award is idempotent: the ledger's unique (user_id, reason, ref_id)
    constraint swallows replays via a savepoint, so hooks can fire twice
    without double-crediting.

All functions only add/modify objects on the caller's session and flush —
they NEVER commit. The calling route's commit persists everything atomically
with the report/vote that triggered it.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import distinct, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models_auth import User
from backend.models_engagement import Notification, UserBadge, UserPoints, UserStats
from backend.models_live import LiveReport

# Reason → points. Reasons double as ledger idempotency keys.
POINTS = {
    "event_confirmed": 10,
    "event_verified": 15,
    "event_fixed": 25,
    "event_promoted": 20,
}

# badge_key → (label, description, check(stats) -> bool). Labels/descriptions
# are mirrored in frontend/src/utils/constants.js (BADGES) for display.
BADGES = {
    "first_report":   ("First report",   "Sent a first hazard report.",
                       lambda s: (s.reports_total or 0) >= 1),
    "confirmed_10":   ("Road scout",     "Ten of your reports were confirmed.",
                       lambda s: (s.confirmed_total or 0) >= 10),
    "confirmed_50":   ("Road guardian",  "Fifty of your reports were confirmed.",
                       lambda s: (s.confirmed_total or 0) >= 50),
    "verified_first": ("Triple checked", "A report of yours reached verified.",
                       lambda s: (s.verified_total or 0) >= 1),
    "streak_7":       ("Week streak",    "Reported on seven days in a row.",
                       lambda s: (s.current_streak_days or 0) >= 7),
    "fixed_1":        ("Fixer",          "A hazard you reported was repaired.",
                       lambda s: (s.fixed_total or 0) >= 1),
    "fixed_5":        ("City changer",   "Five hazards you reported were repaired.",
                       lambda s: (s.fixed_total or 0) >= 5),
    "night_reporter": ("Night watch",    "Reported a hazard late at night.",
                       lambda s: False),   # awarded directly in record_report
}

_NOTIFICATIONS_KEPT_PER_USER = 100
_LOCAL_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Europe/Bucharest"))

# Which UserStats counter each award reason bumps.
_REASON_COUNTER = {
    "event_confirmed": "confirmed_total",
    "event_verified": "verified_total",
    "event_fixed": "fixed_total",
    "event_promoted": None,
}


def get_stats(db: Session, user: User) -> UserStats:
    """Fetch (or create) the denormalized stats row for a user."""
    stats = db.get(UserStats, user.id)
    if stats is None:
        stats = UserStats(user_id=user.id, city=user.city)
        db.add(stats)
        db.flush()
    elif user.city and stats.city != user.city:
        stats.city = user.city
    return stats


def notify(db: Session, user_id, kind: str, title: str, body: str = "",
           link: str = "/impact", ref_id=None) -> None:
    """Insert an in-app notification and trim the user's backlog to 100."""
    db.add(Notification(
        user_id=user_id, kind=kind, title=title, body=body, link=link, ref_id=ref_id,
    ))
    db.flush()
    count = (
        db.query(func.count(Notification.id))
        .filter(Notification.user_id == user_id)
        .scalar() or 0
    )
    if count > _NOTIFICATIONS_KEPT_PER_USER:
        overflow_ids = [
            row[0] for row in (
                db.query(Notification.id)
                .filter(Notification.user_id == user_id)
                .order_by(Notification.created_at.asc())
                .limit(count - _NOTIFICATIONS_KEPT_PER_USER)
                .all()
            )
        ]
        if overflow_ids:
            db.query(Notification).filter(Notification.id.in_(overflow_ids)) \
                .delete(synchronize_session=False)


def _award_badge(db: Session, user_id, badge_key: str) -> bool:
    """Insert a badge if not already held. Returns True when newly awarded."""
    try:
        with db.begin_nested():
            db.add(UserBadge(user_id=user_id, badge_key=badge_key))
            db.flush()
    except IntegrityError:
        return False
    label, description, _ = BADGES[badge_key]
    notify(db, user_id, "badge", f"New badge: {label}", description)
    return True


def check_badges(db: Session, user_id, stats: UserStats) -> None:
    for key, (_label, _desc, check) in BADGES.items():
        if check(stats):
            _award_badge(db, user_id, key)


def award(db: Session, user: User, reason: str, ref_id,
          notify_title: str | None = None, notify_body: str = "") -> bool:
    """Credit POINTS[reason] to a user, once per (reason, ref_id).

    Returns True when the ledger row was new (points actually granted).
    """
    points = POINTS[reason]
    try:
        with db.begin_nested():
            db.add(UserPoints(user_id=user.id, points=points, reason=reason, ref_id=ref_id))
            db.flush()
    except IntegrityError:
        return False   # already awarded — replay or double hook, by design a no-op

    stats = get_stats(db, user)
    stats.points_total = (stats.points_total or 0) + points
    counter = _REASON_COUNTER.get(reason)
    if counter:
        setattr(stats, counter, (getattr(stats, counter) or 0) + 1)

    if notify_title:
        notify(db, user.id, "points", notify_title,
               notify_body or f"You earned {points} points.", ref_id=ref_id)
    check_badges(db, user.id, stats)
    return True


def record_report(db: Session, user: User) -> None:
    """Update report counters and the day streak. Never grants points."""
    stats = get_stats(db, user)
    stats.reports_total = (stats.reports_total or 0) + 1

    local_now = datetime.now(tz=_LOCAL_TZ)
    today = local_now.date()
    if stats.last_report_date == today:
        pass
    elif stats.last_report_date == today - timedelta(days=1):
        stats.current_streak_days = (stats.current_streak_days or 0) + 1
    else:
        stats.current_streak_days = 1
    stats.last_report_date = today
    stats.best_streak_days = max(stats.best_streak_days or 0, stats.current_streak_days or 0)

    if local_now.hour >= 22 or local_now.hour < 5:
        _award_badge(db, user.id, "night_reporter")
    check_badges(db, user.id, stats)


def event_supporters(db: Session, event_id) -> list[User]:
    """Distinct signed-in accounts whose sighting/confirm supports the event."""
    user_ids = [
        row[0] for row in (
            db.query(distinct(LiveReport.user_id))
            .filter(
                LiveReport.event_id == event_id,
                LiveReport.kind.in_(("sighting", "confirm")),
                LiveReport.user_id.isnot(None),
            )
            .all()
        )
    ]
    if not user_ids:
        return []
    return db.query(User).filter(User.id.in_(user_ids)).all()


_STATUS_RANK = {"unverified": 0, "confirmed": 1, "verified": 2}


def award_status_crossings(db: Session, event, old_status: str) -> None:
    """Grant points to every supporter for each newly reached status tier."""
    old_rank = _STATUS_RANK.get(old_status or "unverified", 0)
    new_rank = _STATUS_RANK.get(event.status or "unverified", 0)
    if new_rank <= old_rank:
        return
    supporters = event_supporters(db, event.id)
    label = (event.damage_type or "hazard").replace("_", " ")
    for u in supporters:
        if old_rank < 1 <= new_rank:
            award(db, u, "event_confirmed", event.id,
                  "Your report was confirmed",
                  f"Other drivers confirmed the {label} you reported. +10 points.")
        if old_rank < 2 <= new_rank:
            award(db, u, "event_verified", event.id,
                  "Your report was verified",
                  f"The {label} you reported is now fully verified. +15 points.")


def award_event_outcome(db: Session, event, reason: str, title: str, body: str) -> None:
    """Operator outcome (fixed / promoted): credit every supporter."""
    for u in event_supporters(db, event.id):
        award(db, u, reason, event.id, title, body)
