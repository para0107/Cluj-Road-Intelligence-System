"""
backend/models_engagement.py

Citizen engagement: points, badges, streaks, and in-app notifications.

Design rules (anti-spam economy):
  * Raw reports earn NO points. Points are awarded only when an event reaches
    community-validated states (confirmed / verified), when an operator fixes
    or promotes it — states a single spammer cannot reach alone (distinct-
    device escalation in routes/live.py).
  * `user_points` is an append-only ledger; `user_stats` is the denormalized
    snapshot the leaderboard reads (no SUM over the ledger per request).
  * The unique (user_id, reason, ref_id) constraint makes every award
    idempotent: replays and double-hooks insert-conflict and are ignored.

Tables are created by Base.metadata.create_all at startup; fresh Docker
volumes get them from db/init/06_engagement.sql — keep both in sync.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import (
    Column, String, Integer, Date, DateTime, Boolean, ForeignKey, Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database import Base


class UserPoints(Base):
    """Append-only points ledger. One row per award (or penalty)."""

    __tablename__ = "user_points"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    points = Column(Integer, nullable=False)
    reason = Column(String(40), nullable=False)   # event_confirmed | event_verified | ...
    ref_id = Column(UUID(as_uuid=True))           # usually the live_event id

    __table_args__ = (
        Index("idx_user_points_user", "user_id", "created_at"),
        UniqueConstraint("user_id", "reason", "ref_id", name="uq_user_points_award"),
    )

    def __repr__(self) -> str:
        return f"<UserPoints user={self.user_id} {self.points:+d} {self.reason}>"


class UserStats(Base):
    """Denormalized per-user snapshot read by leaderboards and the Impact page."""

    __tablename__ = "user_stats"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    points_total = Column(Integer, nullable=False, default=0)
    reports_total = Column(Integer, nullable=False, default=0)
    confirmed_total = Column(Integer, nullable=False, default=0)
    verified_total = Column(Integer, nullable=False, default=0)
    fixed_total = Column(Integer, nullable=False, default=0)

    current_streak_days = Column(Integer, nullable=False, default=0)
    best_streak_days = Column(Integer, nullable=False, default=0)
    last_report_date = Column(Date)

    # Copied from users.city at write time so the city leaderboard is one
    # indexed scan (no join against users on every poll).
    city = Column(String(80))

    __table_args__ = (
        Index("idx_user_stats_points", "points_total"),
        Index("idx_user_stats_city", "city"),
    )

    def __repr__(self) -> str:
        return f"<UserStats user={self.user_id} points={self.points_total}>"


class UserBadge(Base):
    __tablename__ = "user_badges"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    awarded_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    badge_key = Column(String(40), nullable=False)   # keys defined in backend/gamification.py

    __table_args__ = (
        UniqueConstraint("user_id", "badge_key", name="uq_user_badges_once"),
        Index("idx_user_badges_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<UserBadge user={self.user_id} {self.badge_key}>"


class Notification(Base):
    """In-app notification. Free transport: the bell polls /notifications."""

    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(30), nullable=False)     # points | badge | fixed | promoted | info
    title = Column(String(120), nullable=False)
    body = Column(String(300))
    link = Column(String(120))                    # SPA route, e.g. /impact
    ref_id = Column(UUID(as_uuid=True))
    read_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_notifications_user", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Notification user={self.user_id} kind={self.kind} read={bool(self.read_at)}>"
