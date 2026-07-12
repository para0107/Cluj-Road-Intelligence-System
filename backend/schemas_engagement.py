"""
backend/schemas_engagement.py

Pydantic schemas for gamification (points, badges, leaderboard) and in-app
notifications.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UserStatsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    points_total: int = 0
    reports_total: int = 0
    confirmed_total: int = 0
    verified_total: int = 0
    fixed_total: int = 0
    current_streak_days: int = 0
    best_streak_days: int = 0


class BadgeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    badge_key: str
    awarded_at: Optional[datetime] = None


class PointsEntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    points: int
    reason: str
    created_at: Optional[datetime] = None


class MyReportRead(BaseModel):
    event_id: UUID
    damage_type: str
    status: str                      # unverified | confirmed | verified
    is_active: bool
    resolved: bool
    promoted: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    last_reported: Optional[datetime] = None


class ImpactResponse(BaseModel):
    stats: UserStatsRead
    rank_city: Optional[int] = None
    rank_global: Optional[int] = None
    badges: List[BadgeRead] = Field(default_factory=list)
    recent_points: List[PointsEntryRead] = Field(default_factory=list)
    my_reports: List[MyReportRead] = Field(default_factory=list)


class LeaderboardRow(BaseModel):
    rank: int
    username: str
    points_total: int
    confirmed_total: int
    city: Optional[str] = None


class LeaderboardResponse(BaseModel):
    scope: str                       # "city" | "global"
    city: Optional[str] = None
    items: List[LeaderboardRow] = Field(default_factory=list)


class NotificationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: str
    title: str
    body: Optional[str] = None
    link: Optional[str] = None
    created_at: Optional[datetime] = None
    read_at: Optional[datetime] = None


class NotificationsResponse(BaseModel):
    unread: int
    total: int
    items: List[NotificationRead] = Field(default_factory=list)


class NotificationsMarkRequest(BaseModel):
    ids: Optional[List[UUID]] = None
    all: bool = False


class NotificationsMarkResponse(BaseModel):
    updated: int
