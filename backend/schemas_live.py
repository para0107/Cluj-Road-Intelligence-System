"""
backend/schemas_live.py

Pydantic v2 schemas for Live (Waze-like) mode.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────

class LiveReportCreate(BaseModel):
    """A raw sighting from one camera / user driving through a location."""
    device_id: str = Field(..., min_length=3, max_length=64)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    damage_type: str = Field(..., min_length=2, max_length=30)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    severity: Optional[int] = Field(None, ge=1, le=5)
    note: Optional[str] = Field(None, max_length=280)


class LiveVoteRequest(BaseModel):
    """Confirm ('still there') or dispute ('not there') an existing event."""
    device_id: str = Field(..., min_length=3, max_length=64)
    note: Optional[str] = Field(None, max_length=280)


# ─────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────

class LiveEventRead(BaseModel):
    id: UUID
    latitude: float
    longitude: float
    damage_type: str
    max_confidence: Optional[float]
    severity: Optional[int]
    status: str
    report_count: int
    reporter_devices: int
    dispute_devices: int
    is_active: bool
    resolved: bool
    first_reported: Optional[datetime]
    last_reported: Optional[datetime]
    expires_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class LiveEventListResponse(BaseModel):
    server_time: datetime
    count: int
    items: List[LiveEventRead]


class LiveActionResponse(BaseModel):
    """Returned by report/confirm/dispute — the event after the action."""
    action: str                       # created | merged | confirmed | disputed | removed | resolved
    event: Optional[LiveEventRead]    # None when the action removed the event


class LiveStatsResponse(BaseModel):
    active_events: int
    verified_events: int
    reports_last_hour: int
    devices_last_hour: int
