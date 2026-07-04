"""
backend/schemas_live.py

Pydantic v2 schemas for Live (Waze-like) mode.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


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


# ─────────────────────────────────────────────
# Paired devices (phone drive mode / dashcam edge agents)
# ─────────────────────────────────────────────

class DevicePairRequest(BaseModel):
    """
    Register a sensor on the caller's account.
    With device_id → instant self-registration (phone/browser drive mode).
    Without device_id → returns a single-use pairing code for an edge agent.
    """
    name: str = Field(..., min_length=1, max_length=80)
    kind: str = Field("dashcam")
    device_id: Optional[str] = Field(None, min_length=3, max_length=64)

    @field_validator("kind")
    @classmethod
    def kind_must_be_known(cls, v: str) -> str:
        if v not in ("phone", "dashcam", "browser", "simulator"):
            raise ValueError("kind must be phone, dashcam, browser, or simulator.")
        return v


class DeviceClaimRequest(BaseModel):
    """Edge agent exchanges a pairing code for a token (no password on the car PC)."""
    code: str = Field(..., min_length=4, max_length=12)
    device_id: str = Field(..., min_length=3, max_length=64)
    name: Optional[str] = Field(None, max_length=80)


class LiveDeviceRead(BaseModel):
    id: UUID
    name: str
    kind: str
    device_id: Optional[str]
    is_active: bool
    last_seen_at: Optional[datetime]
    reports_sent: int
    created_at: Optional[datetime]
    pair_code: Optional[str]                 # only present while unclaimed
    pair_code_expires_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class DeviceListResponse(BaseModel):
    total: int
    items: List[LiveDeviceRead]


class DeviceClaimResponse(BaseModel):
    """The claimed device plus a JWT the edge agent reports with."""
    access_token: str
    token_type: str = "bearer"
    device: LiveDeviceRead
