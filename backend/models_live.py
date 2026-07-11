"""
backend/models_live.py

ORM models for Live (Waze-like) mode.

Two tables, deliberately separate from the survey-pipeline `detections` table:

  live_events   — one row per *clustered, community-validated* road hazard.
                  Created by the first sighting, then confirmed/disputed by
                  other cameras/users driving through the same location.
  live_reports  — one row per raw signal from a single device (sighting,
                  confirm, or dispute). Keeps the full audit trail and lets
                  us count *distinct devices* for validation thresholds.

Lifecycle (mirrors how Waze validates reports):
  unverified  → 1 distinct reporting device
  confirmed   → ≥ LIVE_CONFIRM_DEVICES distinct devices
  verified    → ≥ LIVE_VERIFY_DEVICES distinct devices
  removed     — disputes outweigh support, event resolved, or TTL expired

Events expire (`expires_at`) unless fresh reports keep extending them —
stale hazards fall off the map automatically, no cron needed.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import (
    Column, String, Float, Integer, SmallInteger, DateTime, Boolean, Text, ForeignKey, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from backend.database import Base


class LiveEvent(Base):
    __tablename__ = "live_events"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ── Spatial (representative position = first sighting; refined by mean) ──
    geom = Column(Geometry("POINT", srid=4326), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # ── What ─────────────────────────────────────────────────────────────────
    damage_type = Column(String(30), nullable=False)
    max_confidence = Column(Float, default=0.0)     # best detector confidence seen
    severity = Column(SmallInteger)                  # max severity reported so far

    # ── Validation state ─────────────────────────────────────────────────────
    status = Column(String(12), default="unverified")   # unverified|confirmed|verified
    report_count = Column(Integer, default=0)            # raw sighting rows
    reporter_devices = Column(Integer, default=0)        # DISTINCT devices that sighted/confirmed
    dispute_devices = Column(Integer, default=0)         # DISTINCT devices that disputed
    is_active = Column(Boolean, default=True)
    resolved = Column(Boolean, default=False)            # cleared by an operator

    # ── Temporal ─────────────────────────────────────────────────────────────
    first_reported = Column(DateTime(timezone=True), server_default=func.now())
    last_reported = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # NOTE: no explicit GIST index here — GeoAlchemy2 auto-creates a spatial
    # index named idx_live_events_geom for every Geometry column. Declaring it
    # again collides on the name and makes Base.metadata.create_all() fail,
    # which silently left the live tables missing (manual reports 500'd).
    __table_args__ = (
        Index("idx_live_events_active", "is_active", "expires_at"),
        Index("idx_live_events_type", "damage_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<LiveEvent id={self.id} type={self.damage_type} status={self.status} "
            f"devices={self.reporter_devices} lat={self.latitude:.4f} lon={self.longitude:.4f}>"
        )


class LiveDevice(Base):
    """
    A sensor paired to an account: the user's phone (browser drive mode), a
    dashcam PC running pipeline/live_pipeline.py, or the fleet simulator.

    Pairing flows
    -------------
    * Self-registration (phone/browser): the logged-in client calls
      POST /live/devices/pair with its device_id — claimed instantly.
    * Pairing code (edge agent): POST /live/devices/pair without a device_id
      returns a short-lived single-use code; the agent exchanges it via
      POST /live/devices/claim (no password typed on the vehicle machine).

    Reports whose device_id matches a revoked device are rejected, and every
    accepted signal bumps last_seen_at / reports_sent, so the Live page can
    show "your dashcam uploaded 12 hazards, last seen 2 min ago".
    """

    __tablename__ = "live_devices"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    device_id = Column(String(64), unique=True)      # NULL until a pair code is claimed
    name = Column(String(80), nullable=False)
    kind = Column(String(16), nullable=False, default="dashcam")  # phone|dashcam|browser|simulator

    pair_code = Column(String(12), unique=True)      # single-use, short-lived
    pair_code_expires_at = Column(DateTime(timezone=True))

    is_active = Column(Boolean, nullable=False, default=True)
    last_seen_at = Column(DateTime(timezone=True))
    reports_sent = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("idx_live_devices_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<LiveDevice {self.name} kind={self.kind} device_id={self.device_id} active={self.is_active}>"


class LiveReport(Base):
    __tablename__ = "live_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    event_id = Column(UUID(as_uuid=True), ForeignKey("live_events.id", ondelete="CASCADE"), nullable=False)
    device_id = Column(String(64), nullable=False)   # camera / browser / vehicle identity
    kind = Column(String(10), nullable=False)        # sighting | confirm | dispute

    latitude = Column(Float)
    longitude = Column(Float)
    confidence = Column(Float)
    severity = Column(SmallInteger)
    note = Column(Text)

    __table_args__ = (
        Index("idx_live_reports_event", "event_id"),
        Index("idx_live_reports_device", "event_id", "device_id"),
        # /live/stats counts reports in the last hour; without this the count
        # degrades to a table scan as reports accumulate. Existing volumes get
        # it from the idempotent CREATE INDEX in main.py's startup hook.
        Index("idx_live_reports_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<LiveReport event={self.event_id} device={self.device_id} kind={self.kind}>"
