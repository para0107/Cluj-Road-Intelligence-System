"""
backend/routes/live.py

Live (Waze-like) mode — crowd-validated road hazards in real time.

POST /live/reports                a camera/user reports damage at a location;
                                  the server clusters it into an existing
                                  active event within LIVE_CLUSTER_RADIUS_M
                                  (same damage_type) or creates a new one.
POST /live/events/{id}/confirm    another device says "still there"
POST /live/events/{id}/dispute    another device says "not there" — enough
                                  disputes remove the event
POST /live/events/{id}/resolve    operator marks the hazard fixed
GET  /live/events                 active, non-expired events (polling fallback)
GET  /live/stats                  live-mode counters for the UI header
WS   /live/ws                     push channel: hello snapshot + every mutation

Design notes
------------
* All state lives in PostGIS → REST handlers are stateless and the API scales
  horizontally; only the WS fan-out is per-process (see live_manager.py for
  the Redis pub/sub upgrade path).
* Validation is by DISTINCT device: 1 device = unverified,
  LIVE_CONFIRM_DEVICES = confirmed, LIVE_VERIFY_DEVICES = verified.
  A device confirming twice is idempotent — no self-boosting.
* Events auto-expire: every supporting signal pushes `expires_at` forward by
  LIVE_EVENT_TTL_H; reads lazily sweep expired rows. No cron required.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import func, distinct
from sqlalchemy.orm import Session
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID, ST_Distance

from backend.database import get_db, SessionLocal
from backend.models_live import LiveEvent, LiveReport
from backend.models_auth import User
from backend.auth import get_current_user, require_operator
from backend.schemas_live import (
    LiveReportCreate, LiveVoteRequest, LiveEventRead,
    LiveEventListResponse, LiveActionResponse, LiveStatsResponse,
)
from backend.live_manager import manager

router = APIRouter()

# ── Tunables (env-overridable, sane demo defaults) ─────────────────────────
_CLUSTER_RADIUS_M = float(os.getenv("LIVE_CLUSTER_RADIUS_M", "25"))
_EVENT_TTL_H = float(os.getenv("LIVE_EVENT_TTL_H", "72"))
_CONFIRM_DEVICES = int(os.getenv("LIVE_CONFIRM_DEVICES", "2"))
_VERIFY_DEVICES = int(os.getenv("LIVE_VERIFY_DEVICES", "3"))
_DISPUTE_MIN = int(os.getenv("LIVE_DISPUTE_MIN", "2"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_expiry() -> datetime:
    return _now() + timedelta(hours=_EVENT_TTL_H)


def _read(event: LiveEvent) -> LiveEventRead:
    return LiveEventRead.model_validate(event)


def _broadcast(kind: str, event: LiveEvent | None, event_id=None) -> None:
    """Push a mutation to every connected live client (fire-and-forget)."""
    manager.broadcast_from_thread({
        "type": kind,                                     # event_upsert | event_removed
        "event": _read(event).model_dump() if event else None,
        "event_id": str(event_id or (event.id if event else None)),
        "ts": _now().isoformat(),
    })


def _sweep_expired(db: Session) -> None:
    """Lazily deactivate expired events. Cheap: indexed partial scan."""
    expired = (
        db.query(LiveEvent)
        .filter(LiveEvent.is_active.is_(True), LiveEvent.expires_at < _now())
        .all()
    )
    if not expired:
        return
    for e in expired:
        e.is_active = False
    db.commit()
    for e in expired:
        _broadcast("event_removed", None, event_id=e.id)


def _recount(db: Session, event: LiveEvent) -> None:
    """Recompute distinct-device counters and the validation status."""
    supporters = (
        db.query(func.count(distinct(LiveReport.device_id)))
        .filter(LiveReport.event_id == event.id, LiveReport.kind.in_(("sighting", "confirm")))
        .scalar() or 0
    )
    disputers = (
        db.query(func.count(distinct(LiveReport.device_id)))
        .filter(LiveReport.event_id == event.id, LiveReport.kind == "dispute")
        .scalar() or 0
    )
    reports = (
        db.query(func.count(LiveReport.id))
        .filter(LiveReport.event_id == event.id, LiveReport.kind == "sighting")
        .scalar() or 0
    )
    event.reporter_devices = supporters
    event.dispute_devices = disputers
    event.report_count = reports
    if supporters >= _VERIFY_DEVICES:
        event.status = "verified"
    elif supporters >= _CONFIRM_DEVICES:
        event.status = "confirmed"
    else:
        event.status = "unverified"


def _has_vote(db: Session, event_id, device_id: str, kinds: tuple[str, ...]) -> bool:
    return (
        db.query(LiveReport.id)
        .filter(
            LiveReport.event_id == event_id,
            LiveReport.device_id == device_id,
            LiveReport.kind.in_(kinds),
        )
        .first()
        is not None
    )


def _get_active_event(db: Session, event_id: UUID) -> LiveEvent:
    event = db.query(LiveEvent).filter(LiveEvent.id == event_id).first()
    if not event or not event.is_active:
        raise HTTPException(status_code=404, detail="Live event not found or no longer active.")
    return event


# ─────────────────────────────────────────────────────────────────────────────
# POST /live/reports — sight damage (auto-cluster or create)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/live/reports", response_model=LiveActionResponse, status_code=201)
def create_report(
    payload: LiveReportCreate,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    _sweep_expired(db)

    point = ST_SetSRID(ST_MakePoint(payload.longitude, payload.latitude), 4326)

    # Nearest active event of the same type within the cluster radius
    event = (
        db.query(LiveEvent)
        .filter(
            LiveEvent.is_active.is_(True),
            LiveEvent.damage_type == payload.damage_type,
            ST_DWithin(
                LiveEvent.geom.cast(Geography),
                point.cast(Geography),
                _CLUSTER_RADIUS_M,
            ),
        )
        .order_by(ST_Distance(LiveEvent.geom.cast(Geography), point.cast(Geography)))
        .first()
    )

    created = event is None
    if created:
        event = LiveEvent(
            geom=point,
            latitude=payload.latitude,
            longitude=payload.longitude,
            damage_type=payload.damage_type,
            max_confidence=payload.confidence or 0.0,
            severity=payload.severity,
            expires_at=_new_expiry(),
        )
        db.add(event)
        db.flush()  # get event.id for the report row

    report = LiveReport(
        event_id=event.id,
        device_id=payload.device_id,
        kind="sighting",
        latitude=payload.latitude,
        longitude=payload.longitude,
        confidence=payload.confidence,
        severity=payload.severity,
        note=payload.note,
    )
    db.add(report)

    # Merge signal quality into the event
    if payload.confidence and payload.confidence > (event.max_confidence or 0):
        event.max_confidence = payload.confidence
    if payload.severity and payload.severity > (event.severity or 0):
        event.severity = payload.severity
    event.last_reported = _now()
    event.expires_at = _new_expiry()

    _recount(db, event)
    db.commit()
    db.refresh(event)

    _broadcast("event_upsert", event)
    return LiveActionResponse(action="created" if created else "merged", event=_read(event))


# ─────────────────────────────────────────────────────────────────────────────
# Confirm / dispute / resolve
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/live/events/{event_id}/confirm", response_model=LiveActionResponse)
def confirm_event(
    event_id: UUID,
    payload: LiveVoteRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    event = _get_active_event(db, event_id)

    # Idempotent per device — confirming twice does not double-count
    if not _has_vote(db, event.id, payload.device_id, ("sighting", "confirm")):
        db.add(LiveReport(
            event_id=event.id, device_id=payload.device_id,
            kind="confirm", note=payload.note,
        ))
    event.last_reported = _now()
    event.expires_at = _new_expiry()
    _recount(db, event)
    db.commit()
    db.refresh(event)

    _broadcast("event_upsert", event)
    return LiveActionResponse(action="confirmed", event=_read(event))


@router.post("/live/events/{event_id}/dispute", response_model=LiveActionResponse)
def dispute_event(
    event_id: UUID,
    payload: LiveVoteRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    event = _get_active_event(db, event_id)

    if not _has_vote(db, event.id, payload.device_id, ("dispute",)):
        db.add(LiveReport(
            event_id=event.id, device_id=payload.device_id,
            kind="dispute", note=payload.note,
        ))
    _recount(db, event)

    # Enough independent "not there" signals remove the hazard from the map
    removed = event.dispute_devices >= max(_DISPUTE_MIN, event.reporter_devices)
    if removed:
        event.is_active = False
    db.commit()

    if removed:
        _broadcast("event_removed", None, event_id=event.id)
        return LiveActionResponse(action="removed", event=None)

    db.refresh(event)
    _broadcast("event_upsert", event)
    return LiveActionResponse(action="disputed", event=_read(event))


@router.post("/live/events/{event_id}/resolve", response_model=LiveActionResponse)
def resolve_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    _op: User = Depends(require_operator),
):
    """Operator action (municipality/admin): the hazard was repaired / cleared."""
    event = _get_active_event(db, event_id)
    event.resolved = True
    event.is_active = False
    db.commit()

    _broadcast("event_removed", None, event_id=event.id)
    return LiveActionResponse(action="resolved", event=None)


# ─────────────────────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/events", response_model=LiveEventListResponse)
def list_events(db: Session = Depends(get_db)):
    """Active events — the polling fallback for clients without WebSocket."""
    _sweep_expired(db)
    items = (
        db.query(LiveEvent)
        .filter(LiveEvent.is_active.is_(True))
        .order_by(LiveEvent.last_reported.desc())
        .limit(1000)
        .all()
    )
    return LiveEventListResponse(
        server_time=_now(),
        count=len(items),
        items=[_read(e) for e in items],
    )


@router.get("/live/stats", response_model=LiveStatsResponse)
def live_stats(db: Session = Depends(get_db)):
    hour_ago = _now() - timedelta(hours=1)
    active = db.query(func.count(LiveEvent.id)).filter(LiveEvent.is_active.is_(True)).scalar() or 0
    verified = (
        db.query(func.count(LiveEvent.id))
        .filter(LiveEvent.is_active.is_(True), LiveEvent.status == "verified")
        .scalar() or 0
    )
    reports_1h = (
        db.query(func.count(LiveReport.id))
        .filter(LiveReport.created_at >= hour_ago)
        .scalar() or 0
    )
    devices_1h = (
        db.query(func.count(distinct(LiveReport.device_id)))
        .filter(LiveReport.created_at >= hour_ago)
        .scalar() or 0
    )
    return LiveStatsResponse(
        active_events=active,
        verified_events=verified,
        reports_last_hour=reports_1h,
        devices_last_hour=devices_1h,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — push channel
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/live/ws")
async def live_ws(ws: WebSocket):
    """
    On connect: send {"type": "hello", "events": [...]} with the current
    snapshot, then push every mutation as it happens. Incoming client
    messages are treated as keep-alive pings and ignored.
    """
    await manager.connect(ws)

    # Snapshot via a short-lived session (this handler is async — do not
    # borrow the request-scoped dependency machinery here).
    db = SessionLocal()
    try:
        items = (
            db.query(LiveEvent)
            .filter(LiveEvent.is_active.is_(True), LiveEvent.expires_at > _now())
            .order_by(LiveEvent.last_reported.desc())
            .limit(1000)
            .all()
        )
        snapshot = [_read(e).model_dump() for e in items]
    finally:
        db.close()

    try:
        import json
        await ws.send_text(json.dumps(
            {"type": "hello", "events": snapshot, "clients": manager.client_count},
            default=str,
        ))
        while True:
            await ws.receive_text()   # keep-alive; content ignored
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)
