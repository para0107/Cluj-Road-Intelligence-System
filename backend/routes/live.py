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
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timedelta, timezone, date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import case, func, distinct
from sqlalchemy.orm import Session
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID, ST_Distance

from backend.database import get_db, SessionLocal
from backend.models import Detection
from backend.models_live import LiveEvent, LiveReport, LiveDevice
from backend.models_auth import User
from backend.auth import get_current_user, require_operator, create_token
from backend import gamification
from backend.ratelimit import Limiter, rate_limited
from backend.schemas_live import (
    LiveReportCreate, LiveVoteRequest, LiveEventRead,
    LiveEventListResponse, LiveActionResponse, LiveStatsResponse,
    DevicePairRequest, DeviceClaimRequest, LiveDeviceRead,
    DeviceListResponse, DeviceClaimResponse,
    TriageListResponse, PromoteResponse,
)
from backend.live_manager import manager
from backend.notify import send_thankyou_email

router = APIRouter()

# Thank-you e-mail throttle: at most one per user per day (in-memory — resets
# on restart, which only means an extra thank-you, never a lost report).
_thanked_on: dict = {}

# ── Tunables (env-overridable, sane demo defaults) ─────────────────────────
_CLUSTER_RADIUS_M = float(os.getenv("LIVE_CLUSTER_RADIUS_M", "25"))
_EVENT_TTL_H = float(os.getenv("LIVE_EVENT_TTL_H", "72"))
_CONFIRM_DEVICES = int(os.getenv("LIVE_CONFIRM_DEVICES", "2"))
_VERIFY_DEVICES = int(os.getenv("LIVE_VERIFY_DEVICES", "3"))
_DISPUTE_MIN = int(os.getenv("LIVE_DISPUTE_MIN", "2"))

# ── Abuse limits (in-process, per user unless noted) ────────────────────────
# The client's drive mode already spaces auto-reports 8 s apart; the server
# enforces its own floor plus a daily cap so points cannot be farmed and a
# hostile script cannot flood the map. Votes and pairing get sane budgets.
_report_cooldown = Limiter(
    "live_report", max_events=1, window_s=15.0,
    detail="You are reporting very fast. Wait a moment and try again.",
)
_report_daily = Limiter(
    "live_report_daily", max_events=40, window_s=86400.0,
    detail="Daily report limit reached. Thank you for today, try again tomorrow.",
)
_vote_limiter = Limiter(
    "live_vote", max_events=30, window_s=3600.0,
    detail="Too many confirmations or disputes this hour. Try again later.",
)
_pair_limiter = Limiter(
    "device_pair", max_events=10, window_s=3600.0,
    detail="Too many pairing attempts. Try again later.",
)
_claim_limiter = Limiter(
    "device_claim", max_events=10, window_s=3600.0,
    detail="Too many pairing code attempts from this address. Try again later.",
)

# ── Read micro-cache ────────────────────────────────────────────────────────
# GET /live/events and /live/stats are the polling fallback for clients
# without a WebSocket. A thousand phones polling every 5 s would mean ~200
# DB queries/s for byte-identical answers, so reads are served from a short
# in-process cache and every mutation clears it (a client's own report is
# visible on its very next poll). WS pushes are unaffected.
_READ_CACHE_TTL_S = float(os.getenv("LIVE_READ_CACHE_S", "2.0"))
_read_cache: dict = {}          # key -> (expires_monotonic, value)
_read_cache_lock = threading.Lock()


def _cached_read(key: str, build):
    now = time.monotonic()
    with _read_cache_lock:
        hit = _read_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = build()
    with _read_cache_lock:
        _read_cache[key] = (now + _READ_CACHE_TTL_S, value)
    return value


def _invalidate_read_cache() -> None:
    with _read_cache_lock:
        _read_cache.clear()


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
    _invalidate_read_cache()   # every mutation flows through here
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


def _touch_device(db: Session, device_id: str, counted: bool = False) -> None:
    """
    If the signalling device_id belongs to a paired device: reject it when the
    owner revoked it, otherwise bump last_seen (and reports_sent for accepted
    sightings). Unknown device ids stay allowed — pairing is opt-in, the
    anonymous-device crowd flow keeps working unchanged.
    """
    device = db.query(LiveDevice).filter(LiveDevice.device_id == device_id).first()
    if device is None:
        return
    if not device.is_active:
        raise HTTPException(
            status_code=403,
            detail="This device was disconnected by its owner. Pair it again from the Live page.",
        )
    device.last_seen_at = _now()
    if counted:
        device.reports_sent = (device.reports_sent or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# POST /live/reports — sight damage (auto-cluster or create)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/live/reports", response_model=LiveActionResponse, status_code=201,
    dependencies=[
        Depends(rate_limited(_report_cooldown, by="user")),
        Depends(rate_limited(_report_daily, by="user")),
    ],
)
def create_report(
    payload: LiveReportCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _sweep_expired(db)
    _touch_device(db, payload.device_id, counted=True)

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
        user_id=user.id,
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

    # Flush so _recount's distinct-device queries see THIS report too
    # (autoflush is off; without this the counters lagged one signal behind
    # the documented LIVE_CONFIRM/VERIFY_DEVICES thresholds).
    old_status = "unverified" if created else (event.status or "unverified")
    db.flush()
    _recount(db, event)

    # Streaks/badges for the reporter; points only on validation crossings.
    gamification.record_report(db, user)
    gamification.award_status_crossings(db, event, old_status)

    db.commit()
    db.refresh(event)

    _broadcast("event_upsert", event)

    # Thank the reporter by e-mail (free SMTP; no-op when unconfigured),
    # at most once per day so a driving session doesn't flood their inbox.
    # The dict resets each day so it can never grow past one day's users.
    today = date.today()
    if _thanked_on.get("_day") != today:
        _thanked_on.clear()
        _thanked_on["_day"] = today
    if user.email and user.id not in _thanked_on:
        _thanked_on[user.id] = True
        send_thankyou_email(user.email, user.username, payload.damage_type)

    return LiveActionResponse(action="created" if created else "merged", event=_read(event))


# ─────────────────────────────────────────────────────────────────────────────
# Confirm / dispute / resolve
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/live/events/{event_id}/confirm", response_model=LiveActionResponse,
    dependencies=[Depends(rate_limited(_vote_limiter, by="user"))],
)
def confirm_event(
    event_id: UUID,
    payload: LiveVoteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    event = _get_active_event(db, event_id)
    _touch_device(db, payload.device_id)

    # Idempotent per device — confirming twice does not double-count
    if not _has_vote(db, event.id, payload.device_id, ("sighting", "confirm")):
        db.add(LiveReport(
            event_id=event.id, device_id=payload.device_id, user_id=user.id,
            kind="confirm", note=payload.note,
        ))
    event.last_reported = _now()
    event.expires_at = _new_expiry()

    old_status = event.status or "unverified"
    db.flush()   # make the pending confirm visible to _recount (autoflush off)
    _recount(db, event)
    gamification.award_status_crossings(db, event, old_status)

    db.commit()
    db.refresh(event)

    _broadcast("event_upsert", event)
    return LiveActionResponse(action="confirmed", event=_read(event))


@router.post(
    "/live/events/{event_id}/dispute", response_model=LiveActionResponse,
    dependencies=[Depends(rate_limited(_vote_limiter, by="user"))],
)
def dispute_event(
    event_id: UUID,
    payload: LiveVoteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    event = _get_active_event(db, event_id)
    _touch_device(db, payload.device_id)

    if not _has_vote(db, event.id, payload.device_id, ("dispute",)):
        db.add(LiveReport(
            event_id=event.id, device_id=payload.device_id, user_id=user.id,
            kind="dispute", note=payload.note,
        ))
    db.flush()   # make the pending dispute visible to _recount (autoflush off)
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

    label = (event.damage_type or "hazard").replace("_", " ")
    gamification.award_event_outcome(
        db, event, "event_fixed",
        "A hazard you reported was fixed",
        f"The city resolved the {label} you reported. Thank you. +25 points.",
    )
    db.commit()

    _broadcast("event_removed", None, event_id=event.id)
    return LiveActionResponse(action="resolved", event=None)


# ─────────────────────────────────────────────────────────────────────────────
# Triage (operator): citizen events → official detections
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/triage", response_model=TriageListResponse)
def triage_inbox(
    status_eq: str | None = Query(None, alias="status", pattern="^(unverified|confirmed|verified)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _op: User = Depends(require_operator),
):
    """Active citizen events waiting for an operator decision.

    Most community-validated first (verified, then confirmed), newest signal
    first within each tier.
    """
    _sweep_expired(db)
    q = db.query(LiveEvent).filter(LiveEvent.is_active.is_(True))
    if status_eq:
        q = q.filter(LiveEvent.status == status_eq)
    total = q.count()
    tier = case(
        (LiveEvent.status == "verified", 2),
        (LiveEvent.status == "confirmed", 1),
        else_=0,
    )
    items = (
        q.order_by(tier.desc(), LiveEvent.last_reported.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return TriageListResponse(total=total, items=[_read(e) for e in items])


@router.post("/live/events/{event_id}/promote", response_model=PromoteResponse)
def promote_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    _op: User = Depends(require_operator),
):
    """Turn a citizen-reported live event into an official detection row.

    The detection lands on the survey map, in the priority queue, and in
    work-order planning like any pipeline result; the live event leaves the
    map and every supporting reporter is credited and notified.
    """
    event = _get_active_event(db, event_id)

    today = date.today()
    detection = Detection(
        geom=ST_SetSRID(ST_MakePoint(event.longitude, event.latitude), 4326),
        latitude=event.latitude,
        longitude=event.longitude,
        damage_type=event.damage_type,
        confidence=event.max_confidence or 0.5,
        severity=event.severity,
        first_detected=event.first_reported.date() if event.first_reported else today,
        last_detected=today,
        detection_count=max(event.report_count or 1, 1),
        survey_date=today,
        survey_video_file="live_promotion",
    )
    detection.compute_priority_score()
    db.add(detection)
    db.flush()

    event.promoted_detection_id = detection.id
    event.is_active = False

    label = (event.damage_type or "hazard").replace("_", " ")
    gamification.award_event_outcome(
        db, event, "event_promoted",
        "Your report is now official",
        f"The city accepted the {label} you reported as an official damage record. +20 points.",
    )
    db.commit()

    _broadcast("event_removed", None, event_id=event.id)
    return PromoteResponse(event_id=event.id, detection_id=detection.id)


@router.post("/live/events/{event_id}/dismiss", response_model=LiveActionResponse)
def dismiss_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    _op: User = Depends(require_operator),
):
    """Operator triage: not actionable (duplicate, off-road, bad signal)."""
    event = _get_active_event(db, event_id)
    event.dismissed_at = _now()
    event.is_active = False
    db.commit()

    _broadcast("event_removed", None, event_id=event.id)
    return LiveActionResponse(action="dismissed", event=None)


# ─────────────────────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/events", response_model=LiveEventListResponse)
def list_events(db: Session = Depends(get_db)):
    """Active events — the polling fallback for clients without WebSocket."""
    def build():
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
    return _cached_read("events", build)


@router.get("/live/stats", response_model=LiveStatsResponse)
def live_stats(db: Session = Depends(get_db)):
    def build():
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
    return _cached_read("stats", build)


# ─────────────────────────────────────────────────────────────────────────────
# Paired devices — connect a phone / dashcam so detections auto-upload
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L
_PAIR_CODE_LEN = 8
_PAIR_CODE_TTL_MIN = int(os.getenv("LIVE_PAIR_CODE_TTL_MIN", "15"))


def _new_pair_code() -> str:
    import secrets
    return "".join(secrets.choice(_PAIR_CODE_ALPHABET) for _ in range(_PAIR_CODE_LEN))


@router.post(
    "/live/devices/pair", response_model=LiveDeviceRead, status_code=201,
    dependencies=[Depends(rate_limited(_pair_limiter, by="user"))],
)
def pair_device(
    payload: DevicePairRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Connect a sensor to the caller's account.

    * body WITH device_id  → instant self-registration (the phone/browser
      calling this IS the device — used by the Live page's drive mode).
    * body WITHOUT device_id → creates a pending device and returns a
      single-use pairing code; run
      `python pipeline/live_pipeline.py --pair <CODE>` on the dashcam PC.
    """
    if payload.device_id:
        existing = (
            db.query(LiveDevice).filter(LiveDevice.device_id == payload.device_id).first()
        )
        if existing:
            if existing.user_id != user.id:
                raise HTTPException(409, "This device is already paired to another account.")
            existing.name = payload.name
            existing.kind = payload.kind
            existing.is_active = True
            existing.last_seen_at = _now()
            db.commit()
            db.refresh(existing)
            return existing
        device = LiveDevice(
            user_id=user.id,
            device_id=payload.device_id,
            name=payload.name,
            kind=payload.kind,
            last_seen_at=_now(),
        )
        db.add(device)
        db.commit()
        db.refresh(device)
        return device

    # Pairing-code flow for an external agent
    device = LiveDevice(
        user_id=user.id,
        name=payload.name,
        kind=payload.kind,
        pair_code=_new_pair_code(),
        pair_code_expires_at=_now() + timedelta(minutes=_PAIR_CODE_TTL_MIN),
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


@router.post(
    "/live/devices/claim", response_model=DeviceClaimResponse,
    dependencies=[Depends(rate_limited(_claim_limiter, by="ip"))],
)
def claim_device(payload: DeviceClaimRequest, db: Session = Depends(get_db)):
    """
    Exchange a pairing code for a JWT (called by the edge agent — the vehicle
    machine never needs the account password). Codes are single-use and expire
    after LIVE_PAIR_CODE_TTL_MIN minutes.
    """
    code = payload.code.strip().upper()
    device = (
        db.query(LiveDevice)
        .filter(LiveDevice.pair_code == code, LiveDevice.device_id.is_(None))
        .first()
    )
    if not device or (device.pair_code_expires_at and device.pair_code_expires_at < _now()):
        raise HTTPException(404, "That code was not found or has expired. Generate a new one from the Live page.")

    clash = db.query(LiveDevice).filter(LiveDevice.device_id == payload.device_id).first()
    if clash:
        raise HTTPException(409, "This device id is already paired. Disconnect it first.")

    owner = db.query(User).filter(User.id == device.user_id).first()
    if not owner or not owner.is_active:
        raise HTTPException(403, "The owning account is disabled.")

    device.device_id = payload.device_id
    if payload.name:
        device.name = payload.name
    device.pair_code = None
    device.pair_code_expires_at = None
    device.last_seen_at = _now()
    db.commit()
    db.refresh(device)

    return DeviceClaimResponse(
        access_token=create_token(owner),
        device=LiveDeviceRead.model_validate(device),
    )


@router.get("/live/devices", response_model=DeviceListResponse)
def list_devices(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """The caller's paired devices (pending pairing codes included)."""
    # Expire stale pending codes lazily
    stale = (
        db.query(LiveDevice)
        .filter(
            LiveDevice.user_id == user.id,
            LiveDevice.device_id.is_(None),
            LiveDevice.pair_code_expires_at < _now(),
        )
        .all()
    )
    for d in stale:
        db.delete(d)
    if stale:
        db.commit()

    items = (
        db.query(LiveDevice)
        .filter(LiveDevice.user_id == user.id)
        .order_by(LiveDevice.created_at.desc())
        .all()
    )
    return DeviceListResponse(total=len(items), items=items)


@router.delete("/live/devices/{device_pk}", response_model=DeviceListResponse)
def disconnect_device(
    device_pk: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Disconnect a device. Pending (unclaimed) devices are deleted outright;
    claimed ones are revoked so their future reports are rejected.
    """
    device = db.query(LiveDevice).filter(LiveDevice.id == device_pk).first()
    if not device or (device.user_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Device not found.")

    if device.device_id is None:
        db.delete(device)
    else:
        device.is_active = False
        device.pair_code = None
        device.pair_code_expires_at = None
    db.commit()

    items = (
        db.query(LiveDevice)
        .filter(LiveDevice.user_id == user.id)
        .order_by(LiveDevice.created_at.desc())
        .all()
    )
    return DeviceListResponse(total=len(items), items=items)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — push channel
# ─────────────────────────────────────────────────────────────────────────────

# Comma-separated allow-list shared with the CORS middleware. "*" disables the
# WS origin check (dev-friendly); real deployments set CORS_ORIGINS and get it.
_WS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_WS_MAX_MSG_CHARS = 512
_WS_MAX_MSGS_PER_MIN = 30


@router.websocket("/live/ws")
async def live_ws(ws: WebSocket):
    """
    On connect: send {"type": "hello", "events": [...]} with the current
    snapshot, then push every mutation as it happens. Incoming client
    messages are treated as keep-alive pings and ignored (size- and
    rate-capped so a hostile client cannot spam the loop).
    """
    origin = ws.headers.get("origin")
    if "*" not in _WS_ORIGINS and origin and origin.rstrip("/") not in _WS_ORIGINS:
        await ws.close(code=4403)
        return

    ip = ws.headers.get("x-real-ip") or (ws.client.host if ws.client else "?")
    if not await manager.connect(ws, ip):
        await ws.close(code=1013)   # try again later: per-IP or total cap hit
        return

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
        msg_window_start = time.monotonic()
        msg_count = 0
        while True:
            text = await ws.receive_text()   # keep-alive; content ignored
            if len(text) > _WS_MAX_MSG_CHARS:
                await ws.close(code=1009)    # message too big
                break
            now_mono = time.monotonic()
            if now_mono - msg_window_start > 60.0:
                msg_window_start, msg_count = now_mono, 0
            msg_count += 1
            if msg_count > _WS_MAX_MSGS_PER_MIN:
                await ws.close(code=1008)    # policy violation: ping flood
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await manager.disconnect(ws)
