"""
backend/routes/public_api.py

Developer API: key management for signed-in users and a read-only public API
under /api/v1/public/* authenticated with the X-API-Key header.

Key hygiene:
  * Keys look like "rdds_<40 hex>"; only sha256(key) and a display prefix are
    stored. The plaintext appears exactly once, in the creation response.
  * Every public endpoint is GET-only, serves aggregate/road data only, and
    never exposes users, devices, or evidence paths.
  * Per-key rate limiting uses the key row's rate_limit_per_min through the
    shared in-process limiter; 429s carry Retry-After.
"""

import hashlib
import os
import secrets
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import date, datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import Detection
from backend.models_apikeys import ApiKey
from backend.models_auth import ROLE_ADMIN, User
from backend.ratelimit import Limiter
from backend.routes.quality import QualityCell, compute_grid

router = APIRouter()

_MAX_ACTIVE_KEYS = int(os.getenv("API_KEYS_PER_USER", "3"))

# One shared limiter; the per-key budget comes from the key row. Window is a
# minute; max_events is checked manually against the row value.
_key_limiter = Limiter(
    "public_api", max_events=10_000_000, window_s=60.0,
    detail="API rate limit reached for this key. Slow down and retry.",
)

# Tiny TTL cache: key hash → (expires_monotonic, key_id, user_id, per_min)
# so a busy key does not cost one SELECT per request.
_key_cache: dict = {}
_key_cache_lock = threading.Lock()
_KEY_CACHE_TTL_S = float(os.getenv("API_KEY_CACHE_S", "30.0"))


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Key management (signed-in users)
# ─────────────────────────────────────────────────────────────────────────────

class ApiKeyRead(BaseModel):
    id: UUID
    name: str
    prefix: str
    is_active: bool
    usage_count: int
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    rate_limit_per_min: int

    model_config = {"from_attributes": True}


class ApiKeyCreated(ApiKeyRead):
    key: str   # plaintext — shown exactly once


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class ApiKeyListResponse(BaseModel):
    items: List[ApiKeyRead]


@router.post("/apikeys", response_model=ApiKeyCreated, status_code=201)
def create_api_key(
    payload: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    active = (
        db.query(func.count(ApiKey.id))
        .filter(ApiKey.user_id == user.id, ApiKey.is_active.is_(True))
        .scalar() or 0
    )
    if active >= _MAX_ACTIVE_KEYS:
        raise HTTPException(
            400, f"You already have {_MAX_ACTIVE_KEYS} active keys. Revoke one first.")

    plaintext = "rdds_" + secrets.token_hex(20)
    row = ApiKey(
        user_id=user.id,
        name=payload.name,
        prefix=plaintext[:12],
        key_hash=_hash(plaintext),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    created = ApiKeyCreated.model_validate(row, from_attributes=True)
    created.key = plaintext
    return created


@router.get("/apikeys", response_model=ApiKeyListResponse)
def list_api_keys(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    items = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return ApiKeyListResponse(items=items)


@router.delete("/apikeys/{key_id}")
def revoke_api_key(
    key_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = db.get(ApiKey, key_id)
    if not row or (row.user_id != user.id and user.role != ROLE_ADMIN):
        raise HTTPException(404, "API key not found.")
    row.is_active = False
    db.commit()
    with _key_cache_lock:
        _key_cache.pop(row.key_hash, None)
    return {"revoked": True, "id": str(key_id)}


# ─────────────────────────────────────────────────────────────────────────────
# Public API auth dependency
# ─────────────────────────────────────────────────────────────────────────────

def require_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    if not x_api_key or not x_api_key.startswith("rdds_"):
        raise HTTPException(401, "Missing or malformed X-API-Key header.")
    key_hash = _hash(x_api_key)

    now = time.monotonic()
    with _key_cache_lock:
        hit = _key_cache.get(key_hash)
    if hit and hit[0] > now:
        key_id, per_min = hit[1], hit[2]
    else:
        row = (
            db.query(ApiKey)
            .filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
            .first()
        )
        if not row:
            raise HTTPException(401, "Unknown or revoked API key.")
        key_id, per_min = row.id, row.rate_limit_per_min or 60
        with _key_cache_lock:
            if len(_key_cache) > 1000:
                _key_cache.clear()
            _key_cache[key_hash] = (now + _KEY_CACHE_TTL_S, key_id, per_min)

    # Manual budget check against the row's per-minute allowance.
    bucket_key = str(key_id)
    with _key_limiter._lock:
        count, start = _key_limiter._counts.get(bucket_key, (0, time.time()))
        if time.time() - start > 60.0:
            count, start = 0, time.time()
        count += 1
        if count > per_min:
            retry = max(1, int(start + 60.0 - time.time()) + 1)
            raise HTTPException(
                429, "API rate limit reached for this key. Slow down and retry.",
                headers={"Retry-After": str(retry)},
            )
        _key_limiter._counts[bucket_key] = (count, start)

    # Usage accounting (cheap single-row UPDATE, no ORM round-trip).
    db.query(ApiKey).filter(ApiKey.id == key_id).update(
        {ApiKey.usage_count: ApiKey.usage_count + 1, ApiKey.last_used_at: _now()},
        synchronize_session=False,
    )
    db.commit()
    return key_id


# ─────────────────────────────────────────────────────────────────────────────
# Public read-only endpoints
# ─────────────────────────────────────────────────────────────────────────────

class PublicDetection(BaseModel):
    id: UUID
    damage_type: str
    severity: Optional[int] = None
    latitude: float
    longitude: float
    first_detected: Optional[date] = None
    last_detected: Optional[date] = None
    detection_count: int = 1
    is_fixed: bool = False
    priority_score: float = 0.0

    model_config = {"from_attributes": True}


class PublicDetectionList(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[PublicDetection]


@router.get("/v1/public/detections", response_model=PublicDetectionList)
def public_detections(
    bbox: str | None = Query(None, max_length=120,
                             description="minLon,minLat,maxLon,maxLat"),
    damage_type: str | None = Query(None, max_length=30),
    severity_min: int | None = Query(None, ge=1, le=5),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    q = db.query(Detection)
    if bbox:
        from backend.routes.quality import _parse_bbox
        min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
        q = q.filter(
            Detection.longitude >= min_lon, Detection.longitude <= max_lon,
            Detection.latitude >= min_lat, Detection.latitude <= max_lat,
        )
    if damage_type:
        q = q.filter(Detection.damage_type == damage_type)
    if severity_min is not None:
        q = q.filter(Detection.severity >= severity_min)

    total = q.count()
    items = (
        q.order_by(Detection.priority_score.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return PublicDetectionList(total=total, page=page, page_size=page_size, items=items)


class PublicQualityResponse(BaseModel):
    cell_m: int
    count: int
    cells: List[QualityCell]


@router.get("/v1/public/road-quality", response_model=PublicQualityResponse)
def public_road_quality(
    bbox: str | None = Query(None, max_length=120),
    cell_m: int = Query(120, ge=40, le=1000),
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    cells = compute_grid(db, bbox, cell_m)
    return PublicQualityResponse(cell_m=cell_m, count=len(cells), cells=cells)


class PublicStats(BaseModel):
    total_detections: int
    open_detections: int
    fixed_detections: int
    last_survey_date: Optional[date] = None
    by_damage_type: dict
    by_severity: dict


@router.get("/v1/public/stats", response_model=PublicStats)
def public_stats(db: Session = Depends(get_db), _key=Depends(require_api_key)):
    total = db.query(func.count(Detection.id)).scalar() or 0
    fixed = db.query(func.count(Detection.id)).filter(Detection.is_fixed.is_(True)).scalar() or 0
    last_survey = db.query(func.max(Detection.survey_date)).scalar()
    by_type = dict(
        db.query(Detection.damage_type, func.count(Detection.id))
        .group_by(Detection.damage_type).all()
    )
    by_sev = {
        str(k): v for k, v in (
            db.query(Detection.severity, func.count(Detection.id))
            .filter(Detection.severity.isnot(None))
            .group_by(Detection.severity).all()
        )
    }
    return PublicStats(
        total_detections=int(total),
        open_detections=int(total - fixed),
        fixed_detections=int(fixed),
        last_survey_date=last_survey,
        by_damage_type={k: int(v) for k, v in by_type.items()},
        by_severity=by_sev,
    )
