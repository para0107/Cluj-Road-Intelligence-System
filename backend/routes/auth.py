"""
backend/routes/auth.py

Accounts, sessions, profiles, and role management.

POST /auth/register            create a local account (user or municipality)
POST /auth/login               username-or-email + password → JWT
POST /auth/oauth/google        Google ID token → JWT (only if configured; free)
GET  /auth/config              which optional providers are enabled
GET  /auth/me                  current profile
PATCH /auth/me                 update name / city
PATCH /auth/me/location        record current position (browser geolocation)
GET  /auth/users               list accounts                    [admin]
PATCH /auth/users/{id}/role    change a user's role             [admin]

The starting admin account is seeded at startup (see main.py) from
ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD env, with in-repo defaults.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_auth import User, ROLE_MUNICIPALITY
from backend.auth import (
    hash_password, verify_password, create_token,
    get_current_user, require_admin, GOOGLE_CLIENT_ID,
)
from backend.schemas_auth import (
    RegisterRequest, LoginRequest, LocationUpdate, ProfileUpdate, RoleUpdate,
    UserRead, TokenResponse, AuthConfigResponse, UserListResponse,
)

router = APIRouter()


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(access_token=create_token(user), user=UserRead.model_validate(user))


# ─────────────────────────────────────────────────────────────────────────────
# Registration & login
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if payload.role == ROLE_MUNICIPALITY and not (payload.city and payload.city.strip()):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Municipality accounts must select the city they belong to.",
        )

    clash = (
        db.query(User)
        .filter(or_(
            func.lower(User.username) == payload.username.lower(),
            func.lower(User.email) == payload.email.lower(),
        ))
        .first()
    )
    if clash:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")

    user = User(
        username=payload.username,
        email=payload.email.lower(),
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=payload.role,
        city=payload.city.strip() if payload.city else None,
        auth_provider="local",
        last_login_at=datetime.now(tz=timezone.utc),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(user)


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    ident = payload.identifier.strip().lower()
    user = (
        db.query(User)
        .filter(or_(func.lower(User.username) == ident, func.lower(User.email) == ident))
        .first()
    )
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username/e-mail or password.")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled.")

    user.last_login_at = datetime.now(tz=timezone.utc)
    db.commit()
    db.refresh(user)
    return _token_response(user)


# ─────────────────────────────────────────────────────────────────────────────
# Optional Google OAuth (free — needs only a client id, no billing)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/auth/config", response_model=AuthConfigResponse)
def auth_config():
    return AuthConfigResponse(google_enabled=bool(GOOGLE_CLIENT_ID))


@router.post("/auth/oauth/google", response_model=TokenResponse)
def google_login(body: dict, db: Session = Depends(get_db)):
    """
    Exchange a Google ID token (from Google Identity Services on the frontend)
    for a RIDS JWT. Verification uses Google's public tokeninfo endpoint —
    free, no SDK, no billing. Creates the account on first login.
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Google sign-in is not configured. Create a free OAuth client id at "
            "console.cloud.google.com and set GOOGLE_CLIENT_ID in .env.",
        )
    id_token = (body or {}).get("id_token")
    if not id_token:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing id_token.")

    try:
        resp = httpx.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": id_token},
            timeout=8,
        )
        resp.raise_for_status()
        info = resp.json()
    except httpx.HTTPError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not verify the Google token.")

    if info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token was issued for another app.")
    email = (info.get("email") or "").lower()
    if not email or info.get("email_verified") not in ("true", True):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Google account e-mail not verified.")

    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user:
        base = email.split("@")[0][:32] or "user"
        username = base
        i = 1
        while db.query(User.id).filter(func.lower(User.username) == username.lower()).first():
            i += 1
            username = f"{base}{i}"
        user = User(
            username=username,
            email=email,
            full_name=info.get("name"),
            password_hash=hash_password(os.urandom(24).hex()),   # unusable password
            role="user",
            auth_provider="google",
        )
        db.add(user)

    user.last_login_at = datetime.now(tz=timezone.utc)
    db.commit()
    db.refresh(user)
    return _token_response(user)


# ─────────────────────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/auth/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/auth/me", response_model=UserRead)
def update_me(
    payload: ProfileUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.city is not None:
        user.city = payload.city.strip() or None
    db.commit()
    db.refresh(user)
    return user


@router.patch("/auth/me/location", response_model=UserRead)
def update_location(
    payload: LocationUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the caller's current position (and optionally city)."""
    user.latitude = payload.latitude
    user.longitude = payload.longitude
    user.location_updated_at = datetime.now(tz=timezone.utc)
    if payload.city:
        user.city = payload.city.strip()
    db.commit()
    db.refresh(user)
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Admin: user management
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/auth/users", response_model=UserListResponse)
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    items = db.query(User).order_by(User.created_at.asc()).limit(500).all()
    return UserListResponse(total=len(items), items=items)


@router.patch("/auth/users/{user_id}/role", response_model=UserRead)
def set_role(
    user_id: UUID,
    payload: RoleUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if target.id == admin.id and payload.role != "admin":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot demote yourself.")
    if payload.role == ROLE_MUNICIPALITY and not (payload.city or target.city):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Set a city when assigning the municipality role.",
        )

    target.role = payload.role
    if payload.city:
        target.city = payload.city.strip()
    db.commit()
    db.refresh(target)
    return target
