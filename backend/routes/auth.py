"""
backend/routes/auth.py

Accounts, sessions, profiles, e-mail verification, municipality approval,
and role/account management.

Registration flow (accounts are only created AFTER verification):
    POST /auth/register            → pending registration + e-mailed 6-digit code
    POST /auth/verify-email        → code OK:
                                       user          → account created + JWT
                                       municipality  → awaits admin approval
    POST /auth/resend-code         → re-send the confirmation code
    (When SMTP is not configured, the e-mail step is skipped: user accounts
     are created immediately; municipality still requires admin approval.)

Sessions & profile:
    POST  /auth/login              username-or-email + password → JWT
                                   (per-identifier+IP rate limited)
    POST  /auth/oauth/google       Google ID token → JWT (free; optional)
    GET   /auth/config             which optional providers are enabled
    GET   /auth/me                 current profile
    PATCH /auth/me                 update name / city
    PATCH /auth/me/location        record current position
    DELETE /auth/me                delete MY account (password re-typed)

Admin:
    GET    /auth/users                     list accounts
    PATCH  /auth/users/{id}/role           change a role
    PATCH  /auth/users/{id}/active         enable / disable an account
    DELETE /auth/users/{id}                delete an account
    GET    /auth/registrations/pending     municipality approvals queue
    POST   /auth/registrations/{id}/approve
    POST   /auth/registrations/{id}/deny

SECURITY NOTES
    * No default credentials anywhere — the seed admin comes only from env.
    * Login is rate limited in-process: LOGIN_MAX_ATTEMPTS (5) failures per
      LOGIN_WINDOW_S (900 s) per identifier+IP → locked LOGIN_LOCKOUT_S (900 s).
    * Registration race (two identical concurrent sign-ups) is handled by the
      DB unique constraints → clean 409 instead of a 500.
"""

import hmac
import os
import secrets
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_auth import User, PendingRegistration, ROLE_MUNICIPALITY, ROLE_ADMIN
from backend.auth import (
    hash_password, verify_password, create_token,
    get_current_user, require_admin, GOOGLE_CLIENT_ID,
)
from backend.notify import (
    email_enabled, send_welcome_email, send_verification_email,
    send_admin_approval_request, send_approval_result_email,
)
from backend.schemas_auth import (
    RegisterRequest, LoginRequest, LocationUpdate, ProfileUpdate, RoleUpdate,
    UserRead, TokenResponse, AuthConfigResponse, UserListResponse,
    RegisterOutcome, VerifyEmailRequest, ResendCodeRequest, SelfDeleteRequest,
    ActiveUpdate, PendingRegistrationRead, PendingListResponse,
)

router = APIRouter()

_CODE_TTL_MIN = int(os.getenv("VERIFY_CODE_TTL_MIN", "30"))
_RESEND_MIN_GAP_S = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Login rate limiting (in-process, zero-dependency)
# ─────────────────────────────────────────────────────────────────────────────

_LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "5"))
_LOGIN_WINDOW_S = float(os.getenv("LOGIN_WINDOW_S", "900"))
_LOGIN_LOCKOUT_S = float(os.getenv("LOGIN_LOCKOUT_S", "900"))

_rl_lock = threading.Lock()
_rl_failures: dict[str, tuple[int, float]] = {}   # key → (count, window_start)
_rl_locked_until: dict[str, float] = {}


def _client_ip(request: Request) -> str:
    # Behind the bundled Nginx, X-Real-IP carries the true client address.
    return request.headers.get("X-Real-IP") or (request.client.host if request.client else "?")


def _rl_key(identifier: str, request: Request) -> str:
    return f"{identifier.strip().lower()}|{_client_ip(request)}"


def _rl_check(key: str) -> None:
    now = time.time()
    with _rl_lock:
        until = _rl_locked_until.get(key, 0.0)
        if until > now:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Too many failed sign-in attempts. Try again in {int(until - now) + 1} s.",
            )
        if until:
            _rl_locked_until.pop(key, None)


def _rl_fail(key: str) -> None:
    now = time.time()
    with _rl_lock:
        count, start = _rl_failures.get(key, (0, now))
        if now - start > _LOGIN_WINDOW_S:
            count, start = 0, now
        count += 1
        if count >= _LOGIN_MAX_ATTEMPTS:
            _rl_locked_until[key] = now + _LOGIN_LOCKOUT_S
            _rl_failures.pop(key, None)
        else:
            _rl_failures[key] = (count, start)


def _rl_ok(key: str) -> None:
    with _rl_lock:
        _rl_failures.pop(key, None)
        _rl_locked_until.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(access_token=create_token(user), user=UserRead.model_validate(user))


def _ok_outcome(user: User, message: str) -> RegisterOutcome:
    return RegisterOutcome(
        status="ok", message=message, email=user.email,
        access_token=create_token(user), user=UserRead.model_validate(user),
    )


def _admin_emails(db: Session) -> list[str]:
    rows = (
        db.query(User.email)
        .filter(User.role == ROLE_ADMIN, User.is_active.is_(True))
        .all()
    )
    return [r[0] for r in rows if r[0]]


def _account_clash(db: Session, username: str, email: str):
    return (
        db.query(User)
        .filter(or_(
            func.lower(User.username) == username.lower(),
            func.lower(User.email) == email.lower(),
        ))
        .first()
    )


def _create_user_from_pending(db: Session, pending: PendingRegistration) -> User:
    """Turn a verified/approved pending registration into a real account."""
    user = User(
        username=pending.username,
        email=pending.email.lower(),
        full_name=pending.full_name,
        password_hash=pending.password_hash,
        role=pending.role,
        city=pending.city,
        auth_provider="local",
        last_login_at=_now(),
    )
    db.delete(pending)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")
    db.refresh(user)
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Registration — verify the e-mail BEFORE the account exists
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=RegisterOutcome, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if payload.role == ROLE_MUNICIPALITY and not (payload.city and payload.city.strip()):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Municipality accounts must select the city they belong to.",
        )

    if _account_clash(db, payload.username, payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")

    # A previous half-finished registration for the same identity?
    pending = (
        db.query(PendingRegistration)
        .filter(or_(
            func.lower(PendingRegistration.email) == payload.email.lower(),
            func.lower(PendingRegistration.username) == payload.username.lower(),
        ))
        .first()
    )
    if pending:
        if pending.status == "awaiting_approval":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "This registration is already awaiting admin approval.",
            )
        db.delete(pending)   # awaiting_email → replace with a fresh attempt
        db.flush()

    city = payload.city.strip() if payload.city else None

    # ── No SMTP configured (dev): skip the e-mail step gracefully ───────────
    if not email_enabled():
        if payload.role == ROLE_MUNICIPALITY:
            pr = PendingRegistration(
                username=payload.username, email=payload.email.lower(),
                full_name=payload.full_name, password_hash=hash_password(payload.password),
                role=payload.role, city=city,
                email_verified=True, status="awaiting_approval",
            )
            db.add(pr)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")
            send_admin_approval_request(_admin_emails(db), pr.username, pr.email, pr.city)
            return RegisterOutcome(
                status="awaiting_approval", email=pr.email,
                message="Registration received. A platform admin must approve "
                        "municipality accounts before they are created.",
            )
        user = User(
            username=payload.username, email=payload.email.lower(),
            full_name=payload.full_name, password_hash=hash_password(payload.password),
            role=payload.role, city=city, auth_provider="local", last_login_at=_now(),
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")
        db.refresh(user)
        send_welcome_email(user.email, user.username, user.role, user.city)
        return _ok_outcome(user, "Account created (e-mail verification is off on this server).")

    # ── Normal path: e-mail a confirmation code; nothing is created yet ─────
    code = _new_code()
    pr = PendingRegistration(
        username=payload.username, email=payload.email.lower(),
        full_name=payload.full_name, password_hash=hash_password(payload.password),
        role=payload.role, city=city,
        email_code=code, code_expires_at=_now() + timedelta(minutes=_CODE_TTL_MIN),
        status="awaiting_email",
    )
    db.add(pr)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or e-mail already registered.")
    send_verification_email(pr.email, pr.username, code)
    return RegisterOutcome(
        status="verify_email", email=pr.email,
        message=f"We sent a confirmation code to {pr.email}. "
                f"Enter it within {_CODE_TTL_MIN} minutes to finish.",
    )


@router.post("/auth/verify-email", response_model=RegisterOutcome)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    pending = (
        db.query(PendingRegistration)
        .filter(
            func.lower(PendingRegistration.email) == payload.email.lower(),
            PendingRegistration.status == "awaiting_email",
        )
        .first()
    )
    if not pending:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No registration awaiting verification for this e-mail. Register first.",
        )
    if not pending.code_expires_at or pending.code_expires_at < _now():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The confirmation code expired — request a new one.",
        )
    if not (pending.email_code and hmac.compare_digest(pending.email_code, payload.code.strip())):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Wrong confirmation code.")

    if pending.role == ROLE_MUNICIPALITY:
        pending.email_verified = True
        pending.email_code = None
        pending.code_expires_at = None
        pending.status = "awaiting_approval"
        db.commit()
        send_admin_approval_request(_admin_emails(db), pending.username, pending.email, pending.city)
        return RegisterOutcome(
            status="awaiting_approval", email=pending.email,
            message="E-mail confirmed. A platform admin must now approve your "
                    "municipality account — you will be notified by e-mail.",
        )

    user = _create_user_from_pending(db, pending)
    send_welcome_email(user.email, user.username, user.role, user.city)
    return _ok_outcome(user, "E-mail confirmed — welcome aboard!")


_resend_lock = threading.Lock()
_resend_at: dict[str, float] = {}


@router.post("/auth/resend-code", response_model=RegisterOutcome)
def resend_code(payload: ResendCodeRequest, db: Session = Depends(get_db)):
    email = payload.email.lower()
    now = time.time()
    with _resend_lock:
        if now - _resend_at.get(email, 0.0) < _RESEND_MIN_GAP_S:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Please wait a minute before requesting another code.",
            )
        _resend_at[email] = now

    pending = (
        db.query(PendingRegistration)
        .filter(
            func.lower(PendingRegistration.email) == email,
            PendingRegistration.status == "awaiting_email",
        )
        .first()
    )
    if not pending:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No registration awaiting verification.")

    code = _new_code()
    pending.email_code = code
    pending.code_expires_at = _now() + timedelta(minutes=_CODE_TTL_MIN)
    db.commit()
    send_verification_email(pending.email, pending.username, code)
    return RegisterOutcome(
        status="verify_email", email=pending.email,
        message=f"A new confirmation code was sent to {pending.email}.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Login (rate limited)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    ident = payload.identifier.strip().lower()
    key = _rl_key(ident, request)
    _rl_check(key)

    user = (
        db.query(User)
        .filter(or_(func.lower(User.username) == ident, func.lower(User.email) == ident))
        .first()
    )
    if not user or not verify_password(payload.password, user.password_hash):
        # A half-finished registration? Give a helpful (non-counted) answer.
        pending = (
            db.query(PendingRegistration)
            .filter(or_(
                func.lower(PendingRegistration.username) == ident,
                func.lower(PendingRegistration.email) == ident,
            ))
            .first()
        )
        if pending and verify_password(payload.password, pending.password_hash):
            if pending.status == "awaiting_approval":
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    "Your municipality registration is awaiting admin approval.",
                )
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Confirm your e-mail first — check your inbox for the code.",
            )
        _rl_fail(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username/e-mail or password.")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled.")

    _rl_ok(key)
    user.last_login_at = _now()
    db.commit()
    db.refresh(user)
    return _token_response(user)


# ─────────────────────────────────────────────────────────────────────────────
# Optional Google OAuth (free — needs only a client id, no billing)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/auth/config", response_model=AuthConfigResponse)
def auth_config():
    return AuthConfigResponse(
        google_enabled=bool(GOOGLE_CLIENT_ID),
        google_client_id=GOOGLE_CLIENT_ID,
    )


@router.post("/auth/oauth/google", response_model=TokenResponse)
def google_login(body: dict, db: Session = Depends(get_db)):
    """
    Exchange a Google ID token (from Google Identity Services on the frontend)
    for a RIDS JWT. Verification uses Google's public tokeninfo endpoint —
    free, no SDK, no billing. Creates the account on first login (Google has
    already verified the e-mail, so the code step is skipped).
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

    user.last_login_at = _now()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Account collision — try again.")
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
    user.location_updated_at = _now()
    if payload.city:
        user.city = payload.city.strip()
    db.commit()
    db.refresh(user)
    return user


@router.delete("/auth/me")
def delete_me(
    payload: SelfDeleteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete MY account (any role). Local accounts must re-type their password;
    OAuth accounts (no usable password) just confirm in the UI. The last
    active admin cannot delete itself — promote someone else first.
    """
    if user.auth_provider == "local":
        if not payload.password or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Password incorrect.")

    if user.role == ROLE_ADMIN:
        other_admins = (
            db.query(func.count(User.id))
            .filter(User.role == ROLE_ADMIN, User.is_active.is_(True), User.id != user.id)
            .scalar() or 0
        )
        if other_admins == 0:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "You are the last active admin — promote another admin before "
                "deleting this account.",
            )

    db.delete(user)   # live_devices rows cascade via FK
    db.commit()
    return {"deleted": True}


# ─────────────────────────────────────────────────────────────────────────────
# Admin: user & registration management
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


@router.patch("/auth/users/{user_id}/active", response_model=UserRead)
def set_active(
    user_id: UUID,
    payload: ActiveUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Enable / disable an account without deleting it."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if target.id == admin.id and not payload.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot disable your own account.")

    target.is_active = payload.is_active
    db.commit()
    db.refresh(target)
    return target


@router.delete("/auth/users/{user_id}")
def admin_delete_user(
    user_id: UUID,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if target.id == admin.id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Delete your own account from the profile menu, not the admin panel.",
        )
    db.delete(target)
    db.commit()
    return {"deleted": True, "username": target.username}


@router.get("/auth/registrations/pending", response_model=PendingListResponse)
def pending_registrations(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Every half-finished registration; municipality ones can be approved."""
    items = (
        db.query(PendingRegistration)
        .order_by(PendingRegistration.created_at.asc())
        .limit(500)
        .all()
    )
    return PendingListResponse(total=len(items), items=items)


@router.post("/auth/registrations/{pending_id}/approve", response_model=UserRead)
def approve_registration(
    pending_id: UUID,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    pending = db.query(PendingRegistration).filter(PendingRegistration.id == pending_id).first()
    if not pending:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pending registration not found.")
    if pending.status != "awaiting_approval":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This registration has not confirmed its e-mail yet.",
        )
    user = _create_user_from_pending(db, pending)
    send_approval_result_email(user.email, user.username, approved=True)
    return user


@router.post("/auth/registrations/{pending_id}/deny")
def deny_registration(
    pending_id: UUID,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    pending = db.query(PendingRegistration).filter(PendingRegistration.id == pending_id).first()
    if not pending:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pending registration not found.")
    email, username = pending.email, pending.username
    db.delete(pending)
    db.commit()
    send_approval_result_email(email, username, approved=False)
    return {"denied": True, "username": username}
