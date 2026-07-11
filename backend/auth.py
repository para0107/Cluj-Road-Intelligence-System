"""
backend/auth.py

Authentication core: password hashing, JWT issue/verify, FastAPI dependencies.

Zero-cost by construction:
- Hashing = stdlib hashlib.pbkdf2_hmac (no bcrypt/passlib dependency drift)
- Tokens  = PyJWT HS256 signed with JWT_SECRET from .env
- Google OAuth is supported as an *optional* provider (free — needs only a
  Google Cloud OAuth client id, no billing); when GOOGLE_CLIENT_ID is unset
  the endpoint reports it as disabled and the UI hides the button.
- Apple Sign-In is intentionally NOT offered: it requires a paid Apple
  Developer membership ($99/year), which violates the project's 0-cost rule.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_auth import User, ROLE_ADMIN, ROLE_MUNICIPALITY

# ── Config ──────────────────────────────────────────────────────────────────
# SECURITY: there is deliberately NO hardcoded fallback secret. If JWT_SECRET
# is unset (or left at the old known dev value), a random per-process secret
# is generated: the app still works, but every restart invalidates all
# sessions — a loud, safe nudge to set a real JWT_SECRET in .env.
_BURNT_SECRETS = {"", "rids-dev-secret-change-me"}
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if JWT_SECRET in _BURNT_SECRETS:
    JWT_SECRET = secrets.token_hex(32)
    from loguru import logger as _logger
    _logger.warning(
        "JWT_SECRET is not set (or is the known default). Generated a random "
        "per-process secret — sessions will NOT survive a restart. "
        "Set a strong JWT_SECRET in .env for persistent sessions."
    )
JWT_ALGO = "HS256"
JWT_TTL_H = float(os.getenv("JWT_TTL_H", "168"))          # 7 days
# OWASP-recommended cost for PBKDF2-HMAC-SHA256 (2023+ guidance). Old hashes
# created at 200k keep verifying — the iteration count is stored per hash.
_PBKDF2_ITERATIONS = 600_000

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")       # optional, free


# ── Password hashing (stdlib only) ──────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


# ── JWT ─────────────────────────────────────────────────────────────────────

def create_token(user: User) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=JWT_TTL_H)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token.")


def _extract_bearer(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return None


# ── FastAPI dependencies ────────────────────────────────────────────────────

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Require a valid token; return the active User row."""
    token = _extract_bearer(request)
    if not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Not authenticated — send Authorization: Bearer <token>.",
        )
    payload = decode_token(token)
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account not found or disabled.")
    return user


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Attach the user when a valid token is present; never raise."""
    token = _extract_bearer(request)
    if not token:
        return None
    try:
        payload = decode_token(token)
    except HTTPException:
        return None
    user = db.query(User).filter(User.id == payload["sub"]).first()
    return user if user and user.is_active else None


def require_roles(*roles: str):
    """Dependency factory: allow only the given roles (admin always included)."""
    allowed = set(roles) | {ROLE_ADMIN}

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires one of the roles: {', '.join(sorted(allowed))}.",
            )
        return user

    return checker


require_admin = require_roles(ROLE_ADMIN)
require_operator = require_roles(ROLE_MUNICIPALITY)   # municipality or admin
