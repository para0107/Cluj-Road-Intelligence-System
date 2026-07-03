"""
backend/schemas_auth.py

Pydantic v2 schemas for authentication, profiles, roles, and city landmarks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator


# ─────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_.-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=120)
    role: str = Field("user")                       # user | municipality (admin only by promotion)
    city: Optional[str] = Field(None, max_length=80)

    @field_validator("role")
    @classmethod
    def role_must_be_registerable(cls, v: str) -> str:
        if v not in ("user", "municipality"):
            raise ValueError("Only 'user' and 'municipality' can self-register.")
        return v


class LoginRequest(BaseModel):
    # username OR email in the same field — friendlier login form
    identifier: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)


class LocationUpdate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    city: Optional[str] = Field(None, max_length=80)


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=120)
    city: Optional[str] = Field(None, max_length=80)


class RoleUpdate(BaseModel):
    role: str
    city: Optional[str] = Field(None, max_length=80)   # required when role=municipality

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("user", "municipality", "admin"):
            raise ValueError("Role must be user, municipality, or admin.")
        return v


# ─────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────

class UserRead(BaseModel):
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    role: str
    city: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    auth_provider: str
    is_active: bool
    created_at: Optional[datetime]
    last_login_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class AuthConfigResponse(BaseModel):
    """Tells the frontend which optional providers are configured."""
    google_enabled: bool
    apple_enabled: bool = False       # intentionally off: paid Apple program
    apple_disabled_reason: str = (
        "Sign in with Apple requires a paid Apple Developer membership "
        "($99/year) and is excluded to keep the project cost at zero."
    )


class UserListResponse(BaseModel):
    total: int
    items: List[UserRead]


# ─────────────────────────────────────────────
# City landmarks
# ─────────────────────────────────────────────

class LandmarkRead(BaseModel):
    name: str
    kind: Optional[str]
    latitude: float
    longitude: float

    model_config = ConfigDict(from_attributes=True)


class LandmarksResponse(BaseModel):
    city: str
    source: str                      # cache | nominatim | fallback
    items: List[LandmarkRead]
