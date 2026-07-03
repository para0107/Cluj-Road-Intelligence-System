"""
backend/models_auth.py

Accounts, roles, and the per-city landmark cache.

Roles
-----
  user          — can view everything, upload surveys, report/vote in Live mode
  municipality  — a city administration account: everything a user can do,
                  plus admin actions (resolve hazards, mark repaired, delete)
                  scoped to the city they selected at registration
  admin         — full control, including user/role management

Passwords are stored as PBKDF2-HMAC-SHA256 (stdlib, 200k iterations, per-user
salt) — no external hashing dependency, nothing to license, nothing to pay.

`city_landmarks` caches free OpenStreetMap/Nominatim lookups of notable places
per city so the map's fly-to menu works for any city with exactly one round of
rate-limited requests (see routes/cities.py). This is UI sugar only — it is
NOT the removed pipeline "enrichment" stage and never touches detections.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import Column, String, Float, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database import Base

ROLE_USER = "user"
ROLE_MUNICIPALITY = "municipality"
ROLE_ADMIN = "admin"
ALL_ROLES = (ROLE_USER, ROLE_MUNICIPALITY, ROLE_ADMIN)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    username = Column(String(40), nullable=False, unique=True, index=True)
    email = Column(String(120), nullable=False, unique=True, index=True)
    full_name = Column(String(120))

    # PBKDF2-HMAC-SHA256; format "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
    password_hash = Column(String(300), nullable=False)

    role = Column(String(16), nullable=False, default=ROLE_USER)
    # Required for municipality accounts; optional profile info for the rest.
    city = Column(String(80))

    # Last known position (browser geolocation / edge agent GPS) — lets the
    # map open on the user's city and scopes municipality views.
    latitude = Column(Float)
    longitude = Column(Float)
    location_updated_at = Column(DateTime(timezone=True))

    auth_provider = Column(String(16), nullable=False, default="local")  # local | google
    is_active = Column(Boolean, nullable=False, default=True)
    last_login_at = Column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role} city={self.city}>"


class CityLandmark(Base):
    __tablename__ = "city_landmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())

    city = Column(String(80), nullable=False)
    name = Column(String(160), nullable=False)
    kind = Column(String(40))          # station | stadium | university | ...
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_city_landmarks_city", "city"),
    )

    def __repr__(self) -> str:
        return f"<CityLandmark {self.city}: {self.name}>"
