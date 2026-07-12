"""
backend/models_apikeys.py

Developer API keys for the read-only public API (/api/v1/public/*).

Key material is never stored: a key looks like "rdds_<40 hex chars>", the row
keeps only its SHA-256 hash (lookup) and the first 12 characters (display).
The plaintext is returned exactly once, at creation.

Created by Base.metadata.create_all at startup; fresh Docker volumes get the
table from db/init/08_api_keys.sql — keep both in sync.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(80), nullable=False)
    prefix = Column(String(16), nullable=False)          # "rdds_" + first chars, for display
    key_hash = Column(String(64), nullable=False, unique=True)   # sha256 hex of the full key

    is_active = Column(Boolean, nullable=False, default=True)
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, nullable=False, default=0)
    rate_limit_per_min = Column(Integer, nullable=False, default=60)

    __table_args__ = (
        Index("idx_api_keys_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ApiKey {self.prefix}… user={self.user_id} active={self.is_active}>"
