-- db/init/08_api_keys.sql
--
-- Developer API keys for the read-only public API (/api/v1/public/*).
-- Only a SHA-256 hash of the key is stored; the plaintext is shown once at
-- creation. Runs on a FRESH pgdata volume; existing volumes get the table
-- from Base.metadata.create_all() — keep in sync with backend/models_apikeys.py.

CREATE TABLE IF NOT EXISTS api_keys (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ DEFAULT now(),

    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name                VARCHAR(80) NOT NULL,
    prefix              VARCHAR(16) NOT NULL,
    key_hash            VARCHAR(64) NOT NULL UNIQUE,

    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at        TIMESTAMPTZ,
    usage_count         INTEGER NOT NULL DEFAULT 0,
    rate_limit_per_min  INTEGER NOT NULL DEFAULT 60
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys (user_id);
