-- db/init/04_live_devices.sql
--
-- Paired live-mode sensors (phone drive mode, dashcam edge agents). Runs on a
-- FRESH pgdata volume after 03_auth_schema.sql (it references users.id). On
-- existing volumes the backend's startup Base.metadata.create_all() creates
-- this table instead — keep in sync with backend/models_live.py::LiveDevice.

CREATE TABLE IF NOT EXISTS live_devices (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at            TIMESTAMPTZ DEFAULT now(),

    user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id             VARCHAR(64) UNIQUE,          -- NULL until pair code is claimed
    name                  VARCHAR(80) NOT NULL,
    kind                  VARCHAR(16) NOT NULL DEFAULT 'dashcam',  -- phone|dashcam|browser|simulator

    pair_code             VARCHAR(12) UNIQUE,          -- single-use, short-lived
    pair_code_expires_at  TIMESTAMPTZ,

    is_active             BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at          TIMESTAMPTZ,
    reports_sent          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_live_devices_user ON live_devices (user_id);
