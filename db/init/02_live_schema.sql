-- db/init/02_live_schema.sql
--
-- Live (Waze-like) mode tables. Runs automatically on a FRESH pgdata volume
-- (after 01_schema.sql). On existing volumes the backend's startup
-- Base.metadata.create_all() creates these tables instead — keep the two
-- definitions in sync with backend/models_live.py.

CREATE TABLE IF NOT EXISTS live_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now(),

    geom              geometry(POINT, 4326) NOT NULL,
    latitude          DOUBLE PRECISION NOT NULL,
    longitude         DOUBLE PRECISION NOT NULL,

    damage_type       VARCHAR(30) NOT NULL,
    max_confidence    DOUBLE PRECISION DEFAULT 0.0,
    severity          SMALLINT,

    status            VARCHAR(12) DEFAULT 'unverified',
    report_count      INTEGER DEFAULT 0,
    reporter_devices  INTEGER DEFAULT 0,
    dispute_devices   INTEGER DEFAULT 0,
    is_active         BOOLEAN DEFAULT TRUE,
    resolved          BOOLEAN DEFAULT FALSE,

    first_reported    TIMESTAMPTZ DEFAULT now(),
    last_reported     TIMESTAMPTZ DEFAULT now(),
    expires_at        TIMESTAMPTZ NOT NULL,

    -- Triage audit trail (operator promoted the event into a detection, or
    -- dismissed it from the inbox).
    promoted_detection_id UUID REFERENCES detections(id) ON DELETE SET NULL,
    dismissed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_live_events_geom   ON live_events USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_live_events_active ON live_events (is_active, expires_at);
CREATE INDEX IF NOT EXISTS idx_live_events_type   ON live_events (damage_type);

CREATE TABLE IF NOT EXISTS live_reports (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  TIMESTAMPTZ DEFAULT now(),

    event_id    UUID NOT NULL REFERENCES live_events(id) ON DELETE CASCADE,
    device_id   VARCHAR(64) NOT NULL,
    -- user_id (REFERENCES users) is added by db/init/06_engagement.sql — the
    -- users table does not exist yet at this point in the init order.
    kind        VARCHAR(10) NOT NULL,          -- sighting | confirm | dispute

    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    confidence  DOUBLE PRECISION,
    severity    SMALLINT,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_reports_event   ON live_reports (event_id);
CREATE INDEX IF NOT EXISTS idx_live_reports_device  ON live_reports (event_id, device_id);
-- /live/stats counts reports from the last hour
CREATE INDEX IF NOT EXISTS idx_live_reports_created ON live_reports (created_at);
