-- db/init/06_engagement.sql
--
-- Citizen engagement: points ledger, per-user stats snapshot, badges, and
-- in-app notifications. Runs on a FRESH pgdata volume after 03_auth_schema.sql
-- (users exists by now). On existing volumes the backend's startup
-- Base.metadata.create_all() creates these tables instead — keep in sync with
-- backend/models_engagement.py.

CREATE TABLE IF NOT EXISTS user_points (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  TIMESTAMPTZ DEFAULT now(),

    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    points      INTEGER NOT NULL,
    reason      VARCHAR(40) NOT NULL,   -- event_confirmed | event_verified | ...
    ref_id      UUID,                   -- usually the live_event id

    CONSTRAINT uq_user_points_award UNIQUE (user_id, reason, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_user_points_user ON user_points (user_id, created_at);

CREATE TABLE IF NOT EXISTS user_stats (
    user_id             UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    updated_at          TIMESTAMPTZ DEFAULT now(),

    points_total        INTEGER NOT NULL DEFAULT 0,
    reports_total       INTEGER NOT NULL DEFAULT 0,
    confirmed_total     INTEGER NOT NULL DEFAULT 0,
    verified_total      INTEGER NOT NULL DEFAULT 0,
    fixed_total         INTEGER NOT NULL DEFAULT 0,

    current_streak_days INTEGER NOT NULL DEFAULT 0,
    best_streak_days    INTEGER NOT NULL DEFAULT 0,
    last_report_date    DATE,

    city                VARCHAR(80)
);

CREATE INDEX IF NOT EXISTS idx_user_stats_points ON user_stats (points_total);
CREATE INDEX IF NOT EXISTS idx_user_stats_city   ON user_stats (city);

CREATE TABLE IF NOT EXISTS user_badges (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    awarded_at  TIMESTAMPTZ DEFAULT now(),

    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    badge_key   VARCHAR(40) NOT NULL,   -- keys defined in backend/gamification.py

    CONSTRAINT uq_user_badges_once UNIQUE (user_id, badge_key)
);

CREATE INDEX IF NOT EXISTS idx_user_badges_user ON user_badges (user_id);

CREATE TABLE IF NOT EXISTS notifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  TIMESTAMPTZ DEFAULT now(),

    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind        VARCHAR(30) NOT NULL,   -- points | badge | fixed | promoted | info
    title       VARCHAR(120) NOT NULL,
    body        VARCHAR(300),
    link        VARCHAR(120),           -- SPA route, e.g. /impact
    ref_id      UUID,
    read_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications (user_id, created_at);

-- live_reports gained an account linkage for gamification. The column lives
-- here (not in 02_live_schema.sql) because users does not exist until
-- 03_auth_schema.sql has run.
ALTER TABLE live_reports ADD COLUMN IF NOT EXISTS user_id UUID
    REFERENCES users(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_live_reports_user ON live_reports (user_id, created_at);
