-- db/init/03_auth_schema.sql
--
-- Accounts, roles, and the per-city landmark cache. Runs automatically on a
-- FRESH pgdata volume (after 01/02). On existing volumes the backend's
-- startup Base.metadata.create_all() creates these tables instead — keep in
-- sync with backend/models_auth.py. The seed admin account is created by the
-- backend at startup (main.py::_seed_admin), not here, so the password hash
-- never lives in SQL.

CREATE TABLE IF NOT EXISTS users (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ DEFAULT now(),

    username             VARCHAR(40)  NOT NULL UNIQUE,
    email                VARCHAR(120) NOT NULL UNIQUE,
    full_name            VARCHAR(120),
    password_hash        VARCHAR(300) NOT NULL,

    role                 VARCHAR(16)  NOT NULL DEFAULT 'user',   -- user | municipality | admin
    city                 VARCHAR(80),

    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    location_updated_at  TIMESTAMPTZ,

    auth_provider        VARCHAR(16)  NOT NULL DEFAULT 'local',  -- local | google
    is_active            BOOLEAN      NOT NULL DEFAULT TRUE,
    last_login_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
CREATE INDEX IF NOT EXISTS ix_users_email    ON users (email);

CREATE TABLE IF NOT EXISTS city_landmarks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fetched_at  TIMESTAMPTZ DEFAULT now(),

    city        VARCHAR(80)  NOT NULL,
    name        VARCHAR(160) NOT NULL,
    kind        VARCHAR(40),
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_city_landmarks_city ON city_landmarks (city);
