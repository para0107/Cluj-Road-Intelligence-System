-- ─────────────────────────────────────────────────────────────────────────────
-- db/init/01_schema.sql
--
-- Automatic database schema bootstrap for RIDS.
--
-- The postgis/postgis image runs every *.sql / *.sh file in
-- /docker-entrypoint-initdb.d/ exactly ONCE, on the FIRST start of the `db`
-- container (i.e. when the `pgdata` volume is empty). The default database
-- named by POSTGRES_DB has already been created by the entrypoint, and this
-- script runs *inside* that database — so there is no CREATE DATABASE here.
--
-- This mirrors scripts/setup_db.py (setup_schema). Keeping both is fine: the
-- script is idempotent (IF NOT EXISTS everywhere), so running setup_db.py later
-- against the same database is a no-op.
--
-- If you ever change the schema, change BOTH this file and scripts/setup_db.py.
--
-- NOTE: this only runs on a fresh volume. To re-run it, drop the volume first:
--     docker compose down -v   &&   docker compose up -d --build
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── detections — one row per de-duplicated damage instance ───────────────────
CREATE TABLE IF NOT EXISTS detections (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    geom                GEOMETRY(POINT, 4326),
    latitude            DOUBLE PRECISION NOT NULL,
    longitude           DOUBLE PRECISION NOT NULL,

    damage_type         VARCHAR(30) NOT NULL,
    confidence          FLOAT NOT NULL,
    frame_path          TEXT,
    crop_path           TEXT,

    surface_area_cm2    FLOAT,
    edge_sharpness      FLOAT,
    interior_contrast   FLOAT,
    mask_compactness    FLOAT,

    depth_estimate_cm   FLOAT,
    depth_confidence    FLOAT,
    lighting_condition  VARCHAR(15),

    severity            SMALLINT,
    severity_confidence FLOAT,

    surrounding_density INTEGER DEFAULT 0,

    first_detected      DATE NOT NULL,
    last_detected       DATE NOT NULL,
    detection_count     INTEGER DEFAULT 1,
    deterioration_rate  FLOAT DEFAULT 0.0,

    priority_score      FLOAT DEFAULT 0.0,

    survey_date         DATE NOT NULL,
    survey_video_file   VARCHAR(255),

    is_fixed            BOOLEAN DEFAULT FALSE,
    fixed_at            TIMESTAMPTZ
);

-- ── survey_log — one row per survey_date (unique) ────────────────────────────
CREATE TABLE IF NOT EXISTS survey_log (
    id                  SERIAL PRIMARY KEY,
    survey_date         DATE NOT NULL UNIQUE,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    status              VARCHAR(20) DEFAULT 'pending',
    frames_processed    INTEGER DEFAULT 0,
    detections_found    INTEGER DEFAULT 0,
    new_detections      INTEGER DEFAULT 0,
    updated_detections  INTEGER DEFAULT 0,
    error_message       TEXT,
    video_files         JSONB
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_detections_geom        ON detections USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_detections_severity    ON detections(severity);
CREATE INDEX IF NOT EXISTS idx_detections_damage_type ON detections(damage_type);
CREATE INDEX IF NOT EXISTS idx_detections_survey_date ON detections(survey_date);
CREATE INDEX IF NOT EXISTS idx_detections_priority    ON detections(priority_score DESC);

-- ── Auto-update trigger for updated_at ───────────────────────────────────────
CREATE OR REPLACE FUNCTION fn_update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_detection_metadata ON detections;
CREATE TRIGGER trg_update_detection_metadata
BEFORE UPDATE ON detections
FOR EACH ROW EXECUTE FUNCTION fn_update_updated_at();
