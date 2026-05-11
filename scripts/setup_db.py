"""
scripts/setup_db.py

Creates the PostgreSQL database schema with PostGIS extension.

This version reflects an intentionally simplified detections table:
the following columns are NOT part of the schema and will be dropped
(if they exist from older runs):

  street_name, road_importance, infra_proximity_m, nearest_infra_type,
  weather, shadow_geometry_score, sun_altitude_deg, sun_azimuth_deg

Run after docker-compose up -d:
    python scripts/setup_db.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DB_NAME = os.getenv("POSTGRES_DB", "cluj_monitor")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")


def get_connection(dbname: str = "postgres"):
    return psycopg2.connect(
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def create_database() -> None:
    conn = get_connection(dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{DB_NAME}";')
        logger.info("Database '%s' created.", DB_NAME)
    else:
        logger.info("Database '%s' already exists - skipping.", DB_NAME)

    cur.close()
    conn.close()


def _drop_removed_columns(cur) -> None:
    """
    Migration helper: if older schema variants exist, enforce the simplified schema
    by dropping removed columns. Safe to run multiple times.
    """
    cur.execute(
        """
        ALTER TABLE IF EXISTS detections
          DROP COLUMN IF EXISTS street_name,
          DROP COLUMN IF EXISTS road_importance,
          DROP COLUMN IF EXISTS infra_proximity_m,
          DROP COLUMN IF EXISTS nearest_infra_type,
          DROP COLUMN IF EXISTS weather,
          DROP COLUMN IF EXISTS shadow_geometry_score,
          DROP COLUMN IF EXISTS sun_altitude_deg,
          DROP COLUMN IF EXISTS sun_azimuth_deg;
        """
    )
    logger.info("Dropped deprecated columns (if present).")


def setup_schema() -> None:
    """
    Everything in a single connection / transaction so there are no
    commit-ordering issues between table creation and index creation.
    """
    conn = get_connection(dbname=DB_NAME)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    logger.info("Extensions enabled.")

    # detections table (simplified schema)
    cur.execute(
        """
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
            survey_video_file   VARCHAR(255)
        );
        """
    )
    logger.info("Table 'detections' ready.")

    # Enforce simplified schema even if table already existed
    _drop_removed_columns(cur)

    # survey_log table
    cur.execute(
        """
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
        """
    )
    logger.info("Table 'survey_log' ready.")

    # Indexes (do NOT reference dropped columns)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_geom ON detections USING GIST(geom);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_severity ON detections(severity);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_damage_type ON detections(damage_type);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_survey_date ON detections(survey_date);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_priority ON detections(priority_score DESC);"
    )
    logger.info("Indexes ready.")

    # Auto-update trigger for updated_at
    cur.execute(
        """
        CREATE OR REPLACE FUNCTION fn_update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    cur.execute("DROP TRIGGER IF EXISTS trg_update_detection_metadata ON detections;")
    cur.execute(
        """
        CREATE TRIGGER trg_update_detection_metadata
        BEFORE UPDATE ON detections
        FOR EACH ROW EXECUTE FUNCTION fn_update_updated_at();
        """
    )
    logger.info("Trigger ready.")

    cur.close()
    conn.close()


def verify() -> None:
    conn = get_connection(dbname=DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT PostGIS_Version();")
    logger.info("PostGIS version : %s", cur.fetchone()[0])

    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name='detections';
        """
    )
    logger.info("detections table: %s", "OK" if cur.fetchone()[0] else "MISSING")

    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name='survey_log';
        """
    )
    logger.info("survey_log table: %s", "OK" if cur.fetchone()[0] else "MISSING")

    cur.execute(
        """
        SELECT COUNT(*)
        FROM pg_indexes
        WHERE tablename='detections' AND indexname='idx_detections_geom';
        """
    )
    logger.info("spatial index   : %s", "OK" if cur.fetchone()[0] else "MISSING")

    cur.execute("SELECT COUNT(*) FROM pg_indexes WHERE tablename='detections';")
    logger.info("total indexes   : %d", cur.fetchone()[0])

    # Confirm removed columns are absent
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='detections'
          AND column_name IN (
            'street_name',
            'road_importance',
            'infra_proximity_m',
            'nearest_infra_type',
            'weather',
            'shadow_geometry_score',
            'sun_altitude_deg',
            'sun_azimuth_deg'
          )
        ORDER BY column_name;
        """
    )
    leftover = [r[0] for r in cur.fetchall()]
    if leftover:
        logger.warning("Deprecated columns still present: %s", leftover)
    else:
        logger.success("Deprecated columns are not present (schema is clean).")

    cur.close()
    conn.close()


def main() -> None:
    logger.info("Target: %s:%s/%s  user=%s", DB_HOST, DB_PORT, DB_NAME, DB_USER)

    try:
        get_connection(dbname="postgres").close()
        logger.info("PostgreSQL reachable.")
    except Exception as e:
        logger.error("Cannot connect: %s", e)
        logger.error("Make sure Docker is running: docker-compose up -d")
        sys.exit(1)

    create_database()
    setup_schema()
    verify()
    logger.success("Setup complete — ready to run the backend/pipeline.")


if __name__ == "__main__":
    main()