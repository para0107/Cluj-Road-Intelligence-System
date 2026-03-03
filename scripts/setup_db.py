"""
scripts/setup_db.py

Creates the PostgreSQL database schema with PostGIS extension.
Run this ONCE after the Docker container is up:
    python scripts/setup_db.py

It is safe to run multiple times — all CREATE statements use
IF NOT EXISTS so existing data is never dropped.
"""

import os
import sys

# Allow running from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME     = os.getenv("DB_NAME")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def enable_postgis(conn):
    """Enable the PostGIS extension if not already enabled."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")  # for gen_random_uuid()
    conn.commit()
    logger.info("PostGIS + pgcrypto extensions enabled.")


def create_detections_table(conn):
    """
    Main table that stores every road damage detection.

    One row = one unique physical damage location.
    When the same pothole is detected again in a future survey,
    the existing row is UPDATED (detection_count++, last_detected, etc.)
    rather than inserting a duplicate.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS detections (

        -- Identity
        id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),

        -- Spatial location (WGS84 lat/lon point)
        -- GIST index created separately below for spatial queries
        geom                    GEOMETRY(POINT, 4326) NOT NULL,

        -- ── Model outputs ─────────────────────────────────────────────────
        damage_type             VARCHAR(30) NOT NULL,
        -- pothole | longitudinal_crack | transverse_crack
        -- | alligator_crack | patch_deterioration

        confidence              FLOAT NOT NULL,
        -- RT-DETR detection confidence [0, 1]

        surface_area_cm2        FLOAT,
        -- computed from SAM mask pixel count × calibration factor

        severity                SMALLINT,
        -- S1 (minor) → S5 (critical), assigned by XGBoost

        severity_confidence     FLOAT,
        -- XGBoost prediction confidence [0, 1]

        -- ── Depth estimation ──────────────────────────────────────────────
        depth_estimate_cm       FLOAT,
        -- averaged output of EfficientNet-B3 + Monodepth2

        depth_confidence        FLOAT,
        -- internal confidence flag from depth estimator [0, 1]

        -- ── Mask geometry features ────────────────────────────────────────
        edge_sharpness          FLOAT,
        -- mean Sobel gradient magnitude at mask boundary

        interior_contrast       FLOAT,
        -- mean pixel difference inside vs. immediately outside mask

        mask_compactness        FLOAT,
        -- 4π × area / perimeter²; circle = 1.0, irregular < 1.0

        -- ── Lighting / shadow ─────────────────────────────────────────────
        shadow_geometry_score   FLOAT,
        -- composite score derived from sun angle + shadow shape analysis

        lighting_condition      VARCHAR(15),
        -- daylight | overcast | low_light

        sun_altitude_deg        FLOAT,
        -- solar altitude angle at detection time (degrees above horizon)

        sun_azimuth_deg         FLOAT,
        -- solar azimuth angle at detection time (degrees from north)

        -- ── Location context (from OSM + Nominatim) ───────────────────────
        street_name             VARCHAR(150),
        -- human-readable street name from Nominatim reverse geocode

        road_importance         SMALLINT,
        -- 3 = primary, 2 = secondary, 1 = residential/other
        -- derived from OSM highway tag

        infra_proximity_m       FLOAT,
        -- distance in meters to nearest school / hospital / bus stop / intersection

        nearest_infra_type      VARCHAR(30),
        -- school | hospital | bus_stop | intersection

        -- ── Weather at detection time (Open-Meteo) ────────────────────────
        weather                 JSONB,
        -- {temperature_c, precipitation_mm, weathercode, windspeed_kmh}

        -- ── Spatial context ───────────────────────────────────────────────
        surrounding_density     INTEGER DEFAULT 0,
        -- number of other detections within SURROUNDING_DENSITY_RADIUS_METERS

        -- ── Temporal tracking ─────────────────────────────────────────────
        first_detected          DATE NOT NULL,
        -- date when this damage was first recorded

        last_detected           DATE NOT NULL,
        -- date of most recent survey that confirmed this damage

        detection_count         INTEGER DEFAULT 1,
        -- how many daily surveys have detected this damage

        deterioration_rate      FLOAT DEFAULT 0.0,
        -- average severity increase per day since first_detected
        -- positive = getting worse, 0 = stable, negative = repaired (edge case)

        -- ── Priority ──────────────────────────────────────────────────────
        priority_score          FLOAT,
        -- severity_score × road_weight × infra_weight × log(detection_count + 1)
        -- higher = more urgent to repair

        -- ── Audit ─────────────────────────────────────────────────────────
        created_at              TIMESTAMPTZ DEFAULT NOW(),
        updated_at              TIMESTAMPTZ DEFAULT NOW()
    );
    """

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Table 'detections' created (or already exists).")


def create_survey_log_table(conn):
    """
    Tracks each daily pipeline run.
    Useful for debugging, auditing, and showing city hall
    how frequently surveys are conducted.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS survey_log (
        id                  SERIAL PRIMARY KEY,
        survey_date         DATE NOT NULL UNIQUE,
        footage_file        VARCHAR(255),
        gps_file            VARCHAR(255),
        frames_extracted    INTEGER,
        detections_new      INTEGER,       -- brand-new damage locations found
        detections_updated  INTEGER,       -- existing records updated
        pipeline_started_at TIMESTAMPTZ,
        pipeline_ended_at   TIMESTAMPTZ,
        status              VARCHAR(20),   -- running | completed | failed
        error_message       TEXT,
        created_at          TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Table 'survey_log' created (or already exists).")


def create_indexes(conn):
    """
    Create all indexes needed for dashboard performance.
    GIST index on geom is critical — without it PostGIS
    spatial queries scan the entire table.
    """
    indexes = [
        # Spatial index — mandatory for ST_DWithin queries
        """
        CREATE INDEX IF NOT EXISTS idx_detections_geom
        ON detections USING GIST(geom);
        """,

        # Severity filter — dashboard filter panel
        """
        CREATE INDEX IF NOT EXISTS idx_detections_severity
        ON detections(severity);
        """,

        # Damage type filter
        """
        CREATE INDEX IF NOT EXISTS idx_detections_damage_type
        ON detections(damage_type);
        """,

        # Street name filter + search
        """
        CREATE INDEX IF NOT EXISTS idx_detections_street_name
        ON detections(street_name);
        """,

        # Date range queries
        """
        CREATE INDEX IF NOT EXISTS idx_detections_last_detected
        ON detections(last_detected DESC);
        """,

        # Priority list — sorted by priority_score descending
        """
        CREATE INDEX IF NOT EXISTS idx_detections_priority
        ON detections(priority_score DESC NULLS LAST);
        """,

        # Road importance filter
        """
        CREATE INDEX IF NOT EXISTS idx_detections_road_importance
        ON detections(road_importance);
        """,
    ]

    with conn.cursor() as cur:
        for idx_sql in indexes:
            cur.execute(idx_sql)
    conn.commit()
    logger.info("All indexes created.")


def create_update_trigger(conn):
    """
    Automatically updates the 'updated_at' timestamp
    whenever a detection row is modified.
    Also recomputes priority_score on every update.
    """
    trigger_function_sql = """
    CREATE OR REPLACE FUNCTION update_detection_metadata()
    RETURNS TRIGGER AS $$
    BEGIN
        -- Update timestamp
        NEW.updated_at = NOW();

        -- Recompute priority score
        -- Formula: severity_score × road_weight × infra_weight × log(detection_count + 1)
        -- severity_score: S1=1, S2=2, S3=3, S4=4, S5=5
        -- road_weight:    primary=3, secondary=2, residential=1
        -- infra_weight:   within 50m of school/hospital/bus stop/intersection = 2, else = 1
        NEW.priority_score = (
            COALESCE(NEW.severity, 1)::FLOAT
            *
            COALESCE(NEW.road_importance, 1)::FLOAT
            *
            CASE
                WHEN COALESCE(NEW.infra_proximity_m, 9999) <= 50.0 THEN 2.0
                ELSE 1.0
            END
            *
            LN(COALESCE(NEW.detection_count, 1) + 1)
        );

        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    DROP TRIGGER IF EXISTS trg_update_detection_metadata ON detections;
    CREATE TRIGGER trg_update_detection_metadata
    BEFORE INSERT OR UPDATE ON detections
    FOR EACH ROW
    EXECUTE FUNCTION update_detection_metadata();
    """

    with conn.cursor() as cur:
        cur.execute(trigger_function_sql)
        cur.execute(trigger_sql)
    conn.commit()
    logger.info("Update trigger 'trg_update_detection_metadata' created.")


def verify_setup(conn):
    """
    Run a quick sanity check — confirms tables, indexes,
    and PostGIS are all working correctly.
    """
    checks = {
        "PostGIS version":
            "SELECT PostGIS_Version();",
        "detections table exists":
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'detections';",
        "survey_log table exists":
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'survey_log';",
        "spatial index exists":
            "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_detections_geom';",
        "trigger exists":
            "SELECT COUNT(*) FROM pg_trigger WHERE tgname = 'trg_update_detection_metadata';",
    }

    logger.info("── Verification ──────────────────────────────")
    with conn.cursor() as cur:
        for label, query in checks.items():
            cur.execute(query)
            result = cur.fetchone()[0]
            logger.info(f"  {label}: {result}")
    logger.info("──────────────────────────────────────────────")


def main():
    logger.info(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} ...")

    try:
        conn = get_connection()
        logger.info("Connection successful.")
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        logger.error("Make sure Docker is running: docker compose up -d")
        sys.exit(1)

    try:
        enable_postgis(conn)
        create_detections_table(conn)
        create_survey_log_table(conn)
        create_indexes(conn)
        create_update_trigger(conn)
        verify_setup(conn)
        logger.success("Database setup complete. You are ready to run the pipeline.")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()