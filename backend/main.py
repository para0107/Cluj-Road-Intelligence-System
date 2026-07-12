"""
backend/main.py

FastAPI application entry point.
Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
Or from project root:
    python backend/main.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger

from backend.database import check_connection, engine, Base
from backend.middleware import SecurityHeadersMiddleware
from backend.routes import (
    detections, stats, heatmap, priority, export, ingest, live, auth, cities,
    engagement, workorders, analytics, media, quality, public_api, contact,
)
from backend.live_manager import manager as live_ws_manager
import backend.models_live        # noqa: F401 — register live_events/live_reports on Base.metadata
import backend.models_auth        # noqa: F401 — register users/city_landmarks on Base.metadata
import backend.models_engagement  # noqa: F401 — register points/badges/notifications on Base.metadata
import backend.models_work        # noqa: F401 — register work_orders on Base.metadata
import backend.models_apikeys     # noqa: F401 — register api_keys on Base.metadata

load_dotenv()

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="RDDS — Road Degradation Detection System API",
    description=(
        "Road damage detection, classification, and prioritization system "
        "for Cluj-Napoca, Romania."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# CORS — allow the local frontend to call the API
# ─────────────────────────────────────────────

# Comma-separated list in .env, e.g. CORS_ORIGINS=https://rids.vercel.app
# Default "*" keeps local/dev friction-free; set real origins when deploying.
# Browsers reject the "*" + credentials pairing, and the app authenticates with
# Bearer headers (not cookies), so credentials are only enabled for real origins.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)

# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(detections.router, prefix="/api", tags=["Detections"])
app.include_router(stats.router,      prefix="/api", tags=["Stats"])
app.include_router(heatmap.router,    prefix="/api", tags=["Heatmap"])
app.include_router(priority.router,   prefix="/api", tags=["Priority"])
app.include_router(export.router,     prefix="/api", tags=["Export"])
app.include_router(ingest.router,     prefix="/api", tags=["Ingest"])
app.include_router(live.router,       prefix="/api", tags=["Live"])
app.include_router(auth.router,       prefix="/api", tags=["Auth"])
app.include_router(cities.router,     prefix="/api", tags=["Cities"])
app.include_router(engagement.router, prefix="/api", tags=["Engagement"])
app.include_router(workorders.router, prefix="/api", tags=["Work orders"])
app.include_router(analytics.router,  prefix="/api", tags=["Analytics"])
app.include_router(media.router,      prefix="/api", tags=["Media"])
app.include_router(quality.router,    prefix="/api", tags=["Quality"])
app.include_router(public_api.router, prefix="/api", tags=["Public API"])
app.include_router(contact.router,    prefix="/api", tags=["Contact"])

# ─────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RDDS API...")
    # Live-mode WS broadcasts are scheduled from sync handlers via this loop.
    live_ws_manager.capture_loop()
    ok = check_connection()
    if not ok:
        logger.warning(
            "Could not connect to database at startup. "
            "Make sure Docker is running: docker-compose up -d"
        )
    else:
        # Idempotent: creates only missing tables (live_events/live_reports,
        # users, city_landmarks on volumes that predate them). The core schema
        # still comes from db/init/01_schema.sql on fresh volumes. A failure
        # here means those features 500 — log it LOUDLY, don't whisper.
        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            logger.exception(
                "Base.metadata.create_all FAILED — live/auth tables may be "
                "missing and their endpoints will return 500. Fix the schema "
                "error above and restart the backend."
            )
        # create_all never adds NEW columns or indexes to EXISTING tables, so
        # everything that arrived after a volume was created is ensured here
        # with idempotent DDL instead. Fresh volumes get identical schema from
        # db/init/*.sql — keep the three definitions in sync.
        _upgrade_ddl = [
            # detections: evidence crops + repair verification loop
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS crop_path TEXT",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS fixed_at TIMESTAMPTZ",
            # live_reports: account linkage for gamification
            "ALTER TABLE live_reports ADD COLUMN IF NOT EXISTS user_id UUID "
            "REFERENCES users(id) ON DELETE SET NULL",
            # live_events: triage audit trail
            "ALTER TABLE live_events ADD COLUMN IF NOT EXISTS promoted_detection_id UUID "
            "REFERENCES detections(id) ON DELETE SET NULL",
            "ALTER TABLE live_events ADD COLUMN IF NOT EXISTS dismissed_at TIMESTAMPTZ",
            # indexes on existing tables
            "CREATE INDEX IF NOT EXISTS idx_live_reports_created "
            "ON live_reports (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_live_reports_user "
            "ON live_reports (user_id, created_at)",
        ]
        try:
            from sqlalchemy import text
            with engine.begin() as conn:
                for ddl in _upgrade_ddl:
                    conn.execute(text(ddl))
        except Exception:
            logger.exception("Could not ensure schema upgrades (non-fatal).")
        _seed_admin()
        logger.success("API ready.")


def _seed_admin() -> None:
    """
    Seed a starting admin account from env: ADMIN_USERNAME / ADMIN_EMAIL /
    ADMIN_PASSWORD. Runs on every startup; no-ops if the username or e-mail
    already exists.

    SECURITY: there are deliberately NO in-repo default credentials. If the
    three env vars are not all set, no admin is seeded and a warning is
    logged — an admin can then only be created by promoting a user from an
    existing admin account or by setting the env vars and restarting.
    """
    from backend.database import SessionLocal
    from backend.models_auth import User, ROLE_ADMIN
    from backend.auth import hash_password
    from sqlalchemy import or_, func as sqlfunc

    username = os.getenv("ADMIN_USERNAME", "").strip()
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")

    if not (username and email and password):
        logger.warning(
            "ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD are not all set — "
            "skipping admin seeding. Set them in .env to create the first admin."
        )
        return

    db = SessionLocal()
    try:
        existing = (
            db.query(User)
            .filter(or_(
                sqlfunc.lower(User.username) == username.lower(),
                sqlfunc.lower(User.email) == email,
            ))
            .first()
        )
        if existing:
            if existing.role != ROLE_ADMIN:
                existing.role = ROLE_ADMIN
                db.commit()
                logger.info("Seed admin '{}' promoted back to admin.", existing.username)
            return
        db.add(User(
            username=username,
            email=email,
            full_name="Paraschiv Tudor",
            password_hash=hash_password(password),
            role=ROLE_ADMIN,
            city="Cluj-Napoca",
        ))
        db.commit()
        logger.success("Seed admin account '{}' created.", username)
    except Exception:
        db.rollback()
        logger.exception("Could not seed the admin account.")
    finally:
        db.close()


# ─────────────────────────────────────────────
# Root health check
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "RDDS — Road Degradation Detection System API",
        "version": "1.0.0",
    }


# Every open browser tab probes health every 30 s. The DB round-trip is
# cached briefly so a thousand tabs cost the database one SELECT per 5 s.
_health_cache = {"at": 0.0, "ok": False}


def _db_ok_cached() -> bool:
    import time
    now = time.monotonic()
    if now - _health_cache["at"] > 5.0:
        _health_cache["ok"] = check_connection()
        _health_cache["at"] = now
    return _health_cache["ok"]


def _health_payload() -> dict:
    db_ok = _db_ok_cached()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


@app.get("/health", tags=["Health"])
def health():
    return _health_payload()


@app.get("/api/health", tags=["Health"])
def api_health():
    """
    Same probe, but under /api so it is reachable through the Nginx proxy
    (which only forwards /api/*). Public on purpose: the navbar health dot
    must work on the login page, before any session exists.
    """
    return _health_payload()


# ─────────────────────────────────────────────
# Direct run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )
