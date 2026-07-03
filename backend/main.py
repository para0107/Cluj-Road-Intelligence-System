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
from backend.routes import detections, stats, heatmap, priority, export, ingest, live
from backend.live_manager import manager as live_ws_manager
import backend.models_live  # noqa: F401 — register live_events/live_reports on Base.metadata

load_dotenv()

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="RIDS — Road Infrastructure Detection System API",
    description=(
        "Road damage detection, classification, and prioritization system "
        "for Cluj-Napoca, Romania. Babeș-Bolyai University, 2026."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# CORS — allow the local frontend to call the API
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ─────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RIDS API...")
    # Live-mode WS broadcasts are scheduled from sync handlers via this loop.
    live_ws_manager.capture_loop()
    ok = check_connection()
    if not ok:
        logger.warning(
            "Could not connect to database at startup. "
            "Make sure Docker is running: docker-compose up -d"
        )
    else:
        # Idempotent: creates only missing tables (live_events / live_reports
        # on stacks whose pgdata volume predates Live mode). The core schema
        # still comes from db/init/01_schema.sql on fresh volumes.
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as exc:
            logger.warning("create_all skipped: {}", exc)
        logger.success("API ready.")


# ─────────────────────────────────────────────
# Root health check
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "RIDS — Road Infrastructure Detection System API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    db_ok = check_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


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
