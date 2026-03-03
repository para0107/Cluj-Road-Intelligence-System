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

from backend.database import check_connection
from backend.routes import detections, stats, heatmap, priority

load_dotenv()

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Cluj Urban Monitor API",
    description="Road damage detection and monitoring system for Cluj-Napoca.",
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

# ─────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Cluj Urban Monitor API...")
    ok = check_connection()
    if not ok:
        logger.warning(
            "Could not connect to database at startup. "
            "Make sure Docker is running: docker-compose up -d"
        )
    else:
        logger.success("API ready.")


# ─────────────────────────────────────────────
# Root health check
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Cluj Urban Monitor API", "version": "1.0.0"}


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