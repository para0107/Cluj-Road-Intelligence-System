# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RIDS — Road Infrastructure Detection System. Detects, classifies, and prioritizes road damage
in Cluj-Napoca from dashcam video (.mp4 + optional .gpx) via a 7-stage CV pipeline
(RT-DETR-L → SAM 2.1 → Monodepth2 → rule-based severity → DBSCAN dedup → PostGIS), served by a
FastAPI API and a React + Leaflet SPA. Bachelor's thesis project (Babeș-Bolyai University, 2026).
The README.md is thorough and current — read it for full detail; this file is the condensed
operating guide.

## Commands

```bash
# Docker stack (db + backend + frontend) — this is the normal way to run the web app
docker compose up -d --build            # rebuild + start everything
docker compose up -d --build frontend   # REQUIRED after ANY frontend/src change (static Nginx build)
docker compose logs -f backend          # tail backend logs
docker compose down                     # stop (keeps pgdata volume); add -v to wipe the DB

# Dev mode without Docker (from project root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend && npm install && npm run dev    # Vite :3000, proxies /api → :8000

# Host GPU pipeline (required only to process uploads; needs .venv + ml/weights/)
python pipeline/job_watcher.py          # daemon: watches data/jobs/, must be running for uploads

# Manual pipeline run (bypasses backend + watcher)
python pipeline/orchestrator.py --video data/raw/footage/x.mp4 --gps data/raw/gps_logs/x.gpx \
    --device cuda --fps 2.0             # flags: --resume --dry_run_db --save_debug --verbose
```

- Frontend: http://localhost:3000 · API docs: http://localhost:8000/docs · Health: /health
- DB schema auto-creates from `db/init/01_schema.sql` on a **fresh** pgdata volume only;
  `python scripts/setup_db.py` is the idempotent equivalent for existing volumes.
- **No pytest suite and no linter.** Verification = stand-alone scripts run individually:
  `python scripts/validate_severity.py`, `validate_deduplication.py`, `validate_depth.py`,
  `validate_db_write.py`, `validate_nrdd2024.py`.

## The two-context execution model (most important thing to understand)

The Dockerized backend **cannot** run ML — no GPU, no torch in its image
(`backend/requirements.backend.txt` is runtime-only, keep it that way). The pipeline runs on the
**Windows host** in `.venv` with CUDA. The two sides communicate **only through files** under the
shared bind mount `./data` ⇄ `/app/data` — no sockets, no queues:

1. `POST /api/ingest/upload` saves video/gps under `data/raw/…` and writes
   `data/jobs/<job_id>.json` with `status: "pending"` and **container-side** paths.
2. `pipeline/job_watcher.py` (host daemon) polls `data/jobs/`, rewrites the `/app/data` prefix to
   the host path (`_container_to_host()`), sets status `running`, and spawns
   `orchestrator.py --device cuda --session_id <job_id>`.
3. The orchestrator writes `data/processed/sessions/<job_id>/session.json` **after every stage**.
4. `GET /api/ingest/status/{job_id}` returns session.json (authoritative) or falls back to the
   job file (`initialising`/`pending`).

`job_id` == `session_id`, format `YYYYMMDD_HHMMSS` (UTC). Statuses:
`pending → initialising → running → complete | failed`. One job at a time — the upload route
409s while a job file is pending/running (stale-job escape hatch: `JOB_STALE_TIMEOUT_S`, 2 h).

## The 7 stages and their sync points

Canonical order in `orchestrator.run()`: `preprocessor → detector → segmentor → depth_estimator
→ severity_classifier → deduplicator → db_writer`. Artifacts land in numbered dirs
(`01_manifest/ … 07_db_write/`) under the session folder.

**A stage key must stay in sync in three places:** `pipeline/orchestrator.py` (stage list),
the `stages` array in session.json consumed via `backend/routes/ingest.py`, and the `STAGES` /
stage-label maps in `frontend/src/pages/IngestionPage.jsx`.

**GPS guard:** with no usable GPS, stages 6–7 are skipped and the run still finishes
`complete` (nothing written to the DB). A GPS-less run is NOT a failure — don't "fix" it.

## Database

PostGIS, two tables (`backend/models.py` mirrors `db/init/01_schema.sql`): `detections` (one row
per de-duplicated physical damage; upsert matches same `damage_type` within
`DEDUP_CLUSTER_RADIUS_M` via `ST_DWithin` on geography and bumps `detection_count`) and
`survey_log` (one row per `survey_date`).

**Port nuance:** inside Docker the backend uses `db:5432`; host-run code (pipeline, db_writer)
uses `localhost:${POSTGRES_PORT}` — `.env` sets `POSTGRES_PORT=5433`.

**Enrichment is permanently removed.** No Nominatim/Overpass/Open-Meteo calls, no
`street_name`/`road_importance`/`weather` columns. GPS lat/lon is the only location metadata.
Do not re-introduce it.

## Severity & priority (rule-based, no learned model)

`severity_score = min(Σ w_signal·S_signal · class_weight · 2, 1)` from four normalized signals
(depth, mask area, interior contrast, edge sharpness) with per-class weights, mapped to S1–S5
bands. Marking classes (`lane_line_blur`, `pedestrian_crossing_blur`) are capped low by
`class_weight`, not special-cased. Priority = `severity · log(detection_count + 1)`
(`Detection.compute_priority_score`). 10 damage classes follow the N-RDD2024 schema — the
class list lives in `frontend/src/utils/constants.js` and the detector config.

## Frontend conventions

- React 18 + Vite 5 + React Router 6, react-leaflet, Recharts, lucide-react, axios.
  **One global context only**: `AuthContext` (session/user). Everything else is
  page-local hooks; the cross-page pipeline signal is `localStorage['rids_active_job']`
  (set by IngestionPage, polled by MapPage every 10 s).
- Every route except `/login` and `/register` is wrapped in `RequireAuth`; the JWT
  lives in `localStorage['rids_token']` and is attached by an axios interceptor in
  `utils/api.js` (a 401 outside `/auth/*` clears the session and redirects to /login).
- Map tiles follow the app theme (dark → Carto dark, light → Carto voyager) via
  `hooks/useTheme.js`; MapPage's switcher overrides per-session, LivePage always follows.
- Styling = inline style objects + CSS design tokens (`--bg`, `--accent`, …) in
  `frontend/src/index.css`; dark theme default, `:root.light` overrides. No Tailwind/CSS modules.
- All API calls go through `frontend/src/utils/api.js` (axios, baseURL `/api`).
- The production frontend is a **static build baked into the Nginx image** — changes to
  `frontend/src/**` are invisible until `docker compose up -d --build frontend`.

## Configuration

Single root `.env` (git-ignored, contains the DB password) loaded by python-dotenv and compose.
Paths are project-root-relative; `PROJECT_ROOT` stays unset (watcher auto-detects).
`PROJECT_DATA_DIR=/app/data` must equal the compose bind-mount target. Model weights live under
`ml/weights/` (`best.pt`, `sam2.1_hiera_tiny.pt`, `mono_640x192/`, `networks/`) and are **not in
git** — the Docker stack runs fine without them; only the host pipeline needs them.

## Auth & roles

JWT auth (PyJWT HS256, `JWT_SECRET`), passwords = stdlib PBKDF2 (no bcrypt dep).
Three roles: `user`, `municipality` (operator scoped to a chosen city), `admin`.
`backend/auth.py` provides `get_current_user` / `require_operator` / `require_admin`;
protected: ingest upload (any user), detections mutations + live resolve (operator),
user/role management (admin). Seed admin is created at startup from
`ADMIN_USERNAME/EMAIL/PASSWORD` env (defaults in `main.py::_seed_admin`). Google OAuth
is optional and free (set `GOOGLE_CLIENT_ID`); Apple Sign-In is deliberately absent
(paid Apple program — violates the project's zero-cost rule). Edge scripts authenticate
via `--email/--password/--token` or `RIDS_EMAIL/RIDS_PASSWORD/RIDS_TOKEN`.

**Landmark lookups are NOT pipeline enrichment.** The enrichment ban above applies to
the detection pipeline and DB columns. `backend/routes/cities.py` may call Nominatim
(free, 1 req/s, cached forever in `city_landmarks`) purely for the map's fly-to menu.

## Live (Waze-like) mode & the lite pipeline

Coexists with Survey mode; see `docs/LIVE_MODE.md` for the full design. Key points:
`backend/routes/live.py` + `backend/models_live.py` implement crowd-validated events
(`live_events`/`live_reports`, TTL-expiring, distinct-device escalation
unverified→confirmed→verified). WS push at `/api/live/ws` (nginx has the Upgrade block;
Vite proxy has `ws: true`); clients fall back to polling `GET /api/live/events`.
Broadcasts from sync handlers go through `live_manager.broadcast_from_thread()` — never
`asyncio.run()` in a route. Live tables are created by `Base.metadata.create_all` at startup
(plus `db/init/02_live_schema.sql` + `03_auth_schema.sql` on fresh volumes — keep both
in sync). **Never add an explicit `Index(..., "geom")` to a GeoAlchemy2 model** — the
column auto-creates `idx_<table>_<col>` and the name collision silently breaks create_all.

**Lite pipeline (per-user instance):** `pipeline/live_pipeline.py` = same `best.pt` +
per-class thresholds (imported from `pipeline/detector.py`) + motion gating + frame
stride + fp16 (or one-time free ONNX export for CPU), with `pipeline/lite_severity.py`
feeding the UNMODIFIED stage-5 formula via ~0.4 ms CV proxies (Otsu mask, Sobel
boundary sharpness, interior-vs-ring contrast, and the depth stage's own geometry-proxy
fallback). `pipeline/simulate_fleet.py` = multi-vehicle demo, no GPU. Scaling model:
every user runs their own lite instance; the server stays a stateless aggregator.

**Survey detector efficiency:** stage 2 batches frames (`DETECTOR_BATCH`, auto 8 on
CUDA) and uses fp16 (`DETECTOR_HALF`, auto on CUDA); `DETECTOR_IMGSZ` overrides input
size. Batching never changes detections; fp16 differences are negligible (<1e-3).

## Layout notes

- `backend/` — FastAPI (routes: detections, stats, heatmap, priority, export, ingest).
- `pipeline/` — the 7 stage modules + `orchestrator.py` + `job_watcher.py`.
- `ml/` — training/research only (never imported by the backend); inference reads `ml/weights/`.
- `scripts/` — one-off setup/validation utilities; legacy `validate_*.py` still hardcode
  absolute paths (research tools, not runtime).
- `scheduler/daily_job.py` — optional APScheduler nightly run.
