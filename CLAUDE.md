# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RDDS — Road Degradation Detection System. A complete **municipal road-monitoring platform**:
- Detects, classifies, and prioritizes road damage from dashcam video via a 7-stage CV pipeline
  (RT-DETR-L → SAM 2.1 → Monodepth2 → rule-based severity → DBSCAN dedup → PostGIS)
- Serves citizen gamification (points, badges, streaks, leaderboards) to drive engagement
- Operates a municipality workflow suite (triage → work orders → repair verification with
  reopened-damage guard)
- Publishes a Road Quality Index (RQI) heatmap for city-level planning
- Exposes a free public developer API with per-key rate limiting
- Runs an in-browser AI assistant with guardrails (lexical grounding, injection defense,
  no server-side model)
- Hardened for scale: rate limiting, ALTCHA captcha, CSP, WebSocket origin/connection caps

**The README.md is thorough and current** — read it for full architecture and feature detail.
This file is the condensed operating & development guide.

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

- Frontend: http://localhost:3000 · API docs: http://localhost:8000/docs · Health: `/health`
  (direct) or `/api/health` (public, proxied — the navbar dot uses it pre-login).
- DB schema auto-creates from `db/init/*.sql` (01–05) on a **fresh** pgdata volume only;
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
Uploads are streamed to disk in 8 MB chunks and size-capped (`MAX_UPLOAD_MB` 4 GB /
`MAX_GPS_MB` 50 MB); upload needs an operator account, status polling any authenticated
user, and job ids must match `^\d{8}_\d{6}$` before touching the filesystem.

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

PostGIS. Core survey tables (`backend/models.py` mirrors `db/init/01_schema.sql`):
`detections` (one row per de-duplicated physical damage; upsert matches same
`damage_type` within `DEDUP_CLUSTER_RADIUS_M` via `ST_DWithin` on geography and bumps
`detection_count`) and `survey_log` (one row per `survey_date`).

Feature tables: `live_events` / `live_reports` / `live_devices` (live mode),
`users` / `pending_registrations` / `city_landmarks` (auth), `user_points` /
`user_stats` / `user_badges` / `notifications` (engagement), `work_orders` /
`work_order_items` (repairs), `api_keys` (public API).

**Adding a column or index to an EXISTING table means editing three places** —
`create_all` will not do it for you: (1) the model, (2) an idempotent
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` in the
`_upgrade_ddl` list in `main.py::startup_event` (for existing volumes), and (3) the
matching `db/init/*.sql` (for fresh volumes). `db/init/06–08` add engagement, work
orders and API keys; note `live_reports.user_id` lives in `06` rather than `02`
because `users` does not exist yet at that point in the init order.

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
- Public routes: `/login`, `/register`, `/pricing`, `/developers`. Everything else is
  wrapped in `RequireAuth`; the survey/operations pages (`/map`, `/stats`, `/explorer`,
  `/priority`, `/ingest`, `/triage`, `/workorders`, `/quality`) are additionally wrapped
  in `RequireOperator` (`AuthContext` exposes `isOperator`/`isAdmin`) — citizens get
  Command, Live, My impact, Assistant and About. The JWT lives in
  `localStorage['rids_token']` and is attached by an axios interceptor in `utils/api.js`
  (a 401 outside `/auth/*` clears the session and redirects to /login).
- **All pages are `React.lazy` chunks** (`App.jsx`) and vendors are split in
  `vite.config.js`. The navbar groups operator pages into two dropdowns (Survey,
  Operations) so twelve pages still fit.
- ReactBits components are **vendored** (JS-CSS variants) under `src/reactbits/`, not
  installed as a package. `SpotlightCard.css` was re-pointed at the design tokens (it
  shipped hardcoded `#111`, which broke the light theme). Gate every WebGL/GSAP effect
  on `useMotionOk()` — `AnimatedContent` sets `visibility:hidden` until GSAP reveals it,
  so rendering it with motion off hides the content.
- Map tiles follow the app theme (dark → Carto dark, light → Carto voyager) via
  `hooks/useTheme.js`; MapPage's switcher overrides per-session, LivePage always follows.
- **Maps never hardcode a city.** They open on the signed-in user's city via
  `hooks/useCityCenter.js` (localStorage cache → `GET /cities/center` → country-level
  fallback). Every account must have a city: required at registration, and city-less
  sessions (Google first login) are blocked by `components/CityGate.jsx` until one is set.
- Mobile: single 768px breakpoint (`hooks/useIsMobile.js` + media queries in
  `index.css`). LivePage layout lives in `live-*` CSS classes (bottom-sheet feed,
  GPS-anchored report FAB, follow-mode arrow puck, drive HUD); Navbar collapses to a
  hamburger.
- Styling = inline style objects + CSS design tokens (`--bg`, `--accent`, …) in
  `frontend/src/index.css`; dark theme default, `:root.light` overrides. No Tailwind/CSS modules.
- All API calls go through `frontend/src/utils/api.js` (axios). baseURL is same-origin
  `/api` by default; `VITE_API_URL` (build-time) points a separately hosted frontend
  (Vercel — root `vercel.json` builds `frontend/dist`) at a remote backend.
- The production frontend is a **static build baked into the Nginx image** — changes to
  `frontend/src/**` are invisible until `docker compose up -d --build frontend`.

## Configuration

Single root `.env` (git-ignored, contains the DB password) loaded by python-dotenv and compose.
Paths are project-root-relative; `PROJECT_ROOT` stays unset (watcher auto-detects).
`PROJECT_DATA_DIR=/app/data` must equal the compose bind-mount target. Model weights live under
`ml/weights/` (`best.pt`, `sam2.1_hiera_tiny.pt`, `mono_640x192/`, `networks/`) and are **not in
git** — the Docker stack runs fine without them; only the host pipeline needs them.

## Auth & roles

JWT auth (PyJWT HS256, `JWT_SECRET`), passwords = stdlib PBKDF2 (600k iterations; older
hashes keep verifying — the count is stored per hash). Three roles: `user`,
`municipality` (operator scoped to a chosen city), `admin`. `backend/auth.py` provides
`get_current_user` / `require_operator` / `require_admin`; protected: ingest upload +
detections mutations + live resolve (operator), user/role/registration management
(admin). Login is rate-limited per identifier+IP.

**No credential defaults anywhere — deliberate.** `JWT_SECRET` unset (or the old known
dev value) → a random per-process secret is generated: the app works but sessions die on
restart. `ADMIN_USERNAME/EMAIL/PASSWORD` not all set → no admin is seeded
(`main.py::_seed_admin` just warns). Set all four in `.env`. Don't "fix" this by adding
fallbacks.

**Registration is verified before an account exists.** `POST /auth/register` writes to
`pending_registrations`, not `users`: a 6-digit code (30 min TTL) must be confirmed via
`/auth/verify-email`; municipality registrations then sit in `awaiting_approval` until an
admin approves/denies them (Admin page → Pending approvals). With no e-mail transport
configured the code step is skipped (user accounts create immediately; municipality goes
straight to awaiting_approval). E-mail (`backend/notify.py`) is optional and
fire-and-forget (daemon thread, never fails a request); transports: Brevo HTTPS API
(`BREVO_API_KEY`, free tier — the only option on SMTP-blocked networks, wins when set)
or stdlib SMTP (`SMTP_HOST/PORT/USERNAME/PASSWORD/FROM`).

Google OAuth is optional and free (set `GOOGLE_CLIENT_ID`); Apple Sign-In is deliberately
absent (paid Apple program — violates the project's zero-cost rule). Edge scripts
authenticate via `--email/--password/--token` or `RDDS_EMAIL/RDDS_PASSWORD/RDDS_TOKEN`.

**Landmark lookups are NOT pipeline enrichment.** The enrichment ban above applies to
the detection pipeline and DB columns. `backend/routes/cities.py` may call Nominatim
(free, 1 req/s, cached forever in `city_landmarks`) purely for the map's fly-to menu
and `GET /cities/center` (city geocode, cached under `kind='city'`, excluded from the
menu). All Nominatim traffic is serialized through one module lock — keep it that way.
CORS origins come from `CORS_ORIGINS` (comma-separated, default `*`).

## Live (Waze-like) mode & the lite pipeline

Coexists with Survey mode; see `docs/LIVE_MODE.md` for the full design. Key points:
`backend/routes/live.py` + `backend/models_live.py` implement crowd-validated events
(`live_events`/`live_reports`, TTL-expiring, distinct-device escalation
unverified→confirmed→verified). WS push at `/api/live/ws` (nginx has the Upgrade block;
Vite proxy has `ws: true`); clients fall back to polling `GET /api/live/events`.
Broadcasts from sync handlers go through `live_manager.broadcast_from_thread()` — never
`asyncio.run()` in a route. Live tables are created by `Base.metadata.create_all` at startup
(plus `db/init/02…05_*.sql` on fresh volumes — keep the SQL files and the models in sync). **Never add an explicit `Index(..., "geom")` to a GeoAlchemy2 model** — the
column auto-creates `idx_<table>_<col>` and the name collision silently breaks create_all.

**Devices & phone drive mode:** live reports are tied to paired devices (`live_devices`).
Phones/browsers self-pair (`POST /live/devices/pair` with a device_id); edge agents
instead receive a short-lived single-use pairing code and exchange it via
`/live/devices/claim` — no password ever typed on the vehicle machine. Reports from
revoked devices are rejected. `frontend/src/utils/driveMode.js` turns the phone into a
sensor: a DeviceMotion jolt spike while GPS says the car is moving fires one auto
pothole report, then an 8 s cooldown (iOS ≥ 13 needs `requestPermission()` from a click
handler, so `start()` must run in a click).

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

**Read-path scaling (load-tested at 500 concurrent clients, zero errors):**
`GET /live/events` + `/live/stats` serve from a short in-process cache
(`LIVE_READ_CACHE_S`, 2 s) that every mutation clears via `_broadcast`; the health
probes cache the DB ping for 5 s; WS fan-out sends concurrently (one slow client
can't stall the rest); DB pool is env-tunable (`DB_POOL_SIZE`/`DB_MAX_OVERFLOW`,
15/25). Keep ONE uvicorn worker: WS fan-out is per-process (Redis pub/sub is the
documented multi-worker upgrade path in `live_manager.py`). New indexes on existing
tables must be added BOTH to the model and as idempotent `CREATE INDEX IF NOT
EXISTS` in `main.py::startup` (create_all won't add them) plus the SQL init file.

**User-facing copy style:** plain sentences, no em-dashes, no academic references —
this includes e-mails (`backend/notify.py`, signed "The RDDS team"), API error
`detail` strings, and UI text. The brand is "RDDS — Road Degradation Detection
System"; internal keys (`rids_*` localStorage, `cluj_monitor_*` containers) are
deliberately NOT renamed.

## Citizen engagement (points, badges, notifications)

`backend/gamification.py` + `backend/models_engagement.py` + `routes/engagement.py`.
**A raw report earns zero points — this is load-bearing, not an oversight.** Points
are granted only on outcomes a lone spammer cannot manufacture: the event reaching
`confirmed` (+10) or `verified` (+15) through *distinct-device* escalation, an
operator promoting it (+20), or the damage being fixed (+25). Awards go to every
distinct `live_reports.user_id` supporting the event, and the ledger's unique
`(user_id, reason, ref_id)` makes them idempotent (a replayed hook inserts nothing).
`user_stats` is a denormalized snapshot so the leaderboard is one indexed scan, not
a SUM over the ledger. Reports are additionally capped at 1/15 s and 40/day per user.

Do NOT "fix" this by awarding points on report — it re-opens the farm.

## Municipality suite (the revenue path)

Detect → triage → plan → repair → verify, end to end:

- **Triage** (`/live/triage`, `promote`, `dismiss`) turns a crowd report into a real
  `detections` row (it then flows through the map, priority queue and work orders
  like any pipeline result).
- **Work orders** (`backend/routes/workorders.py`) group detections into one crew job
  through `open → scheduled → in_progress → repaired → verified`. Moving to `repaired`
  stamps `fixed_at` on every item.
- **Repair verification**: a detection is *reopened* when `last_detected > fixed_at`
  (the pipeline saw the damage again after it was signed off). A work order **cannot**
  move to `verified` while any item is reopened — the route 409s with the offending
  ids. This is the feature that makes the product defensible; do not soften it.
- **Route planning** is client-side (`frontend/src/utils/routePlan.js`): haversine +
  nearest-neighbour + 2-opt. **No routing API** — that would cost money and leak the
  city's schedule.
- **Road Quality Index** (`backend/routes/quality.py`): grid cells (~120 m) scored
  `100·exp(-penalty/8)` from severity, recency and repair state, banded A–E. Grid math
  is plain lat/lon — **no new geometry columns**, which also sidesteps the GeoAlchemy2
  index trap.
- **Evidence photos**: `pipeline/db_writer.py` crops the detection's bbox out of its
  source frame into `<session>/07_db_write/evidence/` and stores a data-dir-relative
  `crop_path`. Served only through `GET /media/evidence/{id}` (operator, path-validated
  against `PROJECT_DATA_DIR`). Kill switch: `EVIDENCE_CROPS=false`. Never fatal — every
  failure path silently skips the crop.

## Public API & anti-bot

`routes/public_api.py`: keys are `rdds_<hex>`, stored as SHA-256 + prefix only, shown
once. `/v1/public/*` is GET-only, rate-limited per key, and exposes no user, device or
photo data. `backend/altcha.py` is a **stdlib** proof-of-work captcha (no vendor, no
outbound call) gated by `CAPTCHA_ENABLED`; off by default so dev and scripts keep
working. `backend/ratelimit.py` is the shared limiter for every abusable route — see
docs/SECURITY.md for the budget table.

## The assistant (`/assistant`) — zero cost by construction

Hybrid, entirely client-side. **Never add a paid model API here.**
- Instant mode (default, every device): intent handlers answer *number* questions from
  the real API (`assistant/intents.js`), and MiniSearch retrieves from a curated
  knowledge base (`assistant/knowledge.js`). This is why the assistant cannot invent a
  statistic: those questions never reach a model.
- AI mode (opt-in): WebLLM runs Llama-3.2-1B in the user's browser on WebGPU, and
  transformers.js adds dense retrieval. Both are **dynamic imports** — keep them that
  way, or the 6 MB runtime lands in the initial bundle.
- `assistant/graph.js` is an explicit state machine (guard → intents → retrieve →
  conditional HyDE → generate → ground-check). `assistant/guard.js` drops any sentence
  the retrieved context does not support and strips numbers that are not in the context.
  There is deliberately **no LLM-as-judge** (a second pass on a 1B model costs latency
  and buys little).

## Layout notes

- `backend/` — FastAPI (routes: auth, detections, stats, heatmap, priority, export,
  ingest, live, cities, engagement, workorders, analytics, quality, public_api, media,
  contact).
- `pipeline/` — the 7 stage modules + `orchestrator.py` + `job_watcher.py`.
- `docs/` — `LIVE_MODE.md`, `FUNCTIONALITY.md`, `SECURITY.md`, `FREE_DEPLOYMENT.md`
  (free split hosting: Vercel static frontend + host backend via `VITE_API_URL`).
- `ml/` — training/research only (never imported by the backend); inference reads
  `ml/weights/` (not in git; `python scripts/download_weights.py --dataset <kaggle id>`
  fetches them, or set `KAGGLE_WEIGHTS_DATASET` in `.env`).
- `scripts/` — one-off setup/validation utilities; legacy `validate_*.py` still hardcode
  absolute paths (research tools, not runtime).
- `scheduler/daily_job.py` — optional APScheduler nightly run.
