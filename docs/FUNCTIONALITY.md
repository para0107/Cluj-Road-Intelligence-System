# RDDS — Full Functionality Reference

**Road Degradation Detection System** · 2026

This document describes **everything the system does and how it does it**: the two
processing pipelines and their attributes, the live crowd-sensing mode, authentication and
roles, every backend endpoint, every UI page, the database, the scalability model, and the
configuration surface. It complements `README.md` (setup detail) and `docs/LIVE_MODE.md`
(live-mode design deep-dive).

> **Update (July 2026):** the auth flow was hardened after this document was
> first written — accounts now require **e-mail verification** before creation,
> municipality accounts additionally require **admin approval**, citizens see
> only the Command/Live/System pages (survey/analytics pages and endpoints are
> operator-only), login is rate limited, users can delete their own accounts,
> and admins can enable/disable/delete accounts. The authoritative, current
> description of auth, roles, and endpoint protection lives in `README.md`
> (Accounts, roles & notifications + REST API) and `docs/SECURITY.md`.

---

## 1. What RDDS is

RDDS detects, classifies, and prioritizes **road damage** from ordinary dashcam video.
A city (demo: Cluj-Napoca) gets:

- a **survey workflow** — upload a drive video (+ optional GPS track), get de-duplicated,
  severity-scored, geolocated damage records in a PostGIS database;
- a **live workflow** — every user's phone or dashcam acts as a real-time sensor; hazards
  appear on a shared map within milliseconds and are validated by the crowd (Waze-style);
- a **decision layer** — priority ranking, statistics, heatmaps, repair-cost sketches,
  CSV export, and operator tools (mark repaired, resolve, delete).

**Zero-cost rule:** every component is free — open-source models, free map tiles,
OpenStreetMap/Nominatim geocoding, stdlib SMTP for e-mail, no paid APIs anywhere. Anything
that would cost money (e.g. Apple Sign-In) is deliberately absent.

**Damage taxonomy** — the 10-class N-RDD2024 schema (Kaya & Codur 2024):
`longitudinal_crack (D00)`, `transverse_crack (D10)`, `alligator_crack (D20)`,
`repaired_crack (D30)`, `pothole (D40)`, `pedestrian_crossing_blur (D50)`,
`lane_line_blur (D60)`, `manhole_cover (D70)`, `patchy_road (D80)`, `rutting (D90)`.

---

## 2. Architecture — the two-context execution model

The single most important design fact: **the web stack cannot run ML, and the ML host does
not serve HTTP.** They communicate only through files.

```
┌────────────────────── Docker ──────────────────────┐   ┌───── Windows host (GPU) ─────┐
│  frontend (Nginx, static React build)   :3000      │   │  .venv + CUDA + ml/weights/  │
│  backend  (FastAPI, no torch)           :8000      │   │  pipeline/job_watcher.py     │
│  db       (PostGIS 15)                  :5433→5432 │   │  pipeline/orchestrator.py    │
└──────────────────┬─────────────────────────────────┘   └──────────────┬───────────────┘
                   │            shared bind mount                       │
                   └────────────  ./data ⇄ /app/data  ──────────────────┘
```

Survey job lifecycle (`job_id == session_id`, format `YYYYMMDD_HHMMSS` UTC):

1. `POST /api/ingest/upload` saves the video/GPX under `data/raw/…` and writes
   `data/jobs/<job_id>.json` with `status: "pending"` (container-side paths inside).
2. `pipeline/job_watcher.py` (host daemon, polls every `WATCHER_POLL_S` s) rewrites the
   `/app/data` prefix to the host path, flips the job to `running`, and spawns
   `orchestrator.py --device cuda --session_id <job_id>`.
3. The orchestrator rewrites `data/processed/sessions/<job_id>/session.json` **after every
   stage** — this file is the single source of truth for progress.
4. `GET /api/ingest/status/{job_id}` serves session.json (or the job file while the
   orchestrator hasn't started). Statuses: `pending → initialising → running →
   complete | failed`.

One survey job runs at a time (upload returns **409** while one is pending/running); a job
silent for `JOB_STALE_TIMEOUT_S` (default 2 h) is treated as dead and stops blocking.

---

## 3. The survey pipeline — 7 stages

Run by `pipeline/orchestrator.py` (flags: `--video --gps --device --fps --resume
--dry_run_db --save_debug --verbose`). Each stage writes an artifact directory
`01_manifest/ … 07_db_write/` under the session folder and appends itself to
session.json's `stages` array.

| # | Stage | Model / method | What it produces |
|---|-------|----------------|------------------|
| 1 | `preprocessor` | OpenCV + gpxpy | Frames extracted at `PIPELINE_FPS` (default 2/s), GPS linear-interpolated per frame, lighting condition per frame; manifest.json |
| 2 | `detector` | **RT-DETR-L** fine-tuned on N-RDD2024 (`ml/weights/best.pt`) | Bounding boxes; global conf 0.001 then **per-class thresholds** (0.35 all classes, 0.50 `lane_line_blur`); batched (`DETECTOR_BATCH`, auto 8 on CUDA) + fp16 (`DETECTOR_HALF`, auto on CUDA); `DETECTOR_IMGSZ` overrides input size; TTA disabled (measured zero gain) |
| 3 | `segmentor` | **SAM 2.1 hiera-tiny** (box-prompted) | Pixel mask per box → 4 geometry features: surface area (px→cm²), edge sharpness, interior contrast, mask compactness |
| 4 | `depth_estimator` | **Monodepth2** (`mono_640x192`) | Relative disparity → normalized depth per detection + depth confidence; low-light frames skipped; geometry-proxy fallback when depth is unusable |
| 5 | `severity_classifier` | Rule-based (no learned model) | S1–S5 severity per detection (formula below) |
| 6 | `deduplicator` | **DBSCAN** on lat/lon, Haversine metric | Same physical damage seen on N frames → 1 record; eps = `DEDUP_CLUSTER_RADIUS_M` (default 2 m); keeps the highest-severity member; computes `surrounding_density` within `SURROUNDING_DENSITY_RADIUS_M` (50 m); HTML before/after report |
| 7 | `db_writer` | SQLAlchemy + PostGIS | Upsert into `detections`: an existing row of the same `damage_type` within the radius (`ST_DWithin` on geography) bumps `detection_count` and refreshes `last_detected`; else insert. Recomputes priority score |

**GPS guard:** with no usable GPS, stages 6–7 are skipped and the run still ends
`complete` — nothing enters the DB (a GPS-less run is a valid preview, not a failure).

**Severity formula (stage 5)** — unchanged everywhere it is used (survey and lite):

```
S_depth     = depth_norm                      (0..1)
S_area      = min(surface_area_px / 1000, 1)
S_contrast  = min(interior_contrast / 2.0, 1)
S_sharpness = min(edge_sharpness / 60.0, 1)

raw_score      = w_depth·S_depth + w_area·S_area + w_contrast·S_contrast + w_sharp·S_sharpness
severity_score = min(raw_score · class_weight · 2.0, 1.0)
```

- Signal weights are **per class** (pothole ≈ depth+contrast; alligator crack ≈ area;
  longitudinal crack ≈ sharpness) and sum to 1.
- `class_weight` encodes structural criticality (pothole 0.50 … lane_line_blur 0.10) —
  marking classes are capped low **by the weight**, not special-cased.
- Bands: S1 <0.15 · S2 <0.35 · S3 <0.55 · S4 <0.75 · S5 ≥0.75
  (Monitor / Schedule / Priority / Urgent / Emergency).

**Priority** (backend, `Detection.compute_priority_score`):
`priority = severity · ln(detection_count + 1)` — recurrence amplifies severity.

---

## 4. The lite (live) pipeline — per-user edge agent

`pipeline/live_pipeline.py` — one instance per vehicle/user, running on **their** hardware:

```
dashcam/webcam ─► frame stride (--every, def. 10) ─► motion gate (64×36 gray diff,
skips parked/red-light frames) ─► RT-DETR (same best.pt, fp16, --imgsz 480)
─► same per-class thresholds ─► lite severity (same stage-5 formula fed by ~0.4 ms
CV proxies: Otsu mask, Sobel boundary sharpness, interior-vs-ring contrast, depth
geometry proxy) ─► local dedup (--min-gap-m 20, --cooldown-s 30)
─► POST /api/live/reports (JWT)
```

Attributes and options:

- **Same weights, same thresholds, same severity formula** as the survey pipeline —
  results are directly comparable; the lite severity carries an honest ~0.5
  `severity_confidence`.
- **CPU-only support:** one-time free ONNX export (`--export-onnx [--quantize]`), then
  `--weights ml/weights/best.onnx --device cpu`.
- **Sources:** `--video file.mp4` (replay, `--realtime` for native speed) or `--camera 0`;
  position from `--gps track.gpx` (time-interpolated) or fixed `--lat/--lon`.
- **Auth:** `--pair <CODE>` (pairing code from the Live page — saves a JWT to
  `~/.rids_live_token`, no password ever typed on the car PC), or `--token`, or
  `--email/--password` (env: `RDDS_TOKEN/RDDS_EMAIL/RDDS_PASSWORD`).
- **Identity:** stable per-machine `device_id` persisted in `~/.rids_device_id`.
- `pipeline/simulate_fleet.py` — multi-vehicle demo without GPU: replays several virtual
  vehicles posting reports so the live map can be demonstrated anywhere.

---

## 5. Live mode — crowd-validated hazards (Waze-like)

State lives in two tables (`live_events`, `live_reports`); handlers are stateless.

- **Reporting:** `POST /api/live/reports` clusters the sighting into the nearest active
  event of the same `damage_type` within `LIVE_CLUSTER_RADIUS_M` (default 25 m, PostGIS
  `ST_DWithin` on geography) or creates a new event.
- **Validation by distinct device:** 1 device → `unverified`; ≥`LIVE_CONFIRM_DEVICES` (2)
  → `confirmed`; ≥`LIVE_VERIFY_DEVICES` (3) → `verified`. A device confirming twice is
  idempotent — no self-boosting.
- **Dispute:** enough independent "not there" signals
  (≥ max(`LIVE_DISPUTE_MIN`, reporter count)) deactivate the event.
- **TTL:** every supporting signal pushes `expires_at` forward by `LIVE_EVENT_TTL_H`
  (72 h); reads lazily sweep expired rows — no cron.
- **Resolve:** operators (municipality/admin) mark hazards repaired.
- **Push:** WebSocket `/api/live/ws` sends a `hello` snapshot then every mutation
  (`event_upsert` / `event_removed`). Sync REST handlers broadcast via
  `live_manager.broadcast_from_thread()` (never `asyncio.run()` in a route). Clients
  auto-reconnect with capped backoff and fall back to polling `GET /api/live/events`
  every 5 s while the socket is down.

### 5.1 Connected devices & phone drive mode

Users can **pair sensors to their account from the Live page** so detected damage uploads
automatically:

- **Phone drive mode (browser, no app install):** the Live page registers the browser as
  a `phone` device and switches on drive mode — DeviceMotion + GPS. A vertical jolt above
  ~9 m/s² while the car is actually moving (GPS speed gate) auto-posts a `pothole` report
  with confidence/severity mapped from jolt strength, with an 8 s cooldown (the classic
  smartphone-accelerometer pothole-sensing technique). A live readout shows jolt, GPS fix,
  and reports sent. iOS motion permission is requested from the toggle tap.
- **Dashcam / PC pairing:** the Live page issues a single-use 8-character pairing code
  (TTL `LIVE_PAIR_CODE_TTL_MIN`, default 15 min). Running
  `python pipeline/live_pipeline.py --pair <CODE>` exchanges it for a JWT — full CV
  detection then uploads under the owner's account.
- **Device registry:** `live_devices` table (name, kind, device_id, last_seen_at,
  reports_sent). The panel lists devices with live status; **disconnecting revokes** the
  device — further reports from its device_id are rejected with 403. Anonymous
  (unpaired) device ids keep working — pairing is opt-in.

---

## 6. Accounts, roles, and notifications

- **Auth:** JWT (PyJWT HS256, `JWT_SECRET`, TTL `JWT_TTL_H` default 7 days). Passwords:
  stdlib PBKDF2-HMAC-SHA256, 200 000 iterations, per-user salt — no bcrypt dependency.
- **Roles:**
  - `user` (Citizen) — view everything, upload surveys, report/confirm/dispute live
    hazards, connect devices;
  - `municipality` — operator scoped to a chosen city: everything a user can do **plus**
    resolve live hazards, mark detections repaired, delete detections;
  - `admin` — full control incl. user/role management. Admin is granted only by another
    admin; a seed admin is created at startup from `ADMIN_USERNAME/EMAIL/PASSWORD`.
- **Registration** (`/register`): the user chooses **Citizen or Municipality** with role
  cards; municipality accounts must name their city. Login accepts username **or**
  e-mail.
- **Google OAuth** (optional, free): set `GOOGLE_CLIENT_ID`; token verified against
  Google's public tokeninfo endpoint; account auto-created on first login. Apple Sign-In
  is deliberately absent (paid program).
- **Welcome e-mail (free, optional):** when an account is created via e-mail, the backend
  sends a notification through **stdlib smtplib** (`backend/notify.py`) — works with any
  free SMTP relay (Gmail app password recommended). Configured via
  `SMTP_HOST/PORT/USERNAME/PASSWORD/FROM/STARTTLS`; when unset the feature silently
  no-ops. Sending runs in a daemon thread and can never block or fail registration.
- **Frontend session:** JWT in `localStorage['rids_token']`, attached by an axios
  interceptor; any 401 outside `/auth/*` clears the session and redirects to `/login`.
  Every route except `/login` and `/register` is wrapped in `RequireAuth`.

---

## 7. Backend API (FastAPI, all under `/api`)

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/auth/register` | POST | — | Create account (user/municipality) → JWT; sends welcome e-mail |
| `/auth/login` | POST | — | Username-or-email + password → JWT |
| `/auth/oauth/google` | POST | — | Google ID token → JWT (if configured) |
| `/auth/config` | GET | — | Which optional providers are enabled |
| `/auth/me` | GET/PATCH | user | Profile read / update (name, city) |
| `/auth/me/location` | PATCH | user | Record browser geolocation |
| `/auth/users` | GET | admin | List accounts |
| `/auth/users/{id}/role` | PATCH | admin | Change role (city required for municipality) |
| `/detections` | GET | — | Paginated list; filters: type, severity range, dates; sortable |
| `/detections/{id}` | GET | — | Single detection |
| `/detections/nearby` | GET | — | Radius search (geography-cast `ST_DWithin`) |
| `/detections/{id}/status` | PATCH | operator | Mark fixed / not fixed |
| `/detections/bulk` | DELETE | operator | Bulk delete (+ optional survey_log rows) |
| `/stats` | GET | — | Totals, per-type & per-severity breakdowns, averages |
| `/heatmap` | GET | — | `[lat, lon, weight]`, weight = severity·ln(count+1) |
| `/priority-list` | GET | — | Top-N by priority score |
| `/export/csv` | GET | — | Full CSV export |
| `/ingest/upload` | POST | user | Queue a survey job (mp4 + optional gpx) → 202 + job_id |
| `/ingest/status/{job_id}` | GET | — | Live pipeline progress from session.json |
| `/live/reports` | POST | user | Report a sighting (cluster-or-create) |
| `/live/events` | GET | — | Active events (polling fallback) |
| `/live/events/{id}/confirm` | POST | user | "Still there" (distinct-device counted) |
| `/live/events/{id}/dispute` | POST | user | "Not there" (enough → removed) |
| `/live/events/{id}/resolve` | POST | operator | Hazard repaired/cleared |
| `/live/stats` | GET | — | Live counters for the header |
| `/live/ws` | WS | — | Push channel (hello snapshot + mutations) |
| `/live/devices/pair` | POST | user | Register this device (with device_id) or issue a pairing code (without) |
| `/live/devices/claim` | POST | code | Edge agent exchanges pairing code → JWT |
| `/live/devices` | GET | user | My paired devices |
| `/live/devices/{id}` | DELETE | user | Disconnect/revoke a device |
| `/cities/landmarks` | GET | user | Per-city fly-to landmarks (Nominatim, cached forever in `city_landmarks`) |
| `/health`, `/` | GET | — | Health probes |

Interactive docs: `http://localhost:8000/docs` (Swagger) and `/redoc`.

---

## 8. Frontend (React 18 + Vite 5, served by Nginx)

Tech: React Router 6, react-leaflet, Recharts, lucide-react, axios. **One global context
only** (`AuthContext`); everything else is page-local hooks. Cross-page pipeline signal:
`localStorage['rids_active_job']` (set by IngestionPage, polled by MapPage every 10 s).
Styling: inline style objects + CSS design tokens in `index.css`; dark theme default,
`:root.light` overrides; map tiles follow the theme (Carto dark / voyager).

| Page | Route | What you can do |
|---|---|---|
| Home | `/` | Landing dashboard: headline stats, quick links, system health |
| Live | `/live` | Real-time hazard map (WS badge / polling fallback), report damage by tapping the map (10-class picker), vote *Still there / Not there*, operator resolve, live feed with status chips (UNVERIFIED/CONFIRMED/VERIFIED), toasts, **Devices panel: phone drive mode + dashcam pairing + device list/revoke** |
| Map | `/map` | All survey detections: severity/type filters, basemap switcher (dark/streets/satellite), heatmap overlay, landmark fly-to (per-city, Nominatim-backed), detail popups, operator mark-repaired/delete, printable report |
| Statistics | `/stats` | Recharts dashboards: type breakdown, severity distribution, averages, trends |
| Explorer | `/explorer` | Sortable, filterable, paginated table of every detection; CSV download |
| Priority | `/priority` | Ranked repair queue with severity bands, repair-cost sketch (RON heuristics × severity factor), operator actions |
| Ingestion | `/ingest` | Upload mp4 (+ gpx), watch the 7 stages progress live with per-stage labels; job errors surfaced |
| About | `/about` | Project description |
| Login / Register | `/login`, `/register` | Sessions; role choice (Citizen/Municipality) with city; optional Google button |
| Admin | `/admin` | Accounts table, role changes (admin only) |

---

## 9. Database (PostgreSQL 15 + PostGIS)

Schema sources: `db/init/01_schema.sql` (core, fresh volumes), `02_live_schema.sql`,
`03_auth_schema.sql`, `04_live_devices.sql`; on existing volumes the backend's startup
`Base.metadata.create_all` creates missing **tables** (it does not alter existing ones).
`scripts/setup_db.py` is the idempotent manual equivalent.

- `detections` — one row per de-duplicated physical damage: identity, geom(Point,4326) +
  lat/lon, damage_type, confidence, SAM geometry features, depth estimate + confidence,
  severity + confidence, `detection_count`, first/last detected, priority_score,
  survey metadata, `is_fixed`.
- `survey_log` — one row per survey date (frames processed, new/updated counts, status).
- `live_events` / `live_reports` — clustered hazards + full per-device audit trail.
- `live_devices` — paired sensors (owner, device_id, kind, pairing code, last_seen,
  reports_sent, revocation flag).
- `users`, `city_landmarks` — accounts and the landmark cache.

Port nuance: inside Docker the backend reaches `db:5432`; host-run code uses
`localhost:${POSTGRES_PORT}` (default `.env` sets 5433).
**Enrichment is permanently removed** — no street names / weather / road-importance
columns; GPS lat/lon is the only location metadata. (Landmark lookups are UI sugar, not
pipeline enrichment.)

---

## 10. Scalability model

- **Compute scales with the users, not the server.** Every driver runs their own lite
  pipeline instance on their own hardware; the server does zero inference. Cost per
  report ≈ one indexed `ST_DWithin` query.
- **Stateless REST** — all state in PostGIS, so API replicas scale horizontally behind a
  load balancer without session affinity.
- **WS fan-out** is per-process by design; the documented upgrade path (see
  `live_manager.py`) is a Redis/NATS pub/sub relay — handlers already publish through a
  single choke point (`broadcast()`), so nothing else changes. Polling fallback keeps
  working regardless.
- **Survey GPU work** is serialized (one job at a time) on the owner's machine; scale-out
  = more host workers watching separate job directories (or a queue) — the file-based
  contract stays identical.
- **DB growth** is bounded by design: survey upsert collapses re-sightings into
  `detection_count`; live events expire by TTL; dedup keeps one row per physical damage.
- **Free-tier deployment** (see `FREE_DEPLOYMENT.md` / `DEPLOYMENT_GUIDE.md`): static
  frontend on Vercel (`VITE_API_URL` points at the API origin at build time), backend +
  PostGIS on any free VM; Vercel rewrites can't proxy WebSockets, so the Live page
  connects the socket directly to the API origin.

---

## 11. Configuration (single root `.env`, git-ignored)

| Group | Variables |
|---|---|
| Database | `POSTGRES_USER/PASSWORD/DB`, `POSTGRES_PORT` (host side, 5433), `DATABASE_URL` |
| Paths | `PROJECT_DATA_DIR=/app/data` (must equal the compose mount), `PROJECT_ROOT` (leave unset — watcher auto-detects), `WEIGHTS_DIR` |
| Survey pipeline | `PIPELINE_FPS`, `PIPELINE_DEVICE`, `DEDUP_CLUSTER_RADIUS_M`, `SURROUNDING_DENSITY_RADIUS_M`, `DETECTOR_BATCH`, `DETECTOR_HALF`, `DETECTOR_IMGSZ`, `JOB_STALE_TIMEOUT_S`, `WATCHER_POLL_S`, `LOG_LEVEL`, `LOG_FILE` |
| Auth | `JWT_SECRET` (**must be overridden in any real deployment**), `JWT_TTL_H`, `ADMIN_USERNAME/EMAIL/PASSWORD`, `GOOGLE_CLIENT_ID` (optional) |
| E-mail (optional) | `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS` — unset ⇒ e-mail off |
| Live mode | `LIVE_CLUSTER_RADIUS_M` (25), `LIVE_EVENT_TTL_H` (72), `LIVE_CONFIRM_DEVICES` (2), `LIVE_VERIFY_DEVICES` (3), `LIVE_DISPUTE_MIN` (2), `LIVE_PAIR_CODE_TTL_MIN` (15) |
| Edge agent | `LIVE_API_URL`, `RDDS_TOKEN/RDDS_EMAIL/RDDS_PASSWORD` |
| Frontend build | `VITE_API_URL` (only when hosted away from the API) |

Model weights (`ml/weights/`: `best.pt`, `sam2.1_hiera_tiny.pt`, `mono_640x192/`) are not
in git; the Docker stack runs fine without them — only the host pipeline needs them.

---

## 12. Running it

```bash
docker compose up -d --build          # db + backend + frontend  (http://localhost:3000)
python pipeline/job_watcher.py        # host daemon — required to process survey uploads
python pipeline/live_pipeline.py --pair <CODE> --video drive.mp4 --gps drive.gpx
                                      # edge agent paired from the Live page
python pipeline/simulate_fleet.py     # no-GPU live-mode demo
```

Frontend changes are invisible until `docker compose up -d --build frontend` (static
Nginx build). Dev mode without Docker: `uvicorn backend.main:app --reload` +
`cd frontend && npm run dev` (Vite on :3000 proxies `/api`, incl. WebSocket).

Verification: no pytest suite/linter by design — stand-alone scripts
(`scripts/validate_severity.py`, `validate_deduplication.py`, `validate_depth.py`,
`validate_db_write.py`, `validate_nrdd2024.py`) validate each stage against recorded runs.

---

## 13. Layout map

```
backend/    FastAPI app (main, auth, notify, live_manager, models*, schemas*, routes/*)
pipeline/   7 survey stages + orchestrator + job_watcher + lite pipeline + fleet sim
frontend/   React SPA (pages/, components/, context/AuthContext, utils/, hooks/)
db/init/    01 core · 02 live · 03 auth · 04 live_devices  (fresh-volume schema)
ml/         training/research only — never imported by the backend
scripts/    setup + validation utilities (research tools, some hardcode paths)
scheduler/  optional APScheduler nightly survey job
docs/       LIVE_MODE.md · FUNCTIONALITY.md (this file) · deployment guides
```
