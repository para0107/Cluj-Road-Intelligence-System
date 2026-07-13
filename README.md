# RDDS — Road Degradation Detection System

> **Automated urban road-damage detection, classification, prioritization, and real-time
> crowd-sensing from dashcam footage — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%20%2F%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%20%2B%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20%2B%20PostGIS-15--3.3-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Open-source road monitoring platform · Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Table of contents

- [Overview](#overview)
- [Motivation](#motivation)
- [System architecture](#system-architecture)
- [The two-context execution model](#the-two-context-execution-model)
- [Job lifecycle](#job-lifecycle-end-to-end)
- [The 7-stage survey pipeline](#the-7-stage-survey-pipeline)
- [Severity model](#severity-model)
- [Live mode — crowd-validated hazards](#live-mode--crowd-validated-hazards)
- [The lite pipeline — per-user edge agent](#the-lite-pipeline--per-user-edge-agent)
- [Connected devices & phone drive mode](#connected-devices--phone-drive-mode)
- [Citizen gamification](#citizen-gamification--points-badges-streaks-and-leaderboards)
- [Municipality workflow](#municipality-workflow--triage-work-orders-repair-verification)
- [Road Quality Index (RQI)](#road-quality-index-rqi--municipal-health-scoring)
- [Public developer API](#public-developer-api--free-rate-limited-no-auth-overhead)
- [In-browser assistant](#in-browser-assistant--rag--guardrails-zero-server-cost)
- [Security hardening & scalability](#security-hardening--scalability)
- [Accounts, roles & notifications](#accounts-roles--notifications)
- [Database schema](#database-schema)
- [REST API](#rest-api)
- [Frontend](#frontend)
- [Scalability model](#scalability-model)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Running a survey](#running-a-survey)
- [Running the live stack](#running-the-live-stack)
- [Deployment (free)](#deployment-free)
- [Validation utilities](#validation-utilities)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [License & attribution](#license--attribution)

---

## Overview

RDDS is an end-to-end urban infrastructure monitoring platform with **two complementary
workflows**:

**1 — Survey mode (deep analysis).** A raw `.mp4` drive video (optionally with a `.gpx`
GPS log) is pushed through a seven-stage computer-vision pipeline that:

1. extracts frames and synchronises each one to a GPS coordinate,
2. detects 10 classes of road damage and road features with a fine-tuned **RT-DETR-L** model,
3. segments each detection with **SAM 2.1 Tiny** to derive geometry features,
4. estimates relative depth with **Monodepth2**,
5. assigns a rule-based **S1–S5 severity** score,
6. spatially de-duplicates repeated sightings of the same physical damage with **DBSCAN**, and
7. upserts the surviving detections into a **PostGIS** spatial database with a computed priority score.

**2 — Live mode (real-time crowd-sensing, Waze-style).** Every user is a sensor: hazards
reported by phones, dashcam edge agents, or manual map taps are clustered server-side into
events, appear on everyone's map within milliseconds over a WebSocket, and escalate
**UNVERIFIED → CONFIRMED → VERIFIED** as independent devices re-sight them. Users can pair
their **phone (drive mode — motion-sensor pothole detection in the browser)** or a
**dashcam PC (the lite RT-DETR pipeline)** to their account so detected damage uploads
automatically. Stale hazards expire on their own; operators resolve repaired ones.

Around those two workflows sits a complete **product layer**: citizen gamification
(points, badges, streaks, leaderboards — farm-resistant by design), a municipality
suite (triage → work orders → route planning → repair → verification with a
reopened-damage guard), a Road Quality Index heatmap, a free public developer API,
and an in-browser AI assistant with anti-hallucination guardrails.

Everything is served through a **FastAPI** REST + WebSocket API and visualised in a
**React + Leaflet** SPA: live hazard map, survey detection map, statistics dashboards,
tabular explorer, ranked repair queue with budget planning, triage inbox, work-order
board, quality map, ingestion tracker, developer portal, and an admin console. Access
is controlled by **JWT auth with three roles** (citizen / municipality operator / admin).

**Zero-cost by construction:** open-source models (including the assistant's LLM, which
runs in the visitor's own browser), free map tiles, OpenStreetMap/Nominatim geocoding,
Brevo HTTPS or stdlib-SMTP e-mail, optional free Google OAuth, self-hosted proof-of-work
captcha — no paid API anywhere.

---

## Motivation

Romania has one of the highest road-accident rates in the European Union, and deteriorating
urban road infrastructure is a significant contributing factor. Traditional condition surveys
in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective; a full
city-wide survey can take months.

RDDS proposes an automated alternative that any municipality can adopt with minimal hardware
investment. Footage collected during normal vehicle operation is processed automatically into
a continuously updated georeferenced damage map with severity scores and a ranked repair
list — and the live mode turns every participating driver into a real-time road sensor at
zero marginal server cost.

---

## System architecture

| Layer | Runs in | Responsibility |
|-------|---------|----------------|
| **Frontend** | Docker (Nginx) | React SPA — live map, survey map, stats, explorer, priority, ingest, admin |
| **Backend API** | Docker (Uvicorn) | FastAPI REST + WebSocket + job hand-off + auth; **no ML, no GPU** |
| **Database** | Docker (PostGIS) | Spatial store: detections, survey log, live events/reports, users, devices |
| **Survey pipeline** | Windows host (GPU) | The 7-stage CV pipeline (PyTorch + CUDA) |
| **Edge agents** | Each user's hardware | Lite pipeline / phone drive mode — real-time detection, POSTs to the API |

```
┌──────────────────────────── Docker (Linux) ───────────────────────────────┐
│  db        postgis/postgis:15-3.3        cluj_monitor_db                  │
│  backend   FastAPI + Uvicorn (WS)        cluj_monitor_backend             │
│  frontend  React build served by Nginx   cluj_monitor_frontend            │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │ shared bind mount  ./data ⇄ /app/data
┌───────────────────────────────┴──────────────────────────┐
│  Windows HOST (GPU)                                       │
│  pipeline/job_watcher.py    daemon, watches data/jobs/    │
│  pipeline/orchestrator.py   spawned per job, --device cuda│
│  RTX 2050 · .venv · PyTorch+CUDA · ml/weights/            │
└───────────────────────────────────────────────────────────┘
        ▲ HTTPS/WS (JWT)                    ▲ HTTPS (JWT)
┌───────┴───────────────┐   ┌───────────────┴───────────────────────────────┐
│ Phones (drive mode in │   │ Vehicle PCs: pipeline/live_pipeline.py        │
│ the browser: motion + │   │ same RT-DETR weights, fp16/ONNX, motion gate, │
│ GPS auto-reports)     │   │ lite severity → POST /api/live/reports        │
└───────────────────────┘   └───────────────────────────────────────────────┘
```

---

## The two-context execution model

This is the single most important thing to understand about RDDS.

The backend container **cannot** run the ML pipeline: it has no GPU, no PyTorch/Ultralytics/
SAM/Monodepth2 in its image (`backend/requirements.backend.txt` is runtime-only), and no
access to the Windows weight paths. So the backend never executes the survey pipeline
directly. Instead it **drops a job file** into the shared `data/jobs/` directory; a
host-side watcher (`job_watcher.py`) picks it up and runs the orchestrator on the GPU.

The two sides communicate **only through files under `data/`** — there is no socket, queue,
or RPC between them. This boundary is intentional and load-bearing:

- The backend writes container-side paths (`/app/data/...`) into the job file.
- `job_watcher._container_to_host()` rewrites the `/app/data` prefix to the host `data/`
  path before invoking the orchestrator.
- The orchestrator writes `session.json`, which the backend reads back to report progress.

(Live mode is different by design: edge agents talk to the API directly over HTTP/JWT —
the server does no inference for them, only storage and fan-out.)

---

## Job lifecycle (end-to-end)

```
Frontend IngestionPage
   │ POST /api/ingest/upload   (multipart: video .mp4 [+ gps .gpx], JWT required)
   ▼
backend/routes/ingest.py
   • saves video → data/raw/footage/<job_id>_<name>.mp4
   • saves gps   → data/raw/gps_logs/<job_id>_<name>.gpx   (optional)
   • writes      → data/jobs/<job_id>.json   {status:"pending", *_path_container}
   • returns 202 {job_id, status:"pending"}        (409 if a job is already active)
   ▼
pipeline/job_watcher.py  (HOST, polls every WATCHER_POLL_S seconds)
   • finds a status=="pending" job
   • translates container paths → host paths
   • sets job status → "running", then subprocess:
        python pipeline/orchestrator.py --video … --session_id <job_id>
              --work_dir data/processed/sessions --device cuda --fps N [--gps …] --save_debug
   • on exit 0 → job "complete", else → "failed"
   ▼
pipeline/orchestrator.py  (HOST, on GPU)
   • runs the 7 stages, writing per-stage artifacts under
        data/processed/sessions/<job_id>/0X_*/…
   • writes session.json after EVERY stage (live progress) and at the end
   ▼
Frontend polls GET /api/ingest/status/<job_id> every 10 s
   • backend returns session.json (authoritative) or falls back to the job file
   • IngestionPage renders the per-stage PipelineTracker
   • MapPage silently refreshes detections while a job is running
```

`job_id` and `session_id` are the **same value**, format `YYYYMMDD_HHMMSS` (UTC), generated
by the backend and reused as the orchestrator session id so every path lines up.

Status values: `pending` → `initialising` → `running` → `complete` | `failed`.
One job at a time — the upload route returns **409** while a job is pending/running; a job
silent for `JOB_STALE_TIMEOUT_S` (default 2 h) is treated as stale and stops blocking.

---

## The 7-stage survey pipeline

The canonical order and stage keys are defined by `orchestrator.run()`. A stage key must
stay in sync in three places: `pipeline/orchestrator.py`, the `stages` array in
session.json (consumed via `backend/routes/ingest.py`), and the stage maps in
`frontend/src/pages/IngestionPage.jsx`.

| # | Stage key | Module | What it does |
|---|-----------|--------|--------------|
| 1 | `preprocessor` | `preprocessor.py` | Extract frames (`PIPELINE_FPS`, default 2/s), sync GPS from `.gpx` (timestamp interpolation), solar elevation (pysolar), lighting class (HSV) |
| 2 | `detector` | `detector.py` | **RT-DETR-L** inference (N-RDD2024 fine-tuned `best.pt`): global conf 0.001, then per-class thresholds (0.35 all classes, 0.50 `lane_line_blur`); **batched** (`DETECTOR_BATCH`, auto 8 on CUDA) + **fp16** (`DETECTOR_HALF`, auto on CUDA); `DETECTOR_IMGSZ` overrides input size; TTA disabled (measured zero gain) |
| 3 | `segmentor` | `segmentor.py` | **SAM 2.1 Tiny** box-prompted masks + 4 geometry features (area, edge sharpness, interior contrast, compactness) |
| 4 | `depth_estimator` | `depth_estimator.py` | **Monodepth2** (mono_640x192) relative disparity → depth proxy, geometry-proxy fallback, low-light skip |
| 5 | `severity_classifier` | `severity_classifier.py` | Rule-based **S1–S5** weighted multi-signal score (below) |
| 6 | `deduplicator` | `deduplicator.py` | **DBSCAN** + Haversine clustering (eps `DEDUP_CLUSTER_RADIUS_M`, default 2 m); keeps the highest-severity member; computes `surrounding_density` (`SURROUNDING_DENSITY_RADIUS_M`, 50 m); HTML before/after report |
| 7 | `db_writer` | `db_writer.py` | **PostGIS** upsert + priority-score update |

**10 damage / feature classes** (N-RDD2024 schema, Kaya & Codur 2024): `longitudinal_crack
(D00)`, `transverse_crack (D10)`, `alligator_crack (D20)`, `repaired_crack (D30)`,
`pothole (D40)`, `pedestrian_crossing_blur (D50)`, `lane_line_blur (D60)`,
`manhole_cover (D70)`, `patchy_road (D80)`, `rutting (D90)`.

### GPS guard — GPS-less runs are valid

Stages 6 and 7 require coordinates. If no `.gpx` is supplied and embedded-GPS extraction
fails, frames have `latitude/longitude = None`; the orchestrator **skips stages 6 & 7 and
still finishes `complete`** (nothing is written to the DB). Detection / segmentation /
depth / severity artifacts are still produced. A GPS-less run is **not** a failure.

### Per-stage artifacts

```
data/processed/sessions/<job_id>/
    01_manifest/        manifest.json   (+ frames/)
    02_detections/      detections.json (+ debug/)
    03_segmentations/   segmentations.json (+ debug/)
    04_depth/           depth_estimates.json (+ debug/)
    05_severity/        severity_estimates.json
    06_deduplicated/    deduplicated.json + dedup_report.html
    07_db_write/        db_write_summary.json
    session.json        ← live status source of truth
```

Frontend-triggered runs force `--save_debug` (via `_SAVE_DEBUG` in `job_watcher.py`),
writing overlay images under each `0X_*/debug/`. Non-detection frames are pruned from disk
after Stage 2 (`--keep_all_frames` to disable).

### Enrichment is permanently removed

An earlier "Enricher" stage (Nominatim / OSM Overpass / Open-Meteo, plus columns like
`street_name`, `road_importance`, `weather`) was intentionally dropped from the pipeline,
the ORM, the schemas, and the DB. **GPS lat/lon is the only location metadata stored.**
(The per-city landmark lookup used by the map's fly-to menu is UI sugar, not pipeline
enrichment — see [Accounts, roles & notifications](#accounts-roles--notifications).)

---

## Severity model

Severity is a transparent, rule-based score in `[0, 1]` mapped to five levels. It combines
four normalised signals with **per-class signal weights**, then applies a
**class-importance weight**:

```
S_depth     = depth_norm                              (Monodepth2 relative depth)
S_area      = min(surface_area_px / 1000, 1)          (SAM mask area)
S_contrast  = min(interior_contrast / 2.0, 1)         (SAM interior contrast)
S_sharpness = min(edge_sharpness / 60.0, 1)           (SAM edge sharpness)

raw_score      = Σ  w_signal · S_signal               (per-class signal weights, sum to 1)
severity_score = min(raw_score · class_weight · 2, 1) (class importance)
```

| Score range | Level | Action |
|-------------|-------|--------|
| `[0.00, 0.15)` | **S1** | Monitor |
| `[0.15, 0.35)` | **S2** | Schedule maintenance |
| `[0.35, 0.55)` | **S3** | Priority repair |
| `[0.55, 0.75)` | **S4** | Urgent repair |
| `[0.75, 1.00]` | **S5** | Emergency closure |

Per-class signal weights encode the physics: pothole severity is dominated by
depth + contrast (bowl shape); alligator crack by area (fatigue-network spread);
longitudinal crack by sharpness (active vs healed boundary). Marking classes
(`lane_line_blur`, `pedestrian_crossing_blur`, `class_weight = 0.10`) can never exceed
**S2** by construction — no special-case branches. A `severity_confidence` value reflects
measurement quality and is reduced when the geometry depth-proxy substituted Monodepth2.

**Priority score** (`Detection.compute_priority_score`, no enrichment inputs):
`priority = severity · ln(detection_count + 1)` — recurrence amplifies severity.

---

## Live mode — crowd-validated hazards

Waze-style real-time layer, coexisting with Survey mode (the full design
notes live in the project's internal docs).

- **Reporting** — `POST /api/live/reports` (JWT) clusters a sighting into the nearest
  active event of the same `damage_type` within `LIVE_CLUSTER_RADIUS_M` (default 25 m,
  PostGIS `ST_DWithin` on geography), or creates a new event. Sources: manual map taps on
  the Live page, phone drive mode, lite-pipeline edge agents, the fleet simulator.
- **Validation by distinct device** — 1 device → `unverified`;
  ≥ `LIVE_CONFIRM_DEVICES` (2) → `confirmed`; ≥ `LIVE_VERIFY_DEVICES` (3) → `verified`.
  A device confirming twice is idempotent — no self-boosting.
- **Dispute** — enough independent "not there" votes
  (≥ max(`LIVE_DISPUTE_MIN`, reporter count)) deactivate the event.
- **TTL expiry** — every supporting signal pushes `expires_at` forward by
  `LIVE_EVENT_TTL_H` (72 h); reads lazily sweep expired rows. No cron.
- **Resolve** — operators (municipality/admin) mark hazards repaired.
- **Push channel** — WebSocket `/api/live/ws`: `hello` snapshot on connect, then every
  mutation (`event_upsert` / `event_removed`). Sync REST handlers broadcast via
  `live_manager.broadcast_from_thread()` (never `asyncio.run()` in a route). Nginx has the
  Upgrade block; the Vite dev proxy sets `ws: true`. Clients auto-reconnect with capped
  exponential backoff and fall back to polling `GET /api/live/events` every 5 s.

All live state lives in PostGIS (`live_events` + `live_reports` audit trail), so REST
handlers are stateless; only the WS fan-out is per-process (Redis/NATS pub/sub is the
documented upgrade path in `backend/live_manager.py`).

---

## The lite pipeline — per-user edge agent

`pipeline/live_pipeline.py` — one instance per vehicle/user, running on **their** hardware.
Zero cloud cost: the shared backend only stores and fans out results.

```
dashcam/webcam ─► frame stride (--every, def. 10) ─► motion gate (64×36 gray diff —
parked / red-light frames skip inference entirely) ─► RT-DETR (same best.pt, fp16,
--imgsz 480) ─► same per-class confidence thresholds (imported from pipeline/detector.py)
─► lite severity: the UNMODIFIED stage-5 formula fed by ~0.4 ms CV proxies
(Otsu mask area, Sobel boundary sharpness, interior-vs-ring contrast, depth geometry
proxy — pipeline/lite_severity.py) ─► local dedup (--min-gap-m 20, --cooldown-s 30)
─► POST /api/live/reports (JWT)
```

- **Identical detection behaviour** to survey stage 2 — same checkpoint, same thresholds —
  just on fewer, well-chosen frames; lite severity stays on the calibrated S1–S5 scale
  with an honest ~0.5 `severity_confidence`.
- **CPU-only users:** one-time free ONNX export — `--export-onnx [--quantize]` — then run
  with `--weights ml/weights/best.onnx --device cpu`.
- **Sources:** `--video file.mp4` (replay; `--realtime` for native speed) or `--camera 0`;
  position from `--gps track.gpx` (time-interpolated) or fixed `--lat/--lon`.
- **Auth:** `--pair <CODE>` (pairing code from the Live page → saves a JWT to
  `~/.rids_live_token`; no password ever typed on the car PC), or `--token`, or
  `--email/--password` (env `RDDS_TOKEN` / `RDDS_EMAIL` / `RDDS_PASSWORD`).
- **Identity:** stable per-machine device id persisted in `~/.rids_device_id`.
- `pipeline/simulate_fleet.py` — multi-vehicle live-mode demo, no GPU needed.

---

## Connected devices & phone drive mode

Users pair sensors to their account **from the Live page (Devices panel)** so detected
damage uploads automatically:

- **Phone drive mode (browser, no app install):** the Live page registers the browser as a
  `phone` device and turns on drive mode — DeviceMotion + geolocation
  (`frontend/src/utils/driveMode.js`). A linear-acceleration jolt above ~9 m/s² while GPS
  says the car is actually moving auto-posts a `pothole` report with confidence/severity
  mapped from jolt strength (8 s cooldown; the classic smartphone-accelerometer
  pothole-sensing technique). A live readout shows jolt / GPS fix / reports sent. iOS
  motion permission is requested from the toggle tap.
- **Dashcam / PC pairing:** the panel issues a single-use 8-character pairing code
  (TTL `LIVE_PAIR_CODE_TTL_MIN`, default 15 min). On the vehicle machine:
  `python pipeline/live_pipeline.py --pair <CODE>` exchanges it for a JWT; the full lite
  CV pipeline then uploads under the owner's account.
- **Device registry:** `live_devices` table — name, kind (`phone|dashcam|browser|
  simulator`), device id, `last_seen_at`, `reports_sent`. The panel lists devices with
  live status; **disconnecting revokes** the device (further reports from its device id
  are rejected with 403). Unpaired/anonymous device ids keep working — pairing is opt-in.

---

## Citizen gamification — points, badges, streaks, and leaderboards

Every citizen who reports hazards earns **points, badges and streaks** on outcomes they can't
manufacture alone — preventing farm attacks:

- **Points** — awarded only on community-validated outcomes:
  - Event confirmed (≥2 devices report it) → **+10 points** to every distinct reporter
  - Event verified (≥3 devices) → **+15 more points**
  - Hazard fixed by a municipality → **+25 points** to supporters
  - Promoted by an operator (survey + live) → **+20 points** to supporters
  - A raw report earns **zero points** (anti-spam) — the server-side report cooldown is
    1 per 15 seconds + 40 per day per user
- **Badges**: first report, road scout (10 confirmed), road guardian (50 confirmed),
  triple-checked (one verified), week streak, fixer (one fixed), city changer (five fixed),
  night watch (after 22:00 local time)
- **Leaderboard**: global + city-scoped, ranked by total points; **distinct-device
  escalation** in the live-validation engine ensures a single attacker cannot farm
  (confirm/dispute votes by the same device count once; escalation requires independent
  devices)
- **Notifications bell** and in-app notification history tied to each citizen

---

## Municipality workflow — triage, work orders, repair verification

The revenue-generating suite for municipal operations:

- **Triage** (`/triage` page): operators review live crowd reports sorted by validation
  tier (verified first), **promote** a report to a real detection with a single click
  (creates a Detection row, awards +20 points to all supporters, removes from triage inbox),
  or **dismiss** false alarms
- **Work orders** (`/workorders` page): group detections into repair jobs with crew name,
  scheduled/due dates, and cost estimates
  - Status flow: `open` → `scheduled` → `in_progress` → `repaired` → `verified`
  - **Repair verification guard**: when transitioning to `verified`, the system refuses
    (409) if any detection shows `last_detected > fixed_at` (pipeline saw it again after
    repair was signed off — **reopened**). Operators must resolve the reopened item or
    reschedule. This is load-bearing against false repair claims.
  - **Route optimization**: client-side haversine + nearest-neighbor + 2-opt, outputs total
    km and estimated time; print route sheets with numbered stops and leg distances
- **Repair indicators on the map**: detections show when they were fixed (fixed_at timestamp);
  evidence photos (cropped from pipeline frames, JPG) are stored and visible in the
  detection drawer (operator-only)
- **Analytics dashboard** (`/stats` → Operations section): average time-to-repair, open
  backlog by severity, weekly new-vs-repaired trend, work-order budget rollup,
  reopened-count alert

---

## Road Quality Index (RQI) — municipal health scoring

`/quality` page: a heatmap grid of ~120 m cells (configurable 40–1000 m), where each cell
is scored based on the detected damage within it:

```
score = round(100 * exp(-penalty / 8))
penalty = Σ severity_weight[S1..S5] × exp(-days_since_detection / 180) × (0.2 if fixed)
Bands: A ≥85 (good) | B ≥70 | C ≥50 | D ≥30 | E <30 (poor)
```

- **Grid is stateless**: pure lat/lon math, no new geometry columns (avoids PostGIS index
  traps)
- **Decaying penalty**: fresher damage has higher weight; repaired damage is counted but
  de-weighted (0.2)
- **Exportable**: CSV and GeoJSON for import into GIS tools
- Useful for city planning meetings and work-order prioritisation

---

## Public developer API — free, rate-limited, no auth overhead

`/developers` page docs + `GET /v1/public/*` endpoints. Developers sign up and create API
keys (`rdds_<40-char-token>`), stored as SHA-256 + prefix only (plaintext shown once on
creation).

- **Endpoints** (all GET, read-only):
  - `/v1/public/detections` — paginated survey detections (bbox, damage_type filter, never
    expose crop_path)
  - `/v1/public/road-quality` — grid data like `/quality/grid`
  - `/v1/public/stats` — total/open/fixed counts, by damage type, by severity
- **Rate limits**: per-key (default 60 req/min, configurable per key); 429 responses include
  `Retry-After` header
- **3-key cap** per user to prevent runaway usage

---

## In-browser assistant — RAG + guardrails, zero server cost

`/assistant` page: a hybrid knowledge bot running **entirely in the browser**, never sending
user questions to a remote LLM:

- **Instant mode** (default, always available):
  - Keyword matching for common questions (live hazard count, my points, leaderboard, etc.)
  - Sparse BM25 search (MiniSearch) over a curated knowledge base
  - Returns top result + related links; no model inference

- **AI mode** (opt-in):
  - Download consent modal (900 MB Llama-3.2-1B model via WebLLM)
  - Explicit state machine: guard input → check intents → retrieve docs → [confident?]
    → conditional HyDE re-search → generate → lexical grounding → respond
  - **Guardrails** (all client-side, no judge model):
    - Injection pattern detection (ignore/forget/jailbreak/DAN/etc.)
    - On-topic gate (RDDS/road keywords only; off-topic questions get polite deflection)
    - Per-sentence lexical grounding against retrieved context (drops unsupported claims)
    - Number grounding (any digit in the answer must appear in context or API data)
  - Streaming tokens to UI with stage labels (thinking / writing / checking)

- **Knowledge base**: 33 entries covering reporting, points, severity bands, municipality
  workflow, API, privacy, pricing, navigation
- **Efficiency**: KB embeddings pre-computed at build time; query embedding is fast (tens of ms);
  HyDE only triggers on low-confidence searches

---

## Security hardening & scalability

Key defenses:

- **Proof-of-work CAPTCHA** (ALTCHA): self-hosted, stdlib HMAC-SHA256, no vendor
  involvement. Enabled on login/register/contact forms; browser solves SHA-256 puzzle
  before POST. With CAPTCHA_ENABLED off (dev), forms work normally.
- **Rate limiting**: centralized in-process limiter with configurable budgets per endpoint:
  - Login: 5 failures / 15 min → 15-min lockout
  - Register: 5 / hour / IP
  - Live reports: 1 / 15 sec + 40 / day per user
  - API keys, device pairing, contact forms: configurable per-route
- **Security headers** (Nginx + FastAPI middleware):
  - CSP: `script-src 'self' 'wasm-unsafe-eval'` (WebLLM), `style-src 'unsafe-inline'`,
    `img-src blob: data:`, worker-src blob:
  - X-Content-Type-Options: nosniff, X-Frame-Options: DENY, Referrer-Policy, Permissions-Policy
  - gzip compression on responses (js/css/json/svg)
- **WebSocket hardening**: origin check when CORS_ORIGINS set; per-IP connection cap (4);
  total cap (2000); message-rate cap (30/min)
- **Database binding**: DB and API listen on 127.0.0.1 loopback only (Docker + host both);
  direct exposure requires explicit port mapping
- **One Uvicorn worker**: all in-process state (rate limiters, caches, WS fan-out) stays
  single-threaded; Redis pub/sub is the documented multi-worker upgrade path

---

## Accounts, roles & notifications

- **Auth:** JWT (PyJWT HS256, `JWT_SECRET`, TTL `JWT_TTL_H`, default 7 days). Passwords:
  stdlib **PBKDF2-HMAC-SHA256**, 600 000 iterations, per-user salt — no bcrypt/passlib
  dependency. `backend/auth.py` provides the `get_current_user` / `require_operator` /
  `require_admin` dependencies. **No default credentials exist in the repo**: the seed
  admin is created only when `ADMIN_USERNAME/EMAIL/PASSWORD` are set in `.env`, and an
  unset `JWT_SECRET` produces a random per-process secret (sessions reset on restart)
  plus a loud warning. Login is **rate limited** per identifier+IP (5 failures / 15 min
  → 15 min lockout).
- **Roles & page visibility:**
  | Role | Pages | Powers |
  |------|-------|--------|
  | `user` (Citizen) | **Command, Live, Impact, Assistant, System** | Report/confirm/dispute hazards, connect devices, earn points/badges, view leaderboard |
  | `municipality` | All pages (adds Map, Explorer, Stats, Triage, Repairs, Work Orders, Quality, Upload) | Everything a user can do **plus**: triage live reports, create/manage work orders, verify repairs, view RQI, access public API |
  | `admin` | All pages + Admin | Full control: roles, enable/disable/delete accounts, approve registrations, manage API keys |
  The same split is enforced server-side: `/detections*`, `/heatmap`, `/priority-list`,
  `/export/csv`, `/ingest/*`, `/triage`, `/work-orders`, `/quality` require the operator
  role; `/stats`, `/engagement`, `/notifications` are any authed user.
- **Notifications** (`/notifications` endpoint): in-app alerts on points earned, badges
  awarded, hazards fixed, promotions; polled every 60 s or pushed via bell dropdown; each
  notification can link to a relevant page
- **Registration with e-mail verification** (`/register`): the user picks **Citizen or
  Municipality** via role cards (municipality must name its city). Nothing is created
  yet — a **6-digit confirmation code** is e-mailed (30 min TTL, re-send with a 60 s
  gap) and held in `pending_registrations`. On a correct code, citizen accounts are
  created and signed in immediately; **municipality registrations additionally wait for
  admin approval** — every admin is e-mailed, the Admin page shows an approvals queue,
  and at least one admin must approve (or deny) before the account exists. The
  applicant is notified of the decision by e-mail. When SMTP is unconfigured (dev), the
  code step is skipped gracefully; the municipality approval step still applies.
- **Account lifecycle:** any signed-in user can **delete their own account** from the
  navbar profile menu (local accounts re-type their password). Admins can change roles,
  **enable/disable**, and **delete** accounts from the Admin page. The last active
  admin can neither demote nor delete itself.
- **E-mails (free, optional):** all notifications go through **stdlib smtplib**
  (`backend/notify.py`) over any free SMTP relay (Gmail app password recommended) —
  verification codes, welcome mail, municipality approval requests/decisions, and a
  **thank-you e-mail when a user reports damage** in Live mode (throttled to one per
  user per day). Configure `SMTP_HOST/PORT/USERNAME/PASSWORD/FROM/STARTTLS`; when unset
  every e-mail feature silently no-ops. Sends run in a daemon thread and can never
  block or fail a request.
- **Google sign-in** (optional, free): set `GOOGLE_CLIENT_ID` and the login page
  renders a real **“Sign in with Google”** button (Google Identity Services). The ID
  token is verified against Google's public tokeninfo endpoint and the account is
  auto-created on first login — no code step needed (Google already verified the
  e-mail). **Apple Sign-In is deliberately absent** (paid Apple Developer program —
  violates the zero-cost rule).
- **City landmarks:** `GET /api/cities/landmarks` powers the map's fly-to menu for any
  city — free Nominatim lookups, rate-limited ≥1 s apart, **cached forever** in
  `city_landmarks` (one lookup per city, ever), with a built-in offline fallback for
  Cluj-Napoca. This is *not* the removed pipeline enrichment.
- **Frontend session:** JWT in `localStorage['rids_token']`, attached by an axios
  interceptor (`utils/api.js`); any 401 outside `/auth/*` clears the session and redirects
  to `/login`. Every route except `/login` and `/register` is wrapped in `RequireAuth`.
  After login the browser's (permission-based) geolocation is pushed once so the map can
  open on the user's city.

---

## Database schema

PostgreSQL 15 + PostGIS 3.3. Fresh volumes bootstrap from `db/init/` (executed in order by
the postgres entrypoint); on existing volumes the backend's startup
`Base.metadata.create_all` creates missing **tables** (it never alters existing ones —
keep the SQL files and ORM models in sync). `scripts/setup_db.py` is the idempotent manual
equivalent for the core schema.

| File | Tables |
|------|--------|
| `01_schema.sql` | `detections`, `survey_log` (+ extensions, indexes, trigger) |
| `02_live_schema.sql` | `live_events`, `live_reports` |
| `03_auth_schema.sql` | `users`, `city_landmarks` |
| `04_live_devices.sql` | `live_devices` |
| `05_pending_registrations.sql` | `pending_registrations` (e-mail verification + municipality approval queue) |

### `detections` — one row per de-duplicated damage instance

`id` (UUID), `geom` (POINT/4326), `latitude`, `longitude`, `damage_type`, `confidence`,
`frame_path`, segmentation geometry (`surface_area_cm2`, `edge_sharpness`,
`interior_contrast`, `mask_compactness`), `depth_estimate_cm`, `depth_confidence`,
`lighting_condition`, `severity` (1–5), `severity_confidence`, `surrounding_density`,
temporal fields (`first_detected`, `last_detected`, `detection_count`,
`deterioration_rate`), `priority_score`, `survey_date`, `survey_video_file`, `is_fixed`.

Indexes: GIST on `geom`, plus `severity`, `damage_type`, `survey_date`,
`priority_score DESC`. The upsert matches an existing detection of the same `damage_type`
within `DEDUP_CLUSTER_RADIUS_M` metres (`ST_DWithin` on `geography`) and bumps its
`detection_count` / `deterioration_rate` instead of inserting a duplicate.

### `survey_log` — one row per `survey_date` (unique)

`status`, `started_at`/`finished_at`, `frames_processed`, `detections_found`,
`new_detections`, `updated_detections`, `error_message`, `video_files` (JSONB).

### Live & auth tables

- `live_events` — one row per clustered, community-validated hazard: position
  (geom + lat/lon), `damage_type`, `max_confidence`, max `severity`, validation state
  (`status`, `report_count`, `reporter_devices`, `dispute_devices`, `is_active`,
  `resolved`), timestamps (`first_reported`, `last_reported`, `expires_at`).
- `live_reports` — one row per raw device signal (`sighting | confirm | dispute`) with
  device id, position, confidence, severity, note — the audit trail behind
  distinct-device counting.
- `live_devices` — paired sensors: owner (`user_id` FK), `device_id`, `name`, `kind`,
  single-use `pair_code` (+ expiry), `is_active` (revocation), `last_seen_at`,
  `reports_sent`.
- `users` — account identity, PBKDF2 `password_hash`, `role`, `city`, last known
  position, `auth_provider` (`local | google`), `is_active`, `last_login_at`.
- `city_landmarks` — permanent cache of Nominatim landmark lookups per city.

> **GeoAlchemy2 gotcha:** never add an explicit `Index(..., "geom")` to a model — the
> Geometry column auto-creates `idx_<table>_<col>`, and the name collision silently breaks
> `create_all`.

### Port nuance (host vs container)

- **Inside Docker** the backend reaches the DB by service name: `…@db:5432/…`.
- **On the host** the DB publishes `${POSTGRES_PORT:-5432}:5432` (`.env` sets `5433`), so
  host-run code (pipeline, `db_writer`, dev backend) connects via `localhost:5433`.

---

## REST API

All routes are under `/api` (Nginx proxies `/api/` → `backend:8000`, with a dedicated
WebSocket-upgrade block for `/api/live/ws`; the Vite dev server proxies the same).
Interactive docs: **http://localhost:8000/docs** (Swagger) · `/redoc`.

### Auth & accounts

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/api/auth/register` | — | Start a registration (**user or municipality**) → e-mails a confirmation code (`verify_email`), or account+JWT when SMTP is off |
| `POST` | `/api/auth/verify-email` | — | Confirm the code → user: account+JWT; municipality: `awaiting_approval` |
| `POST` | `/api/auth/resend-code` | — | Re-send the confirmation code (60 s throttle) |
| `POST` | `/api/auth/login` | — | Username-or-e-mail + password → JWT (rate limited; 429 on lockout) |
| `POST` | `/api/auth/oauth/google` | — | Google ID token → JWT (when configured) |
| `GET`  | `/api/auth/config` | — | Which optional providers are enabled (+ Google client id) |
| `GET` / `PATCH` | `/api/auth/me` | user | Profile read / update (name, city) |
| `PATCH`| `/api/auth/me/location` | user | Record browser geolocation |
| `DELETE`| `/api/auth/me` | user | Delete MY account (password re-typed for local accounts) |
| `GET`  | `/api/auth/users` | admin | List accounts |
| `PATCH`| `/api/auth/users/{id}/role` | admin | Change a role (city required for municipality) |
| `PATCH`| `/api/auth/users/{id}/active` | admin | Enable / disable an account |
| `DELETE`| `/api/auth/users/{id}` | admin | Delete an account |
| `GET`  | `/api/auth/registrations/pending` | admin | Municipality approval queue (+ unverified registrations) |
| `POST` | `/api/auth/registrations/{id}/approve` | admin | Approve → account created, applicant e-mailed |
| `POST` | `/api/auth/registrations/{id}/deny` | admin | Deny → registration deleted, applicant e-mailed |

### Survey data & analytics

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET`  | `/api/detections` | operator | Paginated list; filters (type, severity range, dates), sorting |
| `GET`  | `/api/detections/nearby` | operator | Radius search (lat, lon, radius_m, limit) |
| `GET`  | `/api/detections/{id}` | operator | Single detection |
| `PATCH`| `/api/detections/{id}/status` | operator | Set `is_fixed` |
| `DELETE`| `/api/detections/bulk` | operator | Delete by id list (optionally cascade `survey_log`) |
| `GET`  | `/api/stats` | user | Aggregate stats (the Command page shows them to citizens too) |
| `GET`  | `/api/heatmap` | operator | Weighted points (severity·ln(count+1)) for the heat layer |
| `GET`  | `/api/priority-list` | operator | Ranked repair list |
| `GET`  | `/api/export/csv` | operator | CSV export of all detections |
| `POST` | `/api/ingest/upload` | operator | Upload video (+gps), streamed to disk with `MAX_UPLOAD_MB` cap → 202 |
| `GET`  | `/api/ingest/status/{job_id}` | user | Poll job / session status (job id format-validated) |
| `GET`  | `/api/cities/landmarks` | user | Per-city fly-to landmarks (cached Nominatim) |

### Live mode & devices

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/api/live/reports` | user | Report a sighting (cluster-or-create) |
| `GET`  | `/api/live/events` | — | Active events (polling fallback) |
| `POST` | `/api/live/events/{id}/confirm` | user | "Still there" (distinct-device counted) |
| `POST` | `/api/live/events/{id}/dispute` | user | "Not there" (enough votes remove it) |
| `POST` | `/api/live/events/{id}/resolve` | operator | Hazard repaired / cleared |
| `GET`  | `/api/live/stats` | — | Live counters for the UI header |
| `WS`   | `/api/live/ws` | — | Push channel: hello snapshot + every mutation |
| `POST` | `/api/live/devices/pair` | user | Register this device (with `device_id`) **or** issue a pairing code (without) |
| `POST` | `/api/live/devices/claim` | code | Edge agent exchanges a pairing code → JWT |
| `GET`  | `/api/live/devices` | user | List my paired devices |
| `DELETE`| `/api/live/devices/{id}` | user | Disconnect / revoke a device |

### Health

`GET /`, `GET /health`, and the nginx-proxied public probe `GET /api/health`
(used by the navbar dot — works on the login page, before any session exists).

> CORS is currently `allow_origins=["*"]` — tighten before any real deployment.
> The complete security-hardening log is kept in the project's internal docs.

---

## Frontend

React 18 + Vite 5 + React Router 6; Leaflet / react-leaflet for maps, Recharts for charts,
lucide-react icons, axios. **One global context only** — `AuthContext` (session/user);
everything else is page-local hooks. The only cross-page pipeline signal is
`localStorage['rids_active_job']` (set by IngestionPage, polled by MapPage every 10 s).

Citizens see **Command, Live, and System**; the survey/analytics pages (Map, Explorer,
Stats, Repairs, Upload) are visible to **municipality operators and admins only** —
hidden from the navbar, guarded in the router, and enforced by the API.

| Page | Route | Visible to | Purpose |
|------|-------|------------|---------|
| **HomePage (Command)** | `/` | all roles | Landing dashboard: headline stats, quick links, system health |
| **LivePage** | `/live` | all roles | Real-time hazard map: WS badge (polling fallback), tap-to-report with the 10-class picker, *Still there / Not there* voting, operator resolve, live feed with UNVERIFIED/CONFIRMED/VERIFIED chips, toasts, **Devices panel** (phone drive mode, dashcam pairing codes, device list + revoke) |
| **MapPage** | `/map` | operators | Survey detections: severity/type filters, basemap switcher (dark/streets/satellite), heatmap overlay, per-city landmark fly-to, popups, mark-repaired/delete, printable report; silent refresh while a job runs |
| **StatsPage** | `/stats` | operators | City-wide statistics dashboards (Recharts) |
| **ExplorerPage** | `/explorer` | operators | Sortable / filterable / paginated detection table, CSV download |
| **PriorityPage (Repairs)** | `/priority` | operators | Ranked repair queue with severity bands and repair-cost sketches (RON heuristics × severity factor) |
| **IngestionPage (Upload)** | `/ingest` | operators | Upload + per-stage live pipeline tracker |
| **AboutPage (System)** | `/about` | all roles | Project description |
| **LoginPage / RegisterPage** | `/login`, `/register` | public | Sessions; **role choice (Citizen / Municipality)** with city; **e-mail code verification step**; municipality "awaiting approval" screen; **Sign in with Google** button when configured |
| **AdminPage** | `/admin` | admin | Accounts table (roles, enable/disable, delete) + **municipality approvals queue** |

Any signed-in user can delete their own account from the navbar profile menu.

Conventions:

- Every route except `/login` / `/register` is wrapped in `RequireAuth`.
- Styling is inline-`style`-objects + CSS design tokens (`--bg`, `--accent`, …) in
  `index.css`; dark theme default, `:root.light` overrides. No Tailwind/CSS modules.
- Map tiles follow the app theme via `hooks/useTheme.js` (dark → Carto dark, light →
  Carto voyager); MapPage's switcher overrides per-session, LivePage always follows.
- All API calls go through `frontend/src/utils/api.js` (axios, baseURL `/api`, JWT
  interceptor); live-mode helpers in `utils/live.js`; drive-mode sensor logic in
  `utils/driveMode.js`.
- The production frontend is a **static build baked into the Nginx image** — rebuild after
  any `src/` change: `docker compose up -d --build frontend`.

---

## Scalability model

- **Compute scales with the users, not the server.** Every driver runs their own lite
  pipeline / drive mode on their own hardware; the server does zero inference. Cost per
  live report ≈ one indexed `ST_DWithin` query.
- **Stateless REST** — all state in PostGIS, so API replicas scale horizontally without
  session affinity.
- **WS fan-out** is per-process by design; the documented upgrade path
  (`backend/live_manager.py`) is a Redis/NATS pub/sub relay — all broadcasts already go
  through one choke point. The polling fallback works regardless.
- **Survey GPU work** is serialized (one job at a time) on the owner's machine; scale-out
  means more host workers — the file-based contract stays identical.
- **DB growth is bounded:** survey upsert collapses re-sightings into `detection_count`;
  live events expire by TTL; dedup keeps one row per physical damage.

---

## Repository layout

```
backend/                 FastAPI app (Docker)
  main.py                app factory, CORS, router registration, startup (create_all + seed admin)
  database.py            SQLAlchemy engine/session, get_db(), check_connection()
  auth.py                PBKDF2 hashing, JWT issue/verify, role dependencies
  notify.py              zero-cost SMTP e-mail (welcome notification; optional)
  live_manager.py        WebSocket fan-out singleton (thread→loop bridge)
  models.py              ORM: Detection, SurveyLog
  models_live.py         ORM: LiveEvent, LiveReport, LiveDevice
  models_auth.py         ORM: User, CityLandmark
  schemas*.py            Pydantic v2 request/response models (core / live / auth)
  routes/                detections, stats, heatmap, priority, export, ingest,
                         live (events + devices + WS), auth, cities
  Dockerfile             python:3.11-slim + exiftool/ffmpeg/libgl
  requirements.backend.txt   runtime-only deps (NO torch/ultralytics/etc.)

pipeline/                Inference pipelines (host)
  job_watcher.py         host daemon: data/jobs/ → orchestrator subprocess
  orchestrator.py        7-stage coordinator, writes session.json
  preprocessor.py        Stage 1 — frames, GPS sync, sun angle, lighting
  detector.py            Stage 2 — RT-DETR-L (batched, fp16, per-class thresholds)
  segmentor.py           Stage 3 — SAM 2.1 Tiny masks + geometry features
  depth_estimator.py     Stage 4 — Monodepth2 relative depth
  severity_classifier.py Stage 5 — rule-based S1–S5
  deduplicator.py        Stage 6 — DBSCAN spatial dedup
  db_writer.py           Stage 7 — PostGIS upsert
  live_pipeline.py       LITE per-user edge agent (--pair, fp16/ONNX, motion gate)
  lite_severity.py       stage-5 formula on ~0.4 ms CV proxies
  simulate_fleet.py      multi-vehicle live-mode demo (no GPU)
  extract_gpx_from_video.py  embedded-GPS extraction fallback

frontend/                React + Vite SPA (Nginx in Docker)
  src/pages/             Home, Live, Map, Stats, Explorer, Priority, Ingestion,
                         About, Login, Register, Admin
  src/components/        Navbar, ui primitives
  src/context/           AuthContext (the only global state)
  src/hooks/             useApi, useTheme
  src/utils/             api.js (axios+JWT), live.js, driveMode.js, constants.js, format.js
  nginx.conf             SPA fallback + /api/ proxy + /api/live/ws upgrade block
  vite.config.js         dev server :3000, proxies /api (ws: true) → :8000

db/init/                 Fresh-volume schema: 01 core · 02 live · 03 auth · 04 live_devices
ml/                      Training/research layer (never imported by the backend)
  detection/ optimization/ segmentation/ evaluation/
  weights/ best.pt  sam2.1_hiera_tiny.pt  mono_640x192/  networks/   (not in git)
scripts/                 setup_db, run_survey, validate_*, dataset analysis
scheduler/daily_job.py   APScheduler cron — optional nightly pipeline run
docs/                    internal design notes (local only, not tracked in git)
data/                    SHARED bind mount (jobs/, raw/, processed/sessions/, datasets/)
docker-compose.yml  .env  requirements.txt (full host/ML deps)  CLAUDE.md
```

---

## Getting started

### What is included in the delivered package

The delivered ZIP contains everything needed to run, **except the `ml/` folder**:
`backend/`, `frontend/`, `pipeline/`, `scripts/`, `db/`, `docker-compose.yml`,
and a ready-to-use **`.env`** (no manual configuration needed for the containers).

**Excluded from the ZIP (must be added separately to run the GPU pipeline):**

- **`ml/weights/best.pt`** — RT-DETR-L detection checkpoint (download).
- **`ml/weights/sam2.1_hiera_tiny.pt`** — SAM 2.1 Tiny checkpoint (download).
- **`ml/weights/mono_640x192/`** — Monodepth2 depth weights (download + extract).
- **`ml/weights/networks/`** — Monodepth2 network definition `.py` files (from the
  Monodepth2 repo). The depth stage imports this as the `networks` package.

> The **Docker stack (db + backend + frontend) runs without `ml/` at all** — the web app,
> live mode, maps, statistics, explorer, auth, and database work out of the box with just
> `docker compose up --build`. `ml/` is only needed by the **host GPU pipeline** that
> processes uploaded survey videos (and by lite-pipeline edge agents).

### Prerequisites

- **Docker + Docker Compose** — for the db / backend / frontend stack. *This is all you
  need to run the web application, including Live mode.*
- For the **survey pipeline** only:
  - **Windows host with an NVIDIA GPU** + CUDA-capable PyTorch
  - A Python **venv** at `.venv/` with the host/ML deps from `requirements.txt`
  - **Model weights** under `ml/weights/` (below)
  - The Monodepth2 `networks/` code at `ml/weights/networks/` **or** a cloned Monodepth2
    repo pointed to by `MONODEPTH_ROOT`

### Model weights (downloaded separately)

Create `ml/weights/` matching this layout **exactly** (paths are read from `.env`,
relative to the project root):

```
ml/weights/
├── best.pt                    ← RT-DETR-L, fine-tuned on N-RDD2024 (detection)
├── sam2.1_hiera_tiny.pt       ← SAM 2.1 Tiny (segmentation)
├── mono_640x192/              ← Monodepth2 weights (depth)
│   ├── encoder.pth
│   ├── depth.pth
│   ├── pose.pth
│   └── pose_encoder.pth
└── networks/                  ← Monodepth2 network code (depth) — see below
    ├── __init__.py
    ├── resnet_encoder.py
    ├── depth_decoder.py
    ├── pose_decoder.py
    ├── pose_cnn.py
    └── layers.py
```

#### 1. Downloadable checkpoints

| Model | Destination | Source |
|-------|-------------|--------|
| **RT-DETR-L** (detection) | `ml/weights/best.pt` | https://www.kaggle.com/models/paraschiv/rt-detr-l-fine-tuned-on-nrdd2024 |
| **SAM 2.1 Tiny** (segmentation) | `ml/weights/sam2.1_hiera_tiny.pt` | https://www.kaggle.com/models/paraschiv/sam2-1-hiera-tiny-pt |
| **Monodepth2 weights** (depth) | `ml/weights/mono_640x192/{encoder,depth,pose,pose_encoder}.pth` | Official release zip: https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip — extract so the four `.pth` files sit **directly** under `ml/weights/mono_640x192/` |

#### 2. Monodepth2 network code → `ml/weights/networks/`

The depth stage does `import networks` from `MONODEPTH_ROOT` (which `.env` sets to
`ml/weights`). The code comes from **https://github.com/nianticlabs/monodepth2**. Choose
**one** method:

**Method A — populate `ml/weights/networks/` (keeps the default `.env`):**

```bash
git clone https://github.com/nianticlabs/monodepth2.git
mkdir -p ml/weights/networks
cp monodepth2/networks/__init__.py        ml/weights/networks/
cp monodepth2/networks/resnet_encoder.py  ml/weights/networks/
cp monodepth2/networks/depth_decoder.py   ml/weights/networks/
cp monodepth2/networks/pose_decoder.py    ml/weights/networks/
cp monodepth2/networks/pose_cnn.py        ml/weights/networks/
cp monodepth2/layers.py                   ml/weights/networks/   # repo root → into networks/
```

> **Important:** upstream `depth_decoder.py` imports `from layers import *` while
> `layers.py` lives at the repo root. Because this project keeps `layers.py` *inside*
> `networks/`, change that one import in `ml/weights/networks/depth_decoder.py` to:
> ```python
> from networks.layers import *
> ```
> No other edits are needed.

**Method B — point `MONODEPTH_ROOT` at the clone (no copying, no code edit):**

Leave `ml/weights/networks/` empty, clone the repo anywhere, and in `.env` set
`MONODEPTH_ROOT=C:\path\to\monodepth2`. The depth **weights** still go in
`ml/weights/mono_640x192/`.

### 1. Bring up the containerised stack

```bash
docker compose up -d                    # start db, backend, frontend
docker compose up -d --build            # rebuild after Dockerfile/dep changes
docker compose up -d --build frontend   # REQUIRED after editing frontend/src/**
docker compose logs -f backend          # tail backend logs
docker compose down                     # stop (keeps pgdata volume); add -v to wipe the DB
```

- Frontend: **http://localhost:3000** · Backend docs: **http://localhost:8000/docs**
- Backend health: **http://localhost:8000/health**
- First visit: **register an account** at `/register` (Citizen accounts confirm their
  e-mail when SMTP is configured; Municipality accounts additionally need admin
  approval), or log in with the seed admin — **you must set
  `ADMIN_USERNAME/EMAIL/PASSWORD` and `JWT_SECRET` in `.env`**; no defaults ship in
  the repo (documented in the project's internal docs).

### 2. Database schema — created automatically

On a **fresh** `pgdata` volume the postgres entrypoint runs `db/init/01…04.sql` in order
(extensions + all tables + indexes + trigger) before the backend connects. On existing
volumes the backend's startup `Base.metadata.create_all` adds any missing tables
(live/auth/devices). Nothing to run manually.

> Started the stack before a schema file existed and want a clean slate?
> `docker compose down -v && docker compose up -d --build` (`-v` deletes the DB volume).

`scripts/setup_db.py` remains available and is equivalent for the core schema (idempotent).

### 3. Start the host pipeline worker (required to process survey uploads)

```bash
python pipeline/job_watcher.py          # foreground; watches data/jobs/, runs on GPU
# background (Windows):
start /B pythonw pipeline\job_watcher.py
```

Now upload a survey from the **IngestionPage** at http://localhost:3000/ingest.

### Backend / frontend in dev mode (without Docker)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000   # from project root
cd frontend && npm install && npm run dev                       # Vite :3000, proxies /api (incl. WS)
```

---

## Configuration

A single `.env` at the project root is loaded by `python-dotenv` everywhere and by Docker
Compose (`env_file`). Groups:

- **Database**: `POSTGRES_DB / USER / PASSWORD / HOST / PORT`, `DATABASE_URL`
  *(port nuance: `5433` on the host, `db:5432` inside Docker)*.
- **Auth**: `JWT_SECRET` (**required for persistent sessions** — unset generates a
  random per-process secret + warning), `JWT_TTL_H`,
  `ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD` (seed admin — **no in-repo
  defaults**; unset ⇒ no admin is seeded), `GOOGLE_CLIENT_ID` (optional free Google
  sign-in; empty hides the button), `VERIFY_CODE_TTL_MIN` (30), and login rate
  limiting: `LOGIN_MAX_ATTEMPTS` (5), `LOGIN_WINDOW_S` (900), `LOGIN_LOCKOUT_S` (900).
- **E-mail (optional, free)**: `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USERNAME`,
  `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS` — unset ⇒ e-mail off. Gmail app
  passwords work and are free. **On networks that block outbound SMTP** (typical
  university/corporate egress filters) set `BREVO_API_KEY` instead: Brevo's free
  transactional API (300 mails/day, no card) delivers over HTTPS 443 and takes
  precedence over SMTP when both are set; `SMTP_FROM` must be a Brevo-verified
  sender.
- **Live mode**: `LIVE_CLUSTER_RADIUS_M` (25), `LIVE_EVENT_TTL_H` (72),
  `LIVE_CONFIRM_DEVICES` (2), `LIVE_VERIFY_DEVICES` (3), `LIVE_DISPUTE_MIN` (2),
  `LIVE_PAIR_CODE_TTL_MIN` (15). Edge agents: `LIVE_API_URL`,
  `RDDS_TOKEN / RDDS_EMAIL / RDDS_PASSWORD`.
- **Host paths (relative to project root)**: `MONODEPTH_ROOT=ml/weights`,
  `MONODEPTH_WEIGHTS_DIR=ml/weights/mono_640x192`, `WEIGHTS_DIR=ml/weights`,
  `RAW_FOOTAGE_DIR`, `RAW_GPS_DIR`. `PROJECT_ROOT` stays **unset** (the watcher
  auto-detects it).
- **Container mount**: `PROJECT_DATA_DIR=/app/data` — must equal the compose bind-mount
  target; it is the prefix `job_watcher` strips when translating container → host paths.
- **Survey pipeline**: `PIPELINE_DEVICE` (watcher forces `cuda`), `PIPELINE_FPS`,
  `WATCHER_POLL_S`, `DEDUP_CLUSTER_RADIUS_M`, `SURROUNDING_DENSITY_RADIUS_M`,
  `DETECTOR_BATCH`, `DETECTOR_HALF`, `DETECTOR_IMGSZ`, `JOB_STALE_TIMEOUT_S`,
  `MAX_UPLOAD_MB` (4096) / `MAX_GPS_MB` (50) upload caps.
- **Frontend build**: `VITE_API_URL` — only when the frontend is hosted away from the API
  (e.g. Vercel); unset keeps same-origin `/api` through the Nginx/Vite proxy.
- **City / scheduler / logging**: `CITY_NAME / LAT / LON / BBOX`,
  `PIPELINE_RUN_HOUR/MINUTE`, `LOG_LEVEL`, `LOG_FILE`.

> `.env` contains real secrets (DB password, JWT secret, admin password, SMTP password)
> and is **git-ignored**. The `OPEN_METEO_*` / `OSM_OVERPASS_*` vars are legacy/unused
> (enrichment removed).

---

## Running a survey

### Through the UI

Upload an `.mp4` (and optional `.gpx`) on the IngestionPage; watch the seven stages update
live; then open the MapPage to see the new detections.

### Manually (bypasses backend + watcher)

```bash
python pipeline/orchestrator.py --video data/raw/footage/survey.mp4 \
    --gps data/raw/gps_logs/survey.gpx --device cuda --fps 2.0

# useful flags:
#   --resume          reuse completed stage artifacts
#   --dry_run_db      run everything but skip the DB write
#   --save_debug      write per-stage overlay images
#   --keep_all_frames keep non-detection frames on disk
#   --verbose         DEBUG logging

python scripts/run_survey.py    # convenience wrapper for a manual survey
```

The scheduler (`scheduler/daily_job.py`, APScheduler) can fire a pipeline run nightly at
`PIPELINE_RUN_HOUR:MINUTE`.

---

## Running the live stack

```bash
# 1. Pair a dashcam / PC from the Live page (Devices → "Pair a dashcam / PC"), then:
python pipeline/live_pipeline.py --pair ABCD2345

# 2. Drive (or replay a recorded drive with its GPS track):
python pipeline/live_pipeline.py --video data/raw/footage/drive.mp4 \
    --gps data/raw/gps_logs/drive.gpx --realtime

# CPU-only machine? One-time free ONNX export, then:
python pipeline/live_pipeline.py --export-onnx
python pipeline/live_pipeline.py --video drive.mp4 --lat 46.77 --lon 23.62 \
    --weights ml/weights/best.onnx --device cpu

# No GPU, no footage — demo the live map with virtual vehicles:
python pipeline/simulate_fleet.py
```

On a phone: open the site, go to **Live → Devices → Start drive mode**, mount the phone,
and drive — impacts auto-report as potholes with GPS position.

---

## Deployment (free)

See [`FREE_DEPLOYMENT.md`](FREE_DEPLOYMENT.md) and
[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) for the full walkthroughs:

- **Frontend**: static Vite build on Vercel (`vercel.json` provides the SPA fallback);
  set `VITE_API_URL` to the backend origin at build time.
- **Backend + DB**: any free VM (e.g. Oracle Cloud free tier) running the same
  docker-compose stack.
- **WebSockets**: Vercel rewrites cannot proxy WS — the Live page connects the socket
  directly to `VITE_API_URL`, falling back to REST polling automatically.

---

## Validation utilities

There is **no `pytest` suite and no linter**. Verification is done with stand-alone stage
validators under `scripts/`, run individually:

```bash
python scripts/validate_severity.py
python scripts/validate_deduplication.py
python scripts/validate_depth.py
python scripts/validate_db_write.py
python scripts/validate_nrdd2024.py
python scripts/validate_chain_finetune.py
```

Plus dataset/inspection helpers (`dataset_analysis.py`, `inspect_*.py`,
`run_tokyo_validation.py`, `run_kitti_pipeline.py`) and training notebooks (`*.ipynb`).
The legacy `validate_*.py` scripts still hardcode absolute paths — research tools, not
runtime.

---

## Tech stack

- **Backend**: FastAPI 0.111, Uvicorn (+ `websockets`), SQLAlchemy 2.0, GeoAlchemy2,
  Pydantic v2 (+ email-validator), psycopg2, PyJWT (HS256), httpx, loguru, stdlib
  smtplib/PBKDF2. Python 3.11-slim in Docker.
- **Database**: PostgreSQL 15 + PostGIS 3.3 (`postgis/postgis:15-3.3`).
- **Frontend**: React 18, Vite 5, React Router 6, Leaflet / react-leaflet, Recharts,
  axios, lucide-react; browser DeviceMotion + Geolocation APIs (drive mode). Built
  static, served by Nginx alpine.
- **Pipelines / ML (host & edge)**: PyTorch 2.2 + CUDA, Ultralytics RT-DETR-L, SAM 2.1
  Tiny, Monodepth2, scikit-learn (DBSCAN), ONNX Runtime (optional CPU edge), xgboost,
  optuna / mealpy (PSO), pysolar, gpxpy, OpenCV. Python 3.12 venv.
- **Orchestration**: file-based job queue + host watcher, APScheduler (optional nightly),
  docker-compose.

### Models

| Role | Model | Notes |
|------|-------|-------|
| Detection | **RT-DETR-L** | Fine-tuned on N-RDD2024 (10-class), checkpoint `best.pt`; shared by survey stage 2 and the lite edge agent |
| Segmentation | **SAM 2.1 Tiny** | `sam2.1_hiera_tiny.pt`, mask geometry features (survey only) |
| Depth | **Monodepth2** | `mono_640x192`, KITTI-pretrained, relative disparity (survey only) |
| Severity | Rule-based | Weighted multi-signal, no learned model; same formula in survey stage 5 and lite proxies |
| Dedup | DBSCAN | Haversine metric, eps = `DEDUP_CLUSTER_RADIUS_M` |
| Phone sensing | Accelerometer heuristic | Jolt-threshold pothole detection in the browser (drive mode) |

---

## Known limitations

- **GPU required on the survey host.** The watcher forces `--device cuda`; CPU is only for
  explicit testing (`PIPELINE_DEVICE=cpu`). (The lite edge agent *does* support CPU via
  ONNX.)
- **One survey job at a time.** Concurrency is guarded by disk state; a stale job stops
  blocking uploads after `JOB_STALE_TIMEOUT_S` (default 2 h).
- **`surface_area_cm2` currently stores pixel area.** The cm² conversion via
  `focal_length_px` is plumbed but not yet applied.
- **GPS accuracy depends on clock alignment** between camera and GPS logger. The
  preprocessor resolves the video start time from explicit input → MP4 `creation_time` →
  first GPS point, with whole-hour/timezone drift heuristics.
- **Live-mode device identity is self-declared** for anonymous devices; distinct-device
  validation is honest-majority, not Sybil-proof. Paired devices (registry + revocation)
  mitigate but don't eliminate this.
- **WS fan-out is per-process** — fine for one backend replica; use the documented Redis
  pub/sub path to scale out.
- **Phone drive mode is a heuristic sensor** — jolt spikes correlate with potholes but
  also speed bumps and rough patches; reports enter the same crowd-validation funnel as
  everything else, which is the point.
- **Enrichment stays gone** — no Nominatim/Overpass/Open-Meteo in the pipeline, no
  dropped columns. (City landmarks are a UI-only cache.)

---

## License & attribution

Open-source project by **Paraschiv Tudor, 2026**.

**Model & dataset references**

- RT-DETR — Zhao et al., 2024. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- N-RDD2024 — Kaya & Codur, 2024. [doi:10.17632/27c8pwsd6v.3](https://doi.org/10.17632/27c8pwsd6v.3)
- SAM — Kirillov et al., 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- Monodepth2 — Godard et al., 2019. [arXiv:1806.01260](https://arxiv.org/abs/1806.01260)
- DBSCAN — Ester, Kriegel, Sander, Xu, KDD 1996.
- Smartphone pothole sensing — Eriksson et al., *The Pothole Patrol*, MobiSys 2008.

> See [`CLAUDE.md`](CLAUDE.md) for the condensed engineering guide and
> the project's internal docs for the full functionality reference.
