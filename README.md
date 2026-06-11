# RIDS — Road Infrastructure Detection System

> **Automated urban road-damage detection, classification, and prioritization from dashcam footage — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%20%2F%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%20%2B%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20%2B%20PostGIS-15--3.3-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Bachelor's Thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science**
**Specialization: Artificial Intelligence · Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Table of contents

- [Overview](#overview)
- [Motivation](#motivation)
- [System architecture](#system-architecture)
- [The two-context execution model](#the-two-context-execution-model)
- [Job lifecycle](#job-lifecycle-end-to-end)
- [The 7-stage inference pipeline](#the-7-stage-inference-pipeline)
- [Severity model](#severity-model)
- [Database schema](#database-schema)
- [REST API](#rest-api)
- [Frontend](#frontend)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Running a survey](#running-a-survey)
- [Validation utilities](#validation-utilities)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [License & attribution](#license--attribution)

---

## Overview

RIDS is an end-to-end urban infrastructure monitoring platform that automatically
detects, classifies, and prioritizes road damage from smartphone / dashcam video.
A raw `.mp4` survey of city streets (optionally accompanied by a `.gpx` GPS log) is
pushed through a seven-stage computer-vision pipeline that:

1. extracts frames and synchronises each one to a GPS coordinate,
2. detects 10 classes of road damage and road features with a fine-tuned **RT-DETR-L** model,
3. segments each detection with **SAM 2.1 Tiny** to derive geometry features,
4. estimates relative depth with **Monodepth2**,
5. assigns a rule-based **S1–S5 severity** score,
6. spatially de-duplicates repeated sightings of the same physical damage with **DBSCAN**, and
7. upserts the surviving detections into a **PostGIS** spatial database with a computed priority score.

The results are served through a **FastAPI** REST API and visualised in a **React + Leaflet**
single-page app: a live detection map, a statistics dashboard, a tabular explorer, and an
ingestion page that streams real-time, stage-by-stage pipeline progress.

The goal is to replace expensive, infrequent, and subjective manual road inspections with a
low-cost automated alternative — a dashcam, an optional GPS logger, and an overnight processing run.

---

## Motivation

Romania has one of the highest road-accident rates in the European Union, and deteriorating
urban road infrastructure is a significant contributing factor. Traditional condition surveys
in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective; a full
city-wide survey can take months.

RIDS proposes an automated alternative that any municipality can adopt with minimal hardware
investment. Footage collected during normal vehicle operation can be processed automatically,
producing a continuously updated georeferenced damage map with severity scores and a ranked
repair list.

---

## System architecture

The system is split into four layers:

| Layer | Runs in | Responsibility |
|-------|---------|----------------|
| **Frontend** | Docker (Nginx) | React SPA — map, stats, explorer, ingestion |
| **Backend API** | Docker (Uvicorn) | FastAPI REST + job hand-off; **no ML, no GPU** |
| **Database** | Docker (PostGIS) | Spatial store for detections + survey log |
| **Inference pipeline** | Windows host (GPU) | The 7-stage CV pipeline (PyTorch + CUDA) |

```
┌──────────────────────────── Docker (Linux) ────────────────────────────┐
│  db        postgis/postgis:15-3.3        cluj_monitor_db                  │
│  backend   FastAPI + Uvicorn             cluj_monitor_backend             │
│  frontend  React build served by Nginx   cluj_monitor_frontend           │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                  shared bind mount   ./data  ⇄  /app/data
                                   │
┌─────────────────────────── Windows HOST (GPU) ──────────────────────────┐
│  pipeline/job_watcher.py    long-running daemon, watches data/jobs/       │
│  pipeline/orchestrator.py   spawned per job with --device cuda            │
│  RTX 2050 GPU · host Python venv (.venv) · PyTorch+CUDA · model weights   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The two-context execution model

This is the single most important thing to understand about RIDS.

The backend container **cannot** run the ML pipeline: it has no GPU, no PyTorch/Ultralytics/
SAM/Monodepth2 in its image, and no access to the Windows weight paths. So the backend never
executes the pipeline directly. Instead it **drops a job file** into the shared `data/jobs/`
directory; a host-side watcher (`job_watcher.py`) picks it up and runs the orchestrator on the
GPU.

The two sides communicate **only through files under `data/`** — there is no socket, queue, or
RPC between them. This boundary is intentional and load-bearing:

- The backend writes container-side paths (`/app/data/...`) into the job file.
- `job_watcher._container_to_host()` rewrites the `/app/data` prefix to the host `data/` path
  before invoking the orchestrator.
- The orchestrator writes `session.json`, which the backend reads back to report progress.

---

## Job lifecycle (end-to-end)

```
Frontend IngestionPage
   │ POST /api/ingest/upload   (multipart: video .mp4 [+ gps .gpx])
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

`job_id` and `session_id` are the **same value**, format `YYYYMMDD_HHMMSS` (UTC), generated by
the backend and reused as the orchestrator session id so every path lines up.

Status values the frontend understands:
`pending` → `initialising` → `running` → `complete` | `failed`.

---

## The 7-stage inference pipeline

The canonical order and stage keys are defined by `orchestrator.run()`:

| # | Stage key | Module | What it does |
|---|-----------|--------|--------------|
| 1 | `preprocessor` | `preprocessor.py` | Extract frames (~2 fps), sync GPS from `.gpx` (timestamp interpolation), solar elevation (pysolar), lighting class (HSV) |
| 2 | `detector` | `detector.py` | **RT-DETR-L** inference (N-RDD2024 fine-tuned `best.pt`), per-class confidence filter |
| 3 | `segmentor` | `segmentor.py` | **SAM 2.1 Tiny** masks + 4 geometry features (area, edge sharpness, interior contrast, compactness) |
| 4 | `depth_estimator` | `depth_estimator.py` | **Monodepth2** (mono_640x192) relative disparity → depth proxy, with geometry-proxy fallback |
| 5 | `severity_classifier` | `severity_classifier.py` | Rule-based **S1–S5** weighted multi-signal score |
| 6 | `deduplicator` | `deduplicator.py` | **DBSCAN** + Haversine spatial clustering (radius `DEDUP_CLUSTER_RADIUS_M`) |
| 7 | `db_writer` | `db_writer.py` | **PostGIS** upsert + priority-score update |

**10 damage / feature classes** (N-RDD2024 schema): `longitudinal_crack`, `transverse_crack`,
`alligator_crack`, `repaired_crack`, `pothole`, `pedestrian_crossing_blur`, `lane_line_blur`,
`manhole_cover`, `patchy_road`, `rutting`.

### GPS guard — GPS-less runs are valid

Stages 6 and 7 require coordinates. If no `.gpx` is supplied and embedded-GPS extraction fails,
frames have `latitude/longitude = None`; the orchestrator **skips stages 6 & 7 and still finishes
`complete`** (nothing is written to the DB). Detection / segmentation / depth / severity artifacts
are still produced, and the frontend marks the skipped stages accordingly. A GPS-less run is **not**
a failure.

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

Frontend-triggered runs force `--save_debug` (via `_SAVE_DEBUG` in `job_watcher.py`), writing
overlay images under each `0X_*/debug/`. Non-detection frames are pruned from disk after Stage 2
(`--keep_all_frames` to disable).

### Enrichment is permanently removed

An earlier "Enricher" stage (Nominatim / OSM Overpass / Open-Meteo, plus columns like
`street_name`, `road_importance`, `weather`, etc.) was intentionally dropped from the pipeline,
the ORM, the schemas, and the DB. **GPS lat/lon is the only location metadata stored.**

---

## Severity model

Severity is a transparent, rule-based score in `[0, 1]` mapped to five levels. It combines four
normalised signals with **per-class weights**, then applies a **class-importance weight**:

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

By construction, marking classes (`lane_line_blur`, `pedestrian_crossing_blur`,
`class_weight = 0.10`) can never exceed **S2**, and `repaired_crack` (`0.15`) is capped near S2 —
no special-case branches needed. A `severity_confidence` value reflects measurement quality and
is reduced when the geometry depth-proxy was substituted for Monodepth2.

**Priority score** (no enrichment inputs): `severity_weight · log(detection_count + 1)`.

---

## Database schema

PostgreSQL 15 + PostGIS 3.3. Two tables, created by `scripts/setup_db.py` and mirrored by
`backend/models.py`.

### `detections` — one row per de-duplicated damage instance

`id` (UUID), `geom` (POINT/4326), `latitude`, `longitude`, `damage_type`, `confidence`,
`frame_path`, segmentation geometry (`surface_area_cm2`, `edge_sharpness`, `interior_contrast`,
`mask_compactness`), `depth_estimate_cm`, `depth_confidence`, `lighting_condition`,
`severity` (1–5), `severity_confidence`, `surrounding_density`, temporal fields
(`first_detected`, `last_detected`, `detection_count`, `deterioration_rate`), `priority_score`,
`survey_date`, `survey_video_file`, `is_fixed`.

Indexes: GIST on `geom`, plus `severity`, `damage_type`, `survey_date`, `priority_score DESC`.
An upsert matches an existing detection of the same `damage_type` within
`DEDUP_CLUSTER_RADIUS_M` metres (PostGIS `ST_DWithin` on `geography`) and bumps its
`detection_count` / `deterioration_rate` instead of inserting a duplicate.

### `survey_log` — one row per `survey_date` (unique)

Tracks a run's `status`, `started_at` / `finished_at`, `frames_processed`, `detections_found`,
`new_detections`, `updated_detections`, `error_message`, `video_files` (JSONB).

### Port nuance (host vs container)

- **Inside Docker** the backend reaches the DB by service name: `…@db:5432/…`.
- **On the host** the DB container publishes `${POSTGRES_PORT:-5432}:5432` (`.env` sets
  `POSTGRES_PORT=5433`), so host-run code (orchestrator `psycopg2`, `db_writer`, host-run
  backend) connects via `localhost:5433`.

---

## REST API

All routes are under `/api` (Nginx proxies `/api/` → `backend:8000`; Vite dev proxies the same).

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/detections` | Paginated list with filters & sorting |
| `GET`  | `/api/detections/nearby` | Radius search (lat, lon, radius_m, limit) |
| `GET`  | `/api/detections/{id}` | Single detection |
| `PATCH`| `/api/detections/{id}/status` | Set `is_fixed` |
| `DELETE`| `/api/detections/bulk` | Delete by id list (optionally cascade `survey_log`) |
| `GET`  | `/api/stats` | Aggregate stats for the dashboard |
| `GET`  | `/api/heatmap` | Weighted points for the map heat layer |
| `GET`  | `/api/priority-list` | Ranked repair list |
| `GET`  | `/api/export/csv` | CSV export of all detections |
| `POST` | `/api/ingest/upload` | Upload video (+gps), queue a job → 202 |
| `GET`  | `/api/ingest/status/{job_id}` | Poll job / session status |
| `GET`  | `/` , `/health` | Health checks |

Interactive docs: **http://localhost:8000/docs** · CORS is currently `allow_origins=["*"]`
(tighten before any real deployment).

---

## Frontend

React 18 + Vite 5 + React Router 6; Leaflet / react-leaflet for the map, Recharts for charts,
lucide-react icons, axios. **No global state library** — page-local `useState`/`useEffect` plus
`localStorage['rids_active_job']` as the only cross-page "a pipeline is running" signal.

| Page | Route | Purpose |
|------|-------|---------|
| **MapPage** | `/` | Live detection map; box-select to inspect/delete; silent refresh while a job runs |
| **StatsPage** | `/stats` | City-wide statistics dashboard |
| **ExplorerPage** | `/explorer` | Sortable / filterable detection table |
| **IngestionPage** | `/ingest` | Upload + per-stage live pipeline tracker |

Styling is inline-`style`-object based (no Tailwind/CSS modules); colours come from CSS variables
defined in `index.css`. The frontend is a **static production build** baked into the Nginx image —
**rebuild the image after any `src/` change**: `docker compose up -d --build frontend`.

---

## Repository layout

```
backend/                 FastAPI app (Docker)
  main.py                app factory, CORS, router registration, /health
  database.py            SQLAlchemy engine/session, get_db(), check_connection()
  models.py              ORM: Detection, SurveyLog
  schemas.py             Pydantic v2 request/response models
  routes/                detections, stats, heatmap, priority, export, ingest
  Dockerfile             python:3.11-slim + exiftool/ffmpeg/libgl
  requirements.backend.txt   runtime-only deps (NO torch/ultralytics/etc.)

pipeline/                Inference pipeline (Windows host + GPU)
  job_watcher.py         host daemon: data/jobs/ → orchestrator subprocess
  orchestrator.py        7-stage coordinator, writes session.json
  preprocessor.py        Stage 1 — frames, GPS sync, sun angle, lighting
  detector.py            Stage 2 — RT-DETR-L inference
  segmentor.py           Stage 3 — SAM 2.1 Tiny masks + geometry features
  depth_estimator.py     Stage 4 — Monodepth2 relative depth
  severity_classifier.py Stage 5 — rule-based S1–S5
  deduplicator.py        Stage 6 — DBSCAN spatial dedup
  db_writer.py           Stage 7 — PostGIS upsert
  extract_gpx_from_video.py  embedded-GPS extraction fallback

frontend/                React + Vite SPA (Nginx in Docker)
  src/pages/             MapPage, StatsPage, ExplorerPage, IngestionPage
  src/components/        Navbar
  src/hooks/             useApi
  src/utils/             api.js (axios client), constants.js
  nginx.conf             SPA fallback + /api/ reverse proxy
  vite.config.js         dev server :3000, proxies /api → :8000

ml/                      Training/research layer (not part of inference runtime)
  detection/ optimization/ segmentation/ depth/ severity/ evaluation/
  weights/ best.pt  sam2.1_hiera_tiny.pt  mono_640x192/  networks/

scripts/                 One-off + validation utilities (setup_db, run_survey,
                         validate_*, dataset analysis, training notebooks)
scheduler/daily_job.py   APScheduler cron — nightly pipeline run
data/                    SHARED bind mount (jobs/, raw/, processed/sessions/, datasets/)
docker-compose.yml  .env  requirements.txt (full host/ML deps)  CLAUDE.md
```

---

## Getting started

### Prerequisites

- **Docker + Docker Compose** (for db / backend / frontend)
- **Windows host with an NVIDIA GPU** + CUDA-capable PyTorch (for the pipeline)
- A Python **venv** at `.venv/` with the host/ML deps from `requirements.txt`
- Model weights in `ml/weights/`: `best.pt` (RT-DETR-L), `sam2.1_hiera_tiny.pt`, `mono_640x192/`
- The **Monodepth2 repo** cloned separately (see `MONODEPTH_ROOT` in `.env`)

### 1. Bring up the containerised stack

```bash
docker compose up -d                    # start db, backend, frontend
docker compose up -d --build            # rebuild after Dockerfile/dep changes
docker compose up -d --build frontend   # REQUIRED after editing frontend/src/**
docker compose logs -f backend          # tail backend logs
docker compose down                     # stop (keeps pgdata volume)
```

- Frontend: **http://localhost:3000** · Backend docs: **http://localhost:8000/docs**
- Backend health: **http://localhost:8000/health**

### 2. Initialise the database schema (once, host venv)

```bash
python scripts/setup_db.py    # postgis/pgcrypto extensions + tables + indexes + trigger
```

### 3. Start the host pipeline worker (must run for uploads to process)

```bash
python pipeline/job_watcher.py          # foreground; watches data/jobs/, runs on GPU
# background (Windows):
start /B pythonw pipeline\job_watcher.py
```

Now upload a survey from the **IngestionPage** at http://localhost:3000/ingest, or run the
pipeline manually (below).

### Backend / frontend in dev mode (without Docker)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000   # from project root
cd frontend && npm install && npm run dev                       # Vite on :3000
```

---

## Configuration

A single `.env` at the project root is loaded by `python-dotenv` everywhere and by Docker Compose
(`env_file`). Notable groups:

- **Database**: `POSTGRES_DB / USER / PASSWORD / HOST / PORT`, `DATABASE_URL`
  *(see the port nuance above — `5433` on the host, `5432` inside Docker)*.
- **Host paths (Windows, absolute)**: `PROJECT_ROOT`, `MONODEPTH_ROOT`, `MONODEPTH_WEIGHTS_DIR`,
  `WEIGHTS_DIR`, `RAW_FOOTAGE_DIR`, `RAW_GPS_DIR` — not used inside containers.
- **Container mount**: `PROJECT_DATA_DIR=/app/data` — must equal the compose bind-mount target;
  it is the prefix `job_watcher` strips when translating container → host paths.
- **Weights**: `RTDETR_WEIGHTS=best.pt`, SAM `sam2.1_hiera_tiny.pt`, `mono_640x192/`.
- **Pipeline**: `PIPELINE_DEVICE` (watcher forces `cuda`), `PIPELINE_FPS`, `WATCHER_POLL_S`,
  detection thresholds, `DEDUP_CLUSTER_RADIUS_M`.
- **City**: `CITY_NAME / LAT / LON / BBOX` (Cluj-Napoca).
- **Scheduler / logging**: `PIPELINE_RUN_HOUR/MINUTE`, `LOG_LEVEL`, `LOG_FILE`.

> `.env` contains real secrets (DB password) and is **git-ignored**. The `NOMINATIM_*`,
> `OPEN_METEO_*`, `OSM_OVERPASS_*` vars are legacy/unused (enrichment removed).

---

## Running a survey

### Through the UI

Upload an `.mp4` (and optional `.gpx`) on the IngestionPage; watch the seven stages update live;
then open the MapPage to see the new detections.

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

## Validation utilities

There is **no `pytest` suite**. Verification is done with stand-alone stage validators under
`scripts/`, run individually:

```bash
python scripts/validate_severity.py
python scripts/validate_deduplication.py
python scripts/validate_depth.py
python scripts/validate_db_write.py
python scripts/validate_nrdd2024.py
python scripts/validate_chain_finetune.py
```

Plus dataset/inspection helpers (`dataset_analysis.py`, `inspect_*.py`, `run_tokyo_validation.py`,
`run_kitti_pipeline.py`) and training notebooks (`*.ipynb`).

---

## Tech stack

- **Backend**: FastAPI 0.111, Uvicorn, SQLAlchemy 2.0, GeoAlchemy2, Pydantic v2, psycopg2, loguru.
  Python 3.11-slim in Docker.
- **Database**: PostgreSQL 15 + PostGIS 3.3 (`postgis/postgis:15-3.3`).
- **Frontend**: React 18, Vite 5, React Router 6, Leaflet / react-leaflet, Recharts, axios,
  lucide-react. Built static, served by Nginx alpine.
- **Pipeline / ML (host)**: PyTorch 2.2 + CUDA, Ultralytics RT-DETR-L, SAM 2.1 Tiny, Monodepth2,
  scikit-learn (DBSCAN), xgboost, optuna / mealpy (PSO), pysolar, gpxpy, OpenCV. Python 3.12 venv.
- **Orchestration**: APScheduler (nightly), file-based job queue, docker-compose.

### Models

| Role | Model | Notes |
|------|-------|-------|
| Detection | **RT-DETR-L** | Fine-tuned on N-RDD2024 (10-class), checkpoint `best.pt` |
| Segmentation | **SAM 2.1 Tiny** | `sam2.1_hiera_tiny.pt`, mask geometry features |
| Depth | **Monodepth2** | `mono_640x192`, KITTI-pretrained, relative disparity |
| Severity | Rule-based | Weighted multi-signal, no learned model |
| Dedup | DBSCAN | Haversine metric, eps = `DEDUP_CLUSTER_RADIUS_M` |

---

## Known limitations

- **GPU required on the host.** The watcher forces `--device cuda`; CPU is only for explicit
  testing (`PIPELINE_DEVICE=cpu`).
- **One job at a time.** Concurrency is guarded by disk state (the backend rejects uploads with
  `409` while a job file is `pending`/`running`, and the watcher processes one job synchronously).
  A watcher crash can leave a job stuck in `running`; the job file must be cleared manually before
  new uploads are accepted.
- **`surface_area_cm2` currently stores pixel area.** The cm² conversion via `focal_length_px` is
  plumbed but not yet applied; the value is raw mask pixel count.
- **GPS accuracy depends on clock alignment** between the camera and the GPS logger. The
  preprocessor resolves the video start time from explicit input → MP4 `creation_time` → first GPS
  point, and includes heuristics for whole-hour/timezone drift.
- **Some host paths are hardcoded Windows absolute paths** (e.g. Monodepth2 defaults).
- **Enrichment stays gone** — no Nominatim/Overpass/Open-Meteo, no dropped columns.

---

## License & attribution

Academic project — Bachelor's thesis, Babeș-Bolyai University, Faculty of Mathematics and
Computer Science (Artificial Intelligence specialization), **Paraschiv Tudor, 2026**.

**Model & dataset references**

- RT-DETR — Zhao et al., 2024. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- N-RDD2024 — Kaya & Codur, 2024. [doi:10.17632/27c8pwsd6v.3](https://doi.org/10.17632/27c8pwsd6v.3)
- SAM — Kirillov et al., 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- Monodepth2 — Godard et al., 2019. [arXiv:1806.01260](https://arxiv.org/abs/1806.01260)
- DBSCAN — Ester, Kriegel, Sander, Xu, KDD 1996.

> See [`CLAUDE.md`](CLAUDE.md) for the in-depth engineering guide (execution boundaries, gotchas,
> and conventions) used when developing this repository.
