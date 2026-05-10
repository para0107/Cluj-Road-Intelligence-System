# Road Infrastructure Detection System (RIDS)

> **Automated urban road damage detection, classification, and prioritization using computer vision and machine learning — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x+CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL+PostGIS-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Bachelor's Thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science**
**Specialization: Artificial Intelligence · Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Overview

RIDS is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets through a 9-stage machine learning pipeline, enriches each detection with spatial, depth, severity, lighting, weather, and infrastructure context features, and exposes the results through a REST API backed by a PostGIS spatial database.

The goal is to replace expensive, infrequent, and subjective manual road inspections with a low-cost automated alternative — a dashcam, a GPS logger, and an overnight processing run.

---

## Motivation

Romania has one of the highest road accident rates in the European Union. Deteriorating urban road infrastructure is a significant contributing factor. Traditional road condition surveys in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective. A full city-wide survey can take months.

This project proposes an automated alternative that any municipality can adopt with minimal hardware investment. Survey footage collected during normal vehicle operations can be processed automatically every night, producing a continuously updated georeferenced damage map with severity scores and ranked repair lists.

---

## System Architecture

The system has four distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — ML Training          (ml/)                           │
│  RT-DETR-L · SAM 2.1 · Monodepth2 · PSO hyperparameter search  │
│  dataset preparation · N-RDD2024 fine-tune (complete)           │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Inference Pipeline   (pipeline/ · scripts/)          │
│  Frame extraction → Detection → Segmentation → Depth →         │
│  Severity → Enrichment → Deduplication → DB write               │
│  Triggered via orchestrator.py, run_survey.py, or daily_job.py │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Backend API          (backend/)                      │
│  FastAPI · SQLAlchemy · PostGIS · Pydantic v2                   │
│  Routes: detections · stats · heatmap · priority                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Frontend             (frontend/)                     │
│  Interactive map · Severity filters · Priority repair list      │
│  React 18 + Leaflet.js (in development)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline — Stage by Stage

```
┌─────────────────────────────────────────────────────────────────┐
│  PRE-SURVEY: ACO Route Planning [DEFERRED — future work]        │
│  Ant Colony Optimization over Cluj-Napoca OSM road network      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 1 — Preprocessor              [preprocessor.py] ✅       │
│  • Extract frames from .mp4 (1 per 0.5 seconds)                │
│  • Sync GPS coordinates from .gpx to each frame timestamp       │
│  • Compute sun angle per frame (pysolar)                        │
│  • Classify lighting: daylight / overcast / low_light           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 2 — Detector                  [detector.py] ✅            │
│  • RT-DETR-L inference on each frame (640×640)                  │
│  • Per-class confidence thresholds (0.35 damage, 0.50 markings) │
│  • TTA evaluated — zero gain on this dataset, disabled          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 3 — Segmentor                 [segmentor.py] ✅           │
│  • RT-DETR bounding boxes → SAM 2.1 Tiny box prompts            │
│  • SAM outputs pixel-level binary mask per detection            │
│  • Computes: surface_area_px, edge_sharpness,                   │
│    interior_contrast, mask_compactness                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 4 — Depth Estimator           [depth_estimator.py] ✅    │
│  • Monodepth2 mono_640x192 → dense relative disparity per frame │
│  • Three extraction paths: mask_region / central_crop / proxy   │
│  • depth_confidence = 1 − CV(region pixels)                     │
│  • Fallback: SAM geometry proxy when conf < 0.4 or low_light   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 5 — Severity Classifier  [severity_classifier.py] ✅     │
│  • Weighted multi-signal rule-based formula:                    │
│    depth_norm × w_d + surface_area × w_a +                     │
│    interior_contrast × w_c + edge_sharpness × w_s              │
│  • Per-class signal weights + class importance weights          │
│  • Outputs S1–S5 deterministically — no training required       │
│  • Validated on 1,919 detections from Run 3 Cluj footage        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 6 — Enricher                  [enricher.py] ✅            │
│  • Nominatim API    → street_name, road_class                   │
│  • OSM Overpass API → road_importance (1–3), infra_proximity    │
│  • Open-Meteo API   → weather at detection timestamp            │
│  • Rate-limited: 1 req/s Nominatim (usage policy enforced)      │
│  • Graceful GPS-missing passthrough (all fields null)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 7 — Deduplicator              [deduplicator.py] ✅        │
│  • DBSCAN spatial clustering (eps from .env, default 2 m)       │
│  • Haversine metric via sklearn BallTree                        │
│  • Keeps highest severity_score detection per cluster           │
│  • Produces HTML report: Leaflet map + Chart.js bar chart       │
│  • Graceful skip if no GPS data available                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 8 — DB Writer                 [db_writer.py] ✅           │
│  • PostgreSQL 15 + PostGIS upsert via psycopg2                  │
│  • ST_DWithin upsert: UPDATE existing if same class within 2 m  │
│  • Updates detection_count, deterioration_rate, priority_score  │
│  • Updates surrounding_density within 50 m radius               │
│  • All credentials from .env — no hardcoded values              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 9 — Backend API + Dashboard   [FastAPI + frontend/]      │
│  • REST API reads from PostGIS                                   │
│  • Routes: detections · stats · heatmap · priority              │
│  • Daily pipeline trigger via APScheduler (02:00 Bucharest TZ)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Stack

### Detection — RT-DETR-L

**RT-DETR-L** ([Zhao et al., 2024](https://arxiv.org/abs/2304.08069)) is a transformer-based object detector that replaces anchor-based NMS post-processing with end-to-end bipartite matching. Architecture: HGStem backbone → AIFI encoder → RepC3+FPN neck → RTDETRDecoder (300 queries, Hungarian matching).

| Property | Value |
|---|---|
| Architecture | HGStem backbone + AIFI encoder + RepC3 neck + RTDETRDecoder |
| Parameters | 32.8M |
| Input resolution | 640 × 640 |
| Pretrained weights | COCO 2017 (80 classes) |
| Fine-tuned on | RDD2022 + Pothole600 (Run 1), N-RDD2024 (Run 2), chain (Run 3) |
| Output classes | 4 (RDD2022 run) → 10 (N-RDD2024 runs) |
| Training hardware | Kaggle T4 × 2 |

**Three controlled fine-tuning experiments:**

| Run | Init → Dataset | Classes | mAP50 (val) | mAP50 (test) | Notes |
|---|---|---|---|---|---|
| 1 — Baseline | COCO → RDD2022+P600 | 4 | 0.272 | — | Default hyperparameters, 56 epochs |
| 2 — PSO-optimised | COCO → RDD2022+P600 | 4 | **0.465** | **0.458** | PSO best params, +71% over baseline |
| 3 — N-RDD2024 direct | COCO → N-RDD2024 | 10 | **0.577** | — | **Best model — used in pipeline** |
| 4 — N-RDD2024 chain | RDD2022 → N-RDD2024 | 10 | 0.419 | — | Chain hypothesis refuted |

**PSO hyperparameter optimisation** ([Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968)):

10 particles × 4 iterations × 2 eval epochs (~9.3 h on P100). Fitness: mAP50-95.

| Parameter | Default | PSO Best | Δ |
|---|---|---|---|
| `lr0` | 1.0e-04 | 4.47e-04 | +347% |
| `weight_decay` | 5.0e-04 | 5.27e-04 | +5.3% |
| `warmup_epochs` | 3 | 1 | −67% |
| `mosaic` | 1.0 | 0.860 | −14% |
| `mixup` | 0.15 | 0.205 | +37% |
| `box` | 7.5 | 7.685 | +2.5% |
| `cls` | 0.5 | 0.487 | −2.6% |

---

### Segmentation — SAM 2.1 Tiny

**SAM 2.1** ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)), zero-shot, box-prompted. Validated on 1,904 detections from Run 3.

**Four geometry features per mask:**

| Feature | Field | Description |
|---|---|---|
| Surface area | `surface_area_px` | Pixel count — damage extent proxy |
| Edge sharpness | `edge_sharpness` | Mean Sobel gradient on boundary ring |
| Interior contrast | `interior_contrast` | Mean intensity delta inside vs. erosion ring |
| Mask compactness | `mask_compactness` | 4π·area/perimeter² (circle=1.0) |

---

### Depth Estimation — Monodepth2

**Monodepth2** ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)), `mono_640x192`, KITTI pretrained. Validated on 1,330 frames (1,904 boxes), 0 null depth values, 31 ms/frame on RTX 2050.

---

### Severity Classification — Rule-Based (Stage 5)

Weighted multi-signal formula across 4 signals from Stages 3 and 4. Per-class signal weights + class importance weights. Validated on 1,919 detections:

| Level | Score range | Count | Action |
|---|---|---|---|
| S1 | [0.00, 0.15) | 532 (27.7%) | Monitor |
| S2 | [0.15, 0.35) | 1,092 (56.9%) | Schedule maintenance |
| S3 | [0.35, 0.55) | 267 (13.9%) | Priority repair |
| S4 | [0.55, 0.75) | 26 (1.4%) | Urgent repair |
| S5 | [0.75, 1.00] | 2 (0.1%) | Emergency closure |

---

## Detection Classes

**N-RDD2024 schema (10 classes, `best.pt`, mAP50=0.577):**

| ID | D-code | Class | Category |
|---|---|---|---|
| 0 | D00 | `longitudinal_crack` | Damage |
| 1 | D10 | `transverse_crack` | Damage |
| 2 | D20 | `alligator_crack` | Damage |
| 3 | D30 | `repaired_crack` | Damage |
| 4 | D40 | `pothole` | Damage |
| 5 | D50 | `pedestrian_crossing_blur` | Marking (S1 cap) |
| 6 | D60 | `lane_line_blur` | Marking (S1 cap, conf≥0.50) |
| 7 | D70 | `manhole_cover` | Infrastructure |
| 8 | D80 | `patchy_road` | Damage |
| 9 | D90 | `rutting` | Damage |

---

## Validation Results

### Run 3 — Full Pipeline (Stages 1–5, 4,107 frames, Cluj-Napoca)

| Metric | Value |
|---|---|
| Frames processed | 4,107 |
| Frames with detections | 1,330 (32.4%) |
| Total boxes accepted | 1,904 |
| Mean confidence | 0.501 |
| Elapsed (CPU+GPU) | 3,729.8 s (~1.04 h) |

**Detection rate progression:**

| Run | Model | Detection rate |
|---|---|---|
| Run 1 (4-class) | `rtdetr_l_rdd2022.pt` | 4.0% |
| Run 2 (10-class, sample) | `best.pt` | ~29.8% |
| Run 3 (10-class, full + SAM) | `best.pt` + SAM 2.1 | **32.4%** |

### Stage 4 Depth Validation (1,330 frames)

- 0 null depth values across all 1,904 detections
- Per-frame normalisation working correctly (no boundary spikes)
- Physically consistent: near-field road = high disparity, sky = low disparity
- Inference: 31 ms/frame on RTX 2050 (CUDA)

### Stage 5 Severity Validation (1,919 detections)

- `alligator_crack` highest mean score (0.60) — correctly leads severity
- Marking classes capped at S2 by formula (max score 0.20) — no hardcoded if-branches
- S3 band populated with 267 actionable priority repairs

---

## Database

PostgreSQL 15 + PostGIS in Docker (`b0c432af798e` · `cluj-monitor-db` · port 5432).

**`detections` table** (key columns):

| Column | Type | Source |
|---|---|---|
| `geom` | GEOMETRY(POINT, 4326) | GPS from Stage 1 |
| `damage_type` | VARCHAR(30) | RT-DETR class name |
| `confidence` | FLOAT | RT-DETR score |
| `surface_area_cm2` | FLOAT | SAM geometry (px proxy) |
| `edge_sharpness` | FLOAT | SAM geometry |
| `interior_contrast` | FLOAT | SAM geometry |
| `mask_compactness` | FLOAT | SAM geometry |
| `depth_estimate_cm` | FLOAT | Monodepth2 relative (proxy) |
| `depth_confidence` | FLOAT | Monodepth2 CV-based |
| `severity` | SMALLINT | S1=1 … S5=5 |
| `severity_confidence` | FLOAT | Proxy-penalised |
| `lighting_condition` | VARCHAR(15) | HSV classifier |
| `street_name` | VARCHAR(150) | Nominatim |
| `road_importance` | SMALLINT | OSM Overpass (1–3) |
| `infra_proximity_m` | FLOAT | Haversine to nearest hospital/school/fire_station |
| `weather` | JSONB | Open-Meteo |
| `first_detected` | DATE | Pipeline run date |
| `last_detected` | DATE | Updated on upsert |
| `detection_count` | INTEGER | Incremented on upsert |
| `deterioration_rate` | FLOAT | Δseverity / days |
| `surrounding_density` | INTEGER | Detections within 50 m |
| `priority_score` | FLOAT | Formula below |

**Priority score formula:**
```
priority_score = w_severity × w_road × w_infra × log(detection_count + 1)
```

**`survey_log` table:** one row per pipeline run — `survey_date`, `started_at`, `finished_at`, `status`, `frames_processed`, `detections_found`, `new_detections`, `updated_detections`, `error_message`, `video_files`.

---

## Project Structure

```
RIDS/
│
├── pipeline/
│   ├── preprocessor.py          ✅ Frame extraction + GPS sync + lighting
│   ├── detector.py              ✅ RT-DETR-L inference + per-class thresholds
│   ├── segmentor.py             ✅ SAM 2.1 masks + 4 geometry features
│   ├── depth_estimator.py       ✅ Monodepth2 relative depth + proxy fallback
│   ├── severity_classifier.py   ✅ Rule-based S1–S5 weighted multi-signal
│   ├── enricher.py              ✅ Nominatim + Overpass + Open-Meteo
│   ├── deduplicator.py          ✅ DBSCAN spatial clustering + HTML report
│   ├── db_writer.py             ✅ PostgreSQL/PostGIS upsert (Stage 8)
│   └── orchestrator.py          ✅ End-to-end 9-stage coordinator
│
├── scripts/
│   ├── setup_db.py              ✅ Create tables, indexes, triggers
│   ├── detect_and_sam.py        ✅ RT-DETR + SAM 2.1 validation (Stages 1–3)
│   ├── validate_depth.py        ✅ Monodepth2 3-panel depth validation
│   ├── validate_severity.py     ✅ Stage 5 severity formula validation
│   ├── validate_enrichment.py   ✅ Stage 6 enrichment validation
│   ├── validate_deduplication.py ✅ Stage 7 DBSCAN deduplication validation
│   ├── validate_db_write.py     ✅ Stage 8 DB write validation (dry run + live)
│   ├── run_survey.py            ✅ Manual one-shot pipeline trigger
│   ├── run_kitti_pipeline.py    ✅ Full pipeline test on KITTI dataset
│   └── generate_kitti_report.py ✅ HTML visual report for KITTI runs
│
├── backend/
│   ├── main.py                  ✅ FastAPI app + CORS
│   ├── database.py              ✅ SQLAlchemy async engine
│   ├── models.py                ✅ Detection + SurveyLog ORM models
│   ├── schemas.py               ✅ Pydantic v2 schemas
│   └── routes/
│       ├── detections.py        ✅ GET /detections, /{id}, /nearby
│       ├── stats.py             ✅ GET /stats
│       ├── heatmap.py           ✅ GET /heatmap
│       └── priority.py          ✅ GET /priority-list
│
├── ml/
│   ├── detection/
│   │   ├── train.py             ✅ Two-phase training + PSO integration
│   │   ├── evaluate.py          ✅ Per-class AP, mAP50-95, checkpoint compare
│   │   └── data_prep/           ✅ RDD2022 + Pothole600 + N-RDD2024 prep
│   └── weights/
│       ├── best.pt              ✅ N-RDD2024 operational checkpoint (mAP50=0.577)
│       ├── rtdetr_l_rdd2022.pt  ✅ PSO-optimised 4-class (mAP50=0.458 test)
│       ├── sam2.1_hiera_tiny.pt ✅ SAM 2.1 Tiny
│       ├── mono_640x192/        ✅ Monodepth2 encoder.pth + depth.pth
│       └── networks/            ✅ Monodepth2 architecture (ResnetEncoder etc.)
│
├── scheduler/
│   └── daily_job.py             ✅ APScheduler — 02:00 Europe/Bucharest
│
├── data/
│   ├── raw/footage/             Input dashcam .mp4 files
│   ├── raw/gps_logs/            GPX telemetry files
│   ├── processed/sessions/      Per-run session directories (orchestrator output)
│   │   ├── kitti_0001/          ✅ KITTI drive 0001 session
│   │   ├── kitti_0002/          ✅ KITTI drive 0002 session
│   │   ├── kitti_0018/          ✅ KITTI drive 0018 session
│   │   └── kitti_0057/          ✅ KITTI drive 0057 session
│   ├── validation_nrdd_2024/
│   │   ├── sam_masks/cluj/      ✅ SAM overlays (1,904 detections)
│   │   ├── depth_maps/          ✅ Monodepth2 3-panel visualisations
│   │   ├── severity/            ✅ severity_results.json (1,919 detections)
│   │   ├── enriched/            ✅ enriched.json
│   │   ├── deduplicated/        ✅ deduplicated.json + dedup_report.html
│   │   └── db_write/            ✅ db_write_summary.json
│   ├── datasets/kitti/          KITTI 2011_09_26 drives (image_03 + oxts)
│   └── reports/                 HTML visual reports from generate_kitti_report.py
│
├── docker-compose.yml           ✅ PostgreSQL 15 + PostGIS + pgAdmin
├── .env                         ✅ All credentials and paths (never committed)
└── requirements.txt             ✅ Pinned versions
```

---

## Usage

### Environment setup

```bash
# Create .env (copy template and fill in values)
# Required keys: POSTGRES_*, RTDETR_WEIGHTS, MONODEPTH_ROOT,
#                MONODEPTH_WEIGHTS_DIR, NOMINATIM_USER_AGENT

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL + PostGIS
docker-compose up -d

# Create tables, indexes, triggers
python scripts/setup_db.py
```

### Validation pipeline (stage by stage)

```bash
# Stage 1–3: RT-DETR + SAM on Cluj footage
python scripts/detect_and_sam.py

# Stage 4: Monodepth2 depth validation
python scripts/validate_depth.py --device cuda

# Stage 5: Severity classifier validation
python scripts/validate_severity.py

# Stage 6: Enrichment (Nominatim + Overpass + Open-Meteo)
python scripts/validate_enrichment.py --skip_weather   # fast offline test
python scripts/validate_enrichment.py                  # full with all APIs

# Stage 7: DBSCAN deduplication
python scripts/validate_deduplication.py

# Stage 8: DB write (dry run by default)
python scripts/validate_db_write.py           # dry run — no DB writes
python scripts/validate_db_write.py --live    # real writes to cluj_monitor
```

### Orchestrator (full end-to-end run)

```bash
# Full run on dashcam footage
python pipeline/orchestrator.py \
    --video data/raw/footage/survey_01.mp4 \
    --gps   data/raw/gps_logs/survey_01.gpx \
    --device cuda

# No GPS available (enrichment + dedup skip gracefully)
python pipeline/orchestrator.py \
    --video data/raw/footage/survey_01.mp4

# Skip enrichment, dry run DB
python pipeline/orchestrator.py \
    --video data/raw/footage/survey_01.mp4 \
    --skip_enrichment --dry_run_db --verbose

# Resume an interrupted run
python pipeline/orchestrator.py \
    --video data/raw/footage/survey_01.mp4 \
    --session_id 20260510_143022 \
    --resume
```

Each run creates `data/processed/sessions/<session_id>/` containing:
```
01_manifest/manifest.json
02_detections/detections.json
03_segmentations/segmentations.json
04_depth/depth_estimates.json
05_severity/severity_estimates.json
06_enriched/enriched.json
07_deduplicated/deduplicated.json + dedup_report.html
08_db_write/db_write_summary.json
session.json
```

### KITTI dataset pipeline test

```bash
# Full run — all 4 drives, live DB writes
python scripts/run_kitti_pipeline.py

# Single drive
python scripts/run_kitti_pipeline.py --drive 0001

# Quick smoke test — first 10 frames, no DB writes
python scripts/run_kitti_pipeline.py --limit 10 --dry_run_db

# Skip enrichment (faster, no network calls)
python scripts/run_kitti_pipeline.py --skip_enrichment

# Resume after crash
python scripts/run_kitti_pipeline.py --resume

# Generate visual HTML report after the run
python scripts/generate_kitti_report.py
```

KITTI drives processed: `2011_09_26_drive_0001_sync` (108 frames), `0002` (77 frames), `0018`, `0057`. Camera: `image_03` (right colour, focal length 721 px). GPS: from `oxts/data/{N:010d}.txt` (field 0 = lat, field 1 = lon). Timestamps: from `image_03/timestamps.txt`.

### Backend API

```bash
uvicorn backend.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/detections` | All detections, paginated and filterable |
| `GET` | `/detections/{id}` | Single detection with all features |
| `GET` | `/detections/nearby` | Detections within radius of coordinates |
| `GET` | `/stats` | City-wide counts by type and severity |
| `GET` | `/heatmap` | Density grid for map overlay |
| `GET` | `/priority-list` | Ranked repair list by priority_score |
| `POST` | `/process` | Trigger processing of new survey footage |

---

## Environment Variables (.env)

All secrets and paths live in `.env`. Never committed to git.

| Key | Description |
|---|---|
| `POSTGRES_DB` | Database name (`cluj_monitor`) |
| `POSTGRES_USER` | PostgreSQL user |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_HOST` | Host (default `localhost`) |
| `POSTGRES_PORT` | Port (default `5432`) |
| `DATABASE_URL` | SQLAlchemy connection URL |
| `RTDETR_WEIGHTS` | Weight filename under `WEIGHTS_DIR` (e.g. `best.pt`) |
| `WEIGHTS_DIR` | Base directory for model weights (`ml/weights`) |
| `MONODEPTH_ROOT` | Path to cloned Monodepth2 repo |
| `MONODEPTH_WEIGHTS_DIR` | Path to `mono_640x192/` directory |
| `NOMINATIM_USER_AGENT` | Required by Nominatim usage policy |
| `OSM_OVERPASS_URL` | Overpass API endpoint |
| `OPEN_METEO_BASE_URL` | Open-Meteo API endpoint |
| `DEDUP_CLUSTER_RADIUS_M` | DBSCAN epsilon in metres (default `2`) |
| `SURROUNDING_DENSITY_RADIUS_M` | Density search radius in metres (default `50`) |
| `LOG_LEVEL` | Logging level (`INFO` / `DEBUG`) |
| `LOG_FILE` | Log file path (`logs/pipeline.log`) |

---

## Roadmap

**Completed (all implemented in this project):**
- [x] Dataset download, conversion, merge, and verification scripts
- [x] RT-DETR-L two-phase training pipeline + PSO hyperparameter search
- [x] Baseline (mAP50=0.272), PSO-optimised (mAP50=0.465/0.458 test), N-RDD2024 (mAP50=0.577)
- [x] N-RDD2024 chain fine-tune ablation — chain hypothesis refuted
- [x] TTA evaluated — zero gain, disabled
- [x] FastAPI backend — all routes tested on Swagger
- [x] APScheduler daily job (02:00 Europe/Bucharest)
- [x] Docker Compose — PostgreSQL 15 + PostGIS + pgAdmin
- [x] `preprocessor.py` — frame extraction, GPS sync, lighting, shadow, sun angle
- [x] `detector.py` — RT-DETR-L inference, per-class thresholds, save/load
- [x] `segmentor.py` — SAM 2.1 Tiny, 4 geometry features, save/load
- [x] `depth_estimator.py` — Monodepth2, 3 extraction paths, proxy fallback
- [x] `severity_classifier.py` — weighted multi-signal S1–S5, validated on 1,919 dets
- [x] `enricher.py` — Nominatim + Overpass + Open-Meteo, rate-limited, .env-driven
- [x] `deduplicator.py` — DBSCAN Haversine, HTML report (Leaflet map + Chart.js)
- [x] `db_writer.py` — PostGIS upsert, priority score, surrounding density, schema-aligned
- [x] `orchestrator.py` — end-to-end 9-stage coordinator, resume support, survey_log writes
- [x] `validate_depth.py` — Monodepth2 3-panel validation, 1,330 frames, 0 null depths
- [x] `validate_severity.py` — full S1–S5 validation on 1,919 detections
- [x] `validate_enrichment.py` — Stage 6 validation with GPS-missing graceful handling
- [x] `validate_deduplication.py` — Stage 7 validation, HTML report generation
- [x] `validate_db_write.py` — Stage 8 validation (dry run + live mode)
- [x] `run_kitti_pipeline.py` — full pipeline test on KITTI 2011_09_26 (4 drives, real GPS)
- [x] `generate_kitti_report.py` — 7-figure HTML visual report (scatter, heatmap, timing, etc.)
- [x] Thesis chapters 3, 4, 5 restructured (SAM/Monodepth/Severity moved to Chapter 5)

**In progress:**
- [ ] React frontend — interactive map, severity filters, priority list sidebar (React 18 + Leaflet.js)

**Planned:**
- [ ] Real GPS survey run — dashcam + GPX synchronised, Cluj-Napoca streets
- [ ] KITTI full pipeline run with live DB writes and visual report
- [ ] TRIB dataset annotation → Romanian domain fine-tuning ([Abrudan, 2025](https://doi.org/10.1007/s12145-025-01763-7))
- [ ] ACO survey route generation — deferred, future work
- [ ] City Hall pilot demonstration

---

## Known Issues & Compatibility Notes

- **Monodepth2 `networks` import conflict:** A `networks` PyPI package can shadow `ml/weights/networks/`. Fix: `pip uninstall networks -y`. Verify with `python -c "import sys; sys.path.insert(0, 'ml/weights'); import networks; print(hasattr(networks, 'ResnetEncoder'))"` — must print `True`.
- **Monodepth2 `networks/__init__.py`:** Must exist and export `ResnetEncoder`, `DepthDecoder`, `PoseDecoder`, `PoseCNN`. If missing, create it manually.
- **RTDETR_WEIGHTS in .env:** Must match the actual filename in `ml/weights/`. The operational checkpoint is `best.pt` (N-RDD2024, mAP50=0.577). The old `rtdetr_l_rdd2022.pt` is a 4-class checkpoint — loading it with the 10-class `CLASS_NAMES` list causes all predictions to be dropped.
- **survey_log schema:** The running Docker container may have an older schema without `started_at`/`finished_at`. Fix in DataGrip or psql: `ALTER TABLE survey_log ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ, ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ, ...` (see full query in known issues history).
- **PyTorch CUDA:** RTX 2050 + driver 555.97 (CUDA 12.5) — use cu121 wheels: `pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121`.
- **Pillow version:** `requirements.txt` pins `Pillow==9.5.0` for Ultralytics 8.2.18 compatibility. `PIL._util.is_directory` removed in Pillow 10.0.
- **GPS coordinates:** All sessions without a paired `.gpx` file will have `latitude=None` in all frames. Stages 6–8 handle this gracefully (passthrough, no API calls, no DB writes). KITTI sessions have real GPS from `oxts/data/`.
- **KITTI timestamps:** Nanosecond precision (`2011-09-26 13:02:25.961178112`) — truncated to microseconds before parsing with `datetime.strptime`.
- **Pipeline is sequential:** Every stage runs to completion before the next begins. SAM (Stage 3) dominates wall-clock time (~0.5–1 s per detection on RTX 2050). Multi-drive parallelism is possible via separate processes but requires VRAM budget check (RT-DETR ~1.2 GB + SAM ~0.2 GB + Monodepth2 ~0.8 GB = ~2.2 GB per instance).

---

## Papers

All papers used across 10 categories. Papers marked [EVALUATED, NOT DEPLOYED] were considered in design but are outside the operational pipeline.

### 0. Foundational

| Paper | Authors | Year | Link |
|---|---|---|---|
| Deep Residual Learning (ResNet) | He et al. | 2016 | [arxiv](https://arxiv.org/abs/1512.03385) |
| Attention Is All You Need | Vaswani et al. | 2017 | [arxiv](https://arxiv.org/abs/1706.03762) |
| AdamW | Loshchilov, Hutter | 2019 | [arxiv](https://arxiv.org/abs/1711.05101) |
| ImageNet Classification (AlexNet) | Krizhevsky et al. | 2012 | [acm](https://dl.acm.org/doi/10.1145/3065386) |
| Vision Transformer (ViT) | Dosovitskiy et al. | 2021 | [arxiv](https://arxiv.org/abs/2010.11929) |
| Microsoft COCO | Lin et al. | 2014 | [arxiv](https://arxiv.org/abs/1405.0312) |
| PASCAL VOC | Everingham et al. | 2010 | [doi](https://doi.org/10.1007/s11263-009-0275-4) |

### 1. Object Detection & Transformers

| Paper | Authors | Year | Link |
|---|---|---|---|
| RT-DETR | Zhao, Lv et al. | 2024 | [arxiv](https://arxiv.org/abs/2304.08069) |
| RT-DETRv2 | Lv, Zhao et al. | 2024 | [arxiv](https://arxiv.org/abs/2407.17140) |
| DETR | Carion et al. | 2020 | [arxiv](https://arxiv.org/abs/2005.12872) |
| FPN | Lin, Dollár et al. | 2017 | [arxiv](https://arxiv.org/abs/1612.03144) |
| Focal Loss | Lin, Goyal et al. | 2017 | [arxiv](https://arxiv.org/abs/1708.02002) |

### 2. Road Damage Detection

| Paper | Authors | Year | Link |
|---|---|---|---|
| RDD2022 | Arya et al. | 2024 | [doi](https://doi.org/10.1002/gdj3.260) |
| RDDC Challenge Series | Tanaka et al. | 2025 | [Nature MI](https://www.nature.com/articles/s42256-025) |
| N-RDD2024 | Kaya, Çodur | 2024 | [doi](https://doi.org/10.17632/27c8pwsd6v.3) |
| SAM for Road Asset Inventorying | Zhang, Huang, Qin | 2024 | [doi](https://doi.org/10.1016/j.ijtst.2024.10.005) |
| TRIB crack dataset | Abrudan | 2025 | [doi](https://doi.org/10.1007/s12145-025-01763-7) |

### 3. Segmentation

| Paper | Authors | Year | Link |
|---|---|---|---|
| SAM | Kirillov et al. | 2023 | [arxiv](https://arxiv.org/abs/2304.02643) |
| SAM 2 | Ravi et al. | 2024 | [arxiv](https://arxiv.org/abs/2408.00714) |
| Hiera | Bolya et al. | 2023 | [arxiv](https://arxiv.org/abs/2306.00989) |

### 4. Depth Estimation

| Paper | Authors | Year | Link |
|---|---|---|---|
| Monodepth2 | Godard et al. | 2019 | [arxiv](https://arxiv.org/abs/1806.01260) |
| EfficientNet [EVALUATED, NOT DEPLOYED] | Tan, Le | 2019 | [arxiv](https://arxiv.org/abs/1905.11946) |
| Monocular Pothole Distance | Hach, Sankowski | 2015 | [researchgate](https://www.researchgate.net/publication/285578304) |

### 5. Hyperparameter Optimisation

| Paper | Authors | Year | Link |
|---|---|---|---|
| PSO | Kennedy, Eberhart | 1995 | [doi](https://doi.org/10.1109/ICNN.1995.488968) |
| PSO for DNN Hyper-Parameters | Young et al. | 2015 | [doi](https://doi.org/10.1145/3071178.3071208) |
| Optuna [EVALUATED, NOT DEPLOYED] | Akiba et al. | 2019 | [arxiv](https://arxiv.org/abs/1907.10902) |
| WOA [EVALUATED, NOT DEPLOYED] | Mirjalili, Lewis | 2016 | [doi](https://doi.org/10.1016/j.advengsoft.2016.01.008) |

### 6. Training Techniques

| Paper | Authors | Year | Link |
|---|---|---|---|
| SWA | Izmailov et al. | 2018 | [arxiv](https://arxiv.org/abs/1803.05407) |
| Albumentations | Buslaev et al. | 2020 | [doi](https://doi.org/10.3390/info11020125) |
| Mixup | Zhang et al. | 2018 | [arxiv](https://arxiv.org/abs/1710.09412) |
| Label smoothing | Szegedy et al. | 2016 | [doi](https://doi.org/10.1109/CVPR.2016.308) |

### 7. Clustering & Spatial

| Paper | Authors | Year | Link |
|---|---|---|---|
| DBSCAN | Ester, Kriegel et al. | 1996 | [acm](https://dl.acm.org/doi/10.5555/3001460.3001507) |

### 8. Route Optimisation

| Paper | Authors | Year | Link |
|---|---|---|---|
| ACO [DEFERRED] | Dorigo, Maniezzo, Colorni | 1996 | [doi](https://doi.org/10.1109/3477.484436) |

---

## Technology Stack

| Layer | Technology | Version / Notes |
|---|---|---|
| Detection | RT-DETR-L | Ultralytics 8.2.18 |
| Segmentation | SAM 2.1 Tiny | `sam2.1_hiera_tiny.pt`, zero-shot |
| Depth estimation | Monodepth2 | `mono_640x192`, KITTI pretrained |
| Severity classification | Rule-based | S1–S5 weighted multi-signal |
| Hyperparameter optimisation | PSO (custom) | 7-dim, 10 particles × 4 iters |
| Spatial clustering | DBSCAN | scikit-learn, Haversine BallTree |
| Geocoding | Nominatim | OSM, 1 req/s rate limit enforced |
| Road network | OSM Overpass API | road_importance + infra_proximity |
| Weather | Open-Meteo API | free, no API key |
| Sun angle | pysolar | UTC datetime + GPS coords |
| Database | PostgreSQL 15 + PostGIS | Docker, GIST spatial index |
| ORM | SQLAlchemy 2.0 async | GeoAlchemy2 for PostGIS types |
| Backend | FastAPI + Pydantic v2 | 0.111 |
| Scheduler | APScheduler | Europe/Bucharest TZ |
| Frontend | React 18 + Leaflet.js | In development |
| Containerisation | Docker Compose | PostgreSQL + pgAdmin |
| Test dataset | KITTI 2011_09_26 | `image_03`, 4 drives, real GPS |
| Language | Python 3.12 | — |

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 ([Arya et al., 2024](https://doi.org/10.1002/gdj3.260)), Pothole600, N-RDD2024 ([Kaya & Çodur, 2024](https://doi.org/10.17632/27c8pwsd6v.3)), KITTI ([Geiger et al., 2013](http://www.cvlibs.net/datasets/kitti/)).
Model attributions: RT-DETR ([Zhao et al., 2024](https://arxiv.org/abs/2304.08069)), SAM 2.1 ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)), Monodepth2 ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Mathematics and Computer Science · 2026</i>
</div>