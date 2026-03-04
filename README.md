# Cluj Road Intelligence System

> **Automated urban road damage detection, classification, and prioritization using computer vision and machine learning — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL+PostGIS-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Bachelor's Thesis — Babeș-Bolyai University, Faculty of Computer Science**
**Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Overview

Cluj Road Intelligence System (CRIS) is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets through a 9-stage machine learning pipeline, enriches each detection with 26 features covering geometry, depth, severity, lighting, weather, and infrastructure context, and presents findings through an interactive municipal dashboard designed for City Hall decision-making.

The goal is to replace expensive, infrequent, and subjective manual road inspections with a low-cost, automated, data-driven alternative — a dashcam, a GPS logger, and an overnight processing run.

---

## Motivation

Romania has one of the highest road accident rates in the European Union. Deteriorating urban road infrastructure is a significant contributing factor. Traditional road condition surveys in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective. A full survey of the city road network can take months.

This project proposes an automated alternative that any municipality can adopt with minimal hardware investment. Survey footage collected during normal city vehicle operations can be processed automatically every night, producing a continuously updated georeferenced damage map with severity scores and ranked repair lists.

---

## System Architecture

The system consists of a 9-stage inference pipeline, a machine learning training subsystem, a PostgreSQL + PostGIS database, a FastAPI REST backend, and a React dashboard.

```
┌──────────────────────────────────────────────────────────────────┐
│                       PRE-SURVEY                                 │
│          ACO Route Planning (osmnx + Cluj OSM network)          │
│          Finds optimal driving route covering all roads          │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 1 — Preprocessor                    [preprocessor.py]     │
│  • Extract frames from .mp4 (1 per 0.5 seconds)                 │
│  • Sync GPS coordinates from .gpx to each frame timestamp        │
│  • Compute sun angle per frame (pysolar)                         │
│  • Classify lighting: daylight / overcast / low_light            │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 2 — Detector                           [detector.py]      │
│  • RT-DETR-L inference on each frame (640×640)                   │
│  • Test Time Augmentation: flip + rotate, averaged predictions   │
│  • Confidence threshold: discard detections < 0.5               │
│  • 5 output classes (see Detection Classes section)              │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 3 — Segmentor                         [segmentor.py]      │
│  • RT-DETR bounding boxes used as SAM box prompts                │
│  • SAM outputs pixel-level mask per detection                    │
│  • Computes: surface_area, edge_sharpness,                       │
│    interior_contrast, mask_compactness                           │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 4 — Depth Estimator              [depth_estimator.py]     │
│  • EfficientNet-B3 regression → depth in cm (primary path)       │
│  • Monodepth2 dense depth map → depth at detection region        │
│  • Both estimates fused for final depth_estimate                 │
│  • Fallback: mask geometry proxy (low_light / low confidence)    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 5 — Severity Classifier        [severity_classifier.py]   │
│  • XGBoost on WOA-selected feature subset → S1 through S5        │
│  • Fallback: rule-based (depth + area) until Cluj data ready     │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 6 — Enricher                          [enricher.py]       │
│  • Nominatim API     → street_name (reverse geocoding)           │
│  • OSM Overpass API  → road_importance, infra_proximity          │
│  • Open-Meteo API    → weather at detection timestamp            │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 7 — Deduplicator                   [deduplicator.py]      │
│  • DBSCAN spatial clustering (2m radius)                         │
│  • Merges duplicate detections from multiple survey passes       │
│  • PostGIS upsert: UPDATE existing or INSERT new record          │
│  • Updates deterioration_rate and detection_count across runs    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 8 — Database                  [PostgreSQL + PostGIS]      │
│  • 26-column detections table                                    │
│  • GIST spatial index on geom column                             │
│  • Full temporal tracking across survey runs                     │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  STAGE 9 — Dashboard             [FastAPI + React + Leaflet]     │
│  • Interactive map, severity filters, priority repair list       │
│  • Per-detection detail panel with all 26 features               │
│  • Automated daily pipeline trigger (APScheduler)                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Stack

### Detection — RT-DETR-L

**RT-DETR-L** (Real-Time Detection Transformer, Large) is a transformer-based object detector that outperforms YOLO-series models on accuracy while maintaining real-time inference. It replaces the traditional anchor-based detection head with a transformer decoder, eliminating NMS post-processing.

| Property | Value |
|---|---|
| Architecture | HGStem backbone + AIFI encoder + RepC3 neck + RTDETRDecoder |
| Parameters | 32.8M |
| Input resolution | 640 × 640 |
| Pretrained weights | COCO (80 classes) |
| Fine-tuned on | RDD2022 + Pothole600 (27,336 images) |
| Output classes | 5 road damage types |
| Training hardware | NVIDIA GeForce RTX 2050 (4 GB VRAM) |

**Training dataset:**

| Dataset | Images | Annotations | Countries |
|---|---|---|---|
| RDD2022 | 21,479 | 33,913 | India, Japan, Norway, China, USA |
| Pothole600 | 5,857 | 8,970 | Mixed |
| **Total** | **27,336** | **42,883** | 6 |

**Training strategy:**
- **Phase 1** (10 epochs): Frozen backbone — only decoder and detection head trained. LR = 1e-4.
- **Phase 2** (50 epochs): Full fine-tune — backbone unfrozen, LR drops to 1e-5. Early stopping patience = 20.
- **SWA**: Stochastic Weight Averaging over last 5 checkpoints → `swa.pt`.

**Training techniques:**
- Focal Loss (γ=2.0, built into RT-DETR) for class imbalance
- Mixup augmentation (α=0.15) and Mosaic augmentation
- Albumentations pipeline (HSV jitter, rotation, scale, shear, horizontal flip)
- Test Time Augmentation at inference (flip + rotate)
- fp16 AMP, gradient accumulation ×4 (effective batch = 16)
- `cudnn.benchmark = True`, `deterministic = False`

**Hyperparameter optimization — PSO:**
Particle Swarm Optimization searches a 7-dimensional space (`lr0`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `box`, `cls`) with 15 particles over 10 iterations. Best config saved to `ml/optimization/pso_best.json` and auto-loaded on the next training run.

### Segmentation — SAM

Used **zero-shot** — no fine-tuning required. RT-DETR bounding boxes serve as box prompts. Four geometry features computed from each mask:

| Feature | Description |
|---|---|
| `surface_area` | Damage extent in cm² |
| `edge_sharpness` | Sobel gradient magnitude along mask boundary |
| `interior_contrast` | Mean pixel intensity inside vs. outside mask |
| `mask_compactness` | 4π × area / perimeter² (circle=1.0, crack≈0.05) |

### Depth Estimation — EfficientNet-B3 + Monodepth2

- **EfficientNet-B3**: regression head on cropped detection region + sun angle → depth in cm. Trained on Cluj ground truth + Blender synthetic renders. Tuned with Optuna (TPE, 50 trials).
- **Monodepth2**: self-supervised dense depth map — depth at detection region extracted and averaged with EfficientNet estimate.
- **Fallback**: proxy from mask geometry when `depth_confidence < 0.4` or `low_light`.

### Severity Classification — XGBoost + WOA

**Whale Optimization Algorithm** selects the optimal binary feature subset from the 16-feature ML vector before XGBoost training.

| Level | Description | Typical depth | Action |
|---|---|---|---|
| S1 | Superficial | < 1 cm | Monitor |
| S2 | Minor | 1–3 cm | Schedule maintenance |
| S3 | Moderate | 3–6 cm | Priority repair |
| S4 | Severe | 6–10 cm | Urgent repair |
| S5 | Critical | > 10 cm | Emergency closure |

### Route Optimization — ACO

**Ant Colony Optimization** computes the optimal pre-survey driving route through the Cluj-Napoca OSM road network, minimizing total distance while covering all primary and secondary roads.

---

## Detection Classes

| ID | Class | Description |
|---|---|---|
| 0 | `longitudinal_crack` | Cracks parallel to road direction |
| 1 | `transverse_crack` | Cracks perpendicular to road direction |
| 2 | `alligator_crack` | Interconnected crack networks (fatigue damage) |
| 3 | `pothole` | Bowl-shaped depressions with measurable depth |
| 4 | `patch_deterioration` | Degraded previously-repaired sections |

**Class distribution in training set:**

| Class | Instances | Share |
|---|---|---|
| longitudinal_crack | 18,201 | 42.4% |
| pothole | 8,770 | 20.5% |
| transverse_crack | 8,386 | 19.6% |
| alligator_crack | 7,526 | 17.5% |
| patch_deterioration | 0 | Reserved for Cluj data |

---

## Database Schema

```sql
CREATE TYPE damage_type_enum AS ENUM (
    'longitudinal_crack', 'transverse_crack', 'alligator_crack',
    'pothole', 'patch_deterioration'
);
CREATE TYPE severity_enum  AS ENUM ('S1', 'S2', 'S3', 'S4', 'S5');
CREATE TYPE lighting_enum  AS ENUM ('daylight', 'overcast', 'low_light');

CREATE TABLE detections (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    geom                  GEOMETRY(POINT, 4326),

    -- Group 1: Detection Core (RT-DETR)
    damage_type           damage_type_enum,
    confidence            FLOAT,

    -- Group 2: Geometry (SAM)
    surface_area          FLOAT,
    edge_sharpness        FLOAT,
    interior_contrast     FLOAT,
    mask_compactness      FLOAT,

    -- Group 3: Depth + Severity
    depth_estimate        FLOAT,
    depth_confidence      FLOAT,
    severity              severity_enum,
    severity_confidence   FLOAT,

    -- Group 4: Lighting
    lighting_condition    lighting_enum,
    shadow_geometry_score FLOAT,

    -- Group 5: Location Context
    street_name           VARCHAR,
    road_importance       SMALLINT,
    infra_proximity       FLOAT,

    -- Group 6: Weather
    weather               JSONB,

    -- Group 7: Temporal Tracking
    first_detection_date  TIMESTAMP,
    last_detection_date   TIMESTAMP,
    detection_count       INT DEFAULT 1,
    deterioration_rate    FLOAT DEFAULT 0.0,

    -- Group 8: Derived
    surrounding_density   INT DEFAULT 0,
    priority_score        FLOAT
);

CREATE INDEX detections_geom_idx ON detections USING GIST(geom);
```

---

## Project Structure

```
Cluj-Road-Intelligence-System/
│
├── ml/
│   ├── detection/
│   │   ├── train.py                ✅ RT-DETR-L two-phase training + SWA
│   │   ├── evaluate.py             ✅ Per-class AP, mAP50-95, checkpoint comparison
│   │   ├── monitor.py              ✅ Live training dashboard
│   │   └── data_prep/
│   │       └── coco_to_yolo.py     ✅ COCO JSON → YOLO .txt conversion
│   ├── depth/
│   │   └── train.py                ⬜ EfficientNet-B3 depth regression
│   ├── severity/
│   │   ├── train_xgboost.py        ⬜ XGBoost severity classifier
│   │   └── feature_selection_woa.py ⬜ WOA feature selection
│   ├── optimization/
│   │   ├── pso_hyperparams.py      ✅ PSO search (7-dim, 15 particles)
│   │   ├── optuna_search.py        ⬜ Optuna HPO for EfficientNet-B3 + XGBoost
│   │   └── aco_route.py            ⬜ ACO survey route
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     🔄 Training in progress
│       ├── rtdetr_l_cluj.pt        ⬜ Future: fine-tuned on Cluj footage
│       ├── depth_effnet.pt         ⬜ Future: EfficientNet-B3 depth model
│       └── xgboost_severity.json   ⬜ Future: XGBoost classifier
│
├── pipeline/
│   ├── preprocessor.py             ⬜ Frame extraction + GPS sync + lighting
│   ├── detector.py                 ⬜ RT-DETR inference + TTA
│   ├── segmentor.py                ⬜ SAM masks + geometry features
│   ├── depth_estimator.py          ⬜ EfficientNet-B3 + Monodepth2
│   ├── severity_classifier.py      ⬜ XGBoost (rule-based fallback)
│   ├── enricher.py                 ⬜ OSM + Nominatim + Open-Meteo
│   ├── deduplicator.py             ⬜ DBSCAN + PostGIS upsert
│   └── orchestrator.py             ⬜ End-to-end coordinator
│
├── backend/
│   ├── core/
│   │   ├── config.py               ✅ Settings + DB connection
│   │   └── database.py             ✅ SQLAlchemy async session
│   ├── models/
│   │   └── detection.py            ✅ SQLAlchemy ORM model
│   ├── schemas/
│   │   └── detection.py            ✅ Pydantic v2 schemas
│   ├── api/routes/
│   │   ├── detections.py           ✅ GET /detections, /{id}, /nearby
│   │   ├── stats.py                ✅ GET /stats, /heatmap, /priority-list
│   │   └── processing.py           ✅ POST /process + APScheduler daily job
│   └── main.py                     ✅ FastAPI app + CORS
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Map/                ⬜ Leaflet + severity pins + heatmap
│       │   ├── Sidebar/            ⬜ Stats + filters + priority list
│       │   └── DetailPanel/        ⬜ Per-detection 26-feature display
│       └── App.jsx                 ⬜ Root component
│
├── data/
│   ├── datasets/
│   │   ├── rdd2022/                ✅ 21,479 images
│   │   ├── pothole600/             ✅ 5,857 images
│   │   └── cluj/                   ⬜ Planned: locally collected footage
│   └── detection/
│       ├── dataset.yaml            ✅ Ultralytics training config
│       ├── train_images.txt        ✅ 27,336 paths
│       ├── val_images.txt          ✅ 5,857 paths
│       └── test_images.txt         ✅ 5,857 paths
│
├── scripts/
│   ├── setup_db.py                 ✅ PostgreSQL schema + enums + GIST index
│   └── run_survey.py               ⬜ CLI entrypoint for full pipeline
│
├── docker-compose.yml              ✅ PostgreSQL + PostGIS + pgAdmin
└── requirements.txt                ✅
```

**Legend:** ✅ Done · 🔄 In progress · ⬜ Planned

---

## Current Training Status

Phase 1 (10 frozen epochs) complete. Phase 2 (50 full fine-tune epochs) running.

| Epoch | GIoU ↓ | L1 ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|
| 1 | 1.418 | 1.033 | 0.201 | 0.00248 | 0.000631 |
| 3 | 1.157 | 0.721 | 0.335 | 0.0130 | 0.00398 |
| 5 | 1.028 | 0.609 | 0.447 | 0.0204 | 0.00690 |
| 7 | 0.977 | 0.569 | 0.500 | 0.0235 | 0.00845 |
| 10 | 0.942 | 0.546 | 0.533 | 0.0266 | 0.00992 |

---

## Planned: Cluj Data Collection & Fine-tuning

**Equipment:** smartphone dashcam + GPS logger + measuring tape for ground truth depth.

**Survey route:** generated by ACO over the Cluj-Napoca OSM road network.

**Annotation:** Label Studio — bounding boxes for all 5 classes + manual depth measurements.

**Fine-tuning pipeline:**
```
rtdetr_l_rdd2022.pt  →  Cluj frames       →  rtdetr_l_cluj.pt
EfficientNet-B3       →  depth measurements →  depth_effnet.pt
XGBoost               →  Cluj features      →  xgboost_severity.json
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/detections` | All detections, paginated and filterable |
| `GET` | `/detections/{id}` | Single detection with all 26 features |
| `GET` | `/detections/nearby` | Detections within radius of coordinates |
| `GET` | `/stats` | City-wide counts by type and severity |
| `GET` | `/heatmap` | Density grid for map overlay |
| `GET` | `/priority-list` | Ranked repair list by priority_score |
| `POST` | `/process` | Trigger processing of new survey footage |

Daily survey job embedded in FastAPI via APScheduler — fires automatically at 02:00 every night.

---

## Roadmap

**Done:**
- [x] RDD2022 + Pothole600 dataset merge and COCO → YOLO conversion
- [x] Data distribution analysis and plots
- [x] RT-DETR-L training pipeline (two-phase + SWA + PSO integration)
- [x] PSO hyperparameter optimization script
- [x] Evaluation script with per-class AP and checkpoint comparison
- [x] Live training monitor
- [x] Docker Compose — PostgreSQL + PostGIS + pgAdmin
- [x] Database schema, enums, GIST index
- [x] SQLAlchemy ORM models + Pydantic v2 schemas
- [x] FastAPI backend — all endpoints (tested on Swagger)
- [x] Daily APScheduler job

**In progress:**
- [ ] RT-DETR-L Phase 2 training (50 epochs)
- [ ] PSO search + retraining with best hyperparameters

**Planned:**
- [ ] Inference pipeline — all 8 modules
- [ ] React dashboard — map, sidebar, detail panel
- [ ] ACO survey route generation
- [ ] Cluj-Napoca data collection drive
- [ ] Label Studio annotation
- [ ] EfficientNet-B3 depth model training
- [ ] XGBoost + WOA severity classifier
- [ ] RT-DETR fine-tuning on Cluj footage
- [ ] End-to-end integration test
- [ ] City Hall pilot demonstration

---

## Papers

| Paper | Year |
|---|---|
| DETRs Beat YOLOs on Real-Time Object Detection | 2023 |
| RT-DETRv2: Improved Baseline with Bag-of-Freebies | 2024 |
| RT-DETRv3: Real-Time End-to-End Object Detection | 2024 |
| Segment Anything | 2023 |
| SAM 2: Segment Anything in Images and Videos | 2024 |
| When Segment Anything Model Meets Invisible | 2024 |
| EfficientNet: Rethinking Model Scaling for CNNs | 2019 |
| Digging into Self-Supervised Monocular Depth Estimation | 2019 |
| Robust Video-Based Pothole Detection | — |
| Computer Vision for Road Imaging and Pavement Distress | — |
| A Comparison of Low-Cost Monocular Vision Systems | — |
| XGBoost: A Scalable Tree Boosting System | 2016 |
| Particle Swarm Optimization | 1995 |
| Particle Swarm Optimization for Hyperparameter Selection | 2015 |
| Particle Swarm Optimization-Based Automated Detection | — |
| Modified XGBoost Hyper-Parameter Tuning | — |
| A Review of Whale Optimization Algorithm | — |
| Ant Colony Optimisation for Vehicle Routing | 1996 |
| Optuna: A Next-generation Hyperparameter Optimization Framework | 2019 |
| Focal Loss for Dense Object Detection | 2017 |
| Feature Pyramid Networks for Object Detection | 2017 |
| Averaging Weights Leads to Wider Optima and Better Generalization | 2018 |
| Albumentations: Fast and Flexible Image Augmentations | 2020 |
| A Density-Based Algorithm for Discovering Clusters (DBSCAN) | 1996 |
| RDD2022: A Multi-National Image Dataset for Road Damage Detection | 2024 |
| Road Damage Detection and Classification | — |
| Real-Time Road Damage Detection System | — |
| Pothole Detection in Adverse Weather | — |
| An Annotated Street View Image Dataset | — |
| End-to-End Object Detection with Transformers (DETR) | 2020 |

> Full citations with authors, venues, and DOIs are listed in the thesis bibliography.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Detection | RT-DETR-L (Ultralytics 8.2) |
| Segmentation | SAM — Segment Anything Model |
| Depth estimation | EfficientNet-B3 + Monodepth2 |
| Severity classification | XGBoost |
| Hyperparameter optimization | PSO (custom) · Optuna (TPE) |
| Feature selection | WOA — Whale Optimization Algorithm |
| Survey route planning | ACO · osmnx |
| Spatial clustering | DBSCAN (scikit-learn) |
| Augmentation | Albumentations · Mixup · Mosaic |
| Training | Focal Loss · SWA · TTA · fp16 AMP · gradient accumulation |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| ORM | SQLAlchemy 2.0 (async) |
| Backend | FastAPI + Pydantic v2 |
| Scheduler | APScheduler |
| Frontend | React 18 + Leaflet.js + OpenStreetMap |
| Containerization | Docker Compose |
| Geocoding | Nominatim (OpenStreetMap) |
| Road network | OSM Overpass API |
| Weather | Open-Meteo API |
| Sun angle | pysolar |
| Annotation | Label Studio |
| Language | Python 3.12 |

---

## Setup

```bash
git clone https://github.com/para0107/Cluj-Road-Intelligence-System.git
cd Cluj-Road-Intelligence-System

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Convert dataset labels (once)
python ml/detection/data_prep/coco_to_yolo.py

# Train
python ml/detection/train.py

# Monitor (second terminal)
python ml/detection/monitor.py --live

# Database
docker-compose up -d
python scripts/setup_db.py

# Backend
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 (Arya et al., 2024), Pothole600.
Model attributions: RT-DETR (Zhao et al., 2023), SAM (Kirillov et al., 2023), Monodepth2 (Godard et al., 2019).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Computer Science · 2026</i>
</div>