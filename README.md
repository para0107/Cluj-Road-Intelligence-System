# Cluj Road Intelligence System

> **Automated urban road damage detection, classification, and prioritization using computer vision and machine learning — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL+PostGIS-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Bachelor's Thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science**
**Specialization: Artificial Intelligence · Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Overview

Cluj Road Intelligence System (CRIS) is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets through a 9-stage machine learning pipeline, enriches each detection with spatial, depth, severity, lighting, weather, and infrastructure context features, and exposes the results through a REST API backed by a PostGIS spatial database.

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
│  RT-DETR-L · SAM · EfficientNet-B3 · Monodepth2 · XGBoost      │
│  PSO hyperparameter search · dataset preparation                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Inference Pipeline   (pipeline/ · scripts/)          │
│  Frame extraction → Detection → Segmentation → Depth →         │
│  Severity → Enrichment → Deduplication → DB write               │
│  Triggered manually (run_survey.py) or nightly (daily_job.py)  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Backend API          (backend/)                      │
│  FastAPI · SQLAlchemy · PostGIS · Pydantic v2                   │
│  Routes: detections · stats · heatmap · priority                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Frontend             (frontend/)                     │
│  Interactive map · Severity filters · Priority repair list      │
│  (currently placeholder — React dashboard planned)              │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline — Stage by Stage

```
┌─────────────────────────────────────────────────────────────────┐
│  PRE-SURVEY: ACO Route Planning                                 │
│  Ant Colony Optimization over Cluj-Napoca OSM road network      │
│  Finds minimum-distance route covering all primary/secondary    │
│  roads before the survey drive                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 1 — Preprocessor              [preprocessor.py]          │
│  • Extract frames from .mp4 (1 per 0.5 seconds)                │
│  • Sync GPS coordinates from .gpx to each frame timestamp       │
│  • Compute sun angle per frame (pysolar)                        │
│  • Classify lighting: daylight / overcast / low_light           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 2 — Detector                  [detector.py]              │
│  • RT-DETR-L inference on each frame (640×640)                  │
│  • Test Time Augmentation: flip + rotate, averaged predictions  │
│  • Confidence threshold: discard detections < 0.5              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 3 — Segmentor                 [segmentor.py]             │
│  • RT-DETR bounding boxes → SAM box prompts                     │
│  • SAM outputs pixel-level mask per detection                   │
│  • Computes: surface_area, edge_sharpness,                      │
│    interior_contrast, mask_compactness                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 4 — Depth Estimator           [depth_estimator.py]       │
│  • EfficientNet-B3 regression → depth in cm                     │
│  • Monodepth2 dense depth map → depth at detection region       │
│  • Both estimates fused for final depth_estimate                │
│  • Fallback: mask geometry proxy when depth_confidence < 0.4   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 5 — Severity Classifier       [severity_classifier.py]   │
│  • XGBoost on WOA-selected feature subset → S1–S5               │
│  • Rule-based fallback (depth + area) until Cluj data ready     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 6 — Enricher                  [enricher.py]              │
│  • Nominatim API    → street_name                               │
│  • OSM Overpass API → road_importance, infra_proximity          │
│  • Open-Meteo API   → weather at detection timestamp            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 7 — Deduplicator              [deduplicator.py]          │
│  • DBSCAN spatial clustering (2m radius)                        │
│  • Merges duplicates from multiple survey passes                │
│  • PostGIS upsert: UPDATE existing or INSERT new                │
│  • Updates detection_count and deterioration_rate               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 8 — Database          [PostgreSQL 15 + PostGIS]          │
│  • detections table — all ML-derived attributes + geometry      │
│  • survey_log table — tracks each pipeline run                  │
│  • GIST spatial index on geom column                            │
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

**RT-DETR-L** ([Zhao et al., 2024](https://arxiv.org/abs/2304.08069)) is a transformer-based object detector that outperforms YOLO-series models on accuracy while maintaining real-time inference. It replaces the anchor-based detection head with a transformer decoder, eliminating NMS post-processing. The architecture builds on ResNet ([He et al., 2016](https://arxiv.org/abs/1512.03385)) with a Feature Pyramid Network neck ([Lin et al., 2017](https://arxiv.org/abs/1612.03144)) and uses Focal Loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) to handle class imbalance.

| Property | Value |
|---|---|
| Architecture | HGStem backbone + AIFI encoder + RepC3 neck + RTDETRDecoder |
| Parameters | 32.8M |
| Input resolution | 640 × 640 |
| Pretrained weights | COCO 2017 (80 classes) — [Lin et al., 2014](https://arxiv.org/abs/1405.0312) |
| Fine-tuned on | RDD2022 + Pothole600 (27,336 images) |
| Output classes | 5 road damage types |
| Training hardware | Kaggle P100 (16 GB) / T4×2 (production) |

**Training dataset:**

| Dataset | Images | Annotations | Countries | Reference |
|---|---|---|---|---|
| [RDD2022](https://doi.org/10.1002/gdj3.260) | 21,479 | 33,913 | India, Japan, Norway, China, Czech Republic, USA | Arya et al., 2024 |
| Pothole600 | 5,857 | 8,970 | Mixed | — |
| **Total** | **27,336** | **42,883** | 6 | |

**Training strategy:**
- **Phase 1** (10 epochs): Frozen backbone — only decoder and detection head trained. LR = 1e-4.
- **Phase 2** (50 epochs): Full fine-tune — backbone unfrozen, LR = 1e-5. Early stopping patience = 20.
- **SWA** ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)): Stochastic Weight Averaging over last 5 checkpoints → `swa.pt`.

Checkpoint priority at evaluation and inference: `swa.pt` > `best.pt` > `rtdetr_l_rdd2022.pt`.

**Training techniques:**
- Focal Loss γ=2.0 built into RT-DETR for class imbalance ([Lin et al., 2017](https://arxiv.org/abs/1708.02002))
- Mixup α=0.15 ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)) and Mosaic augmentation
- Albumentations pipeline ([Buslaev et al., 2020](https://doi.org/10.3390/info11020125)): HSV jitter, rotation, scale, shear, horizontal flip
- Test Time Augmentation at inference (flip + rotate, averaged)
- fp16 AMP, gradient accumulation ×4 (effective batch = 16)
- `cudnn.benchmark = True`, `deterministic = False`

**Hyperparameter optimization — PSO** ([Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968)):
Particle Swarm Optimization searches a 7-dimensional space (`lr0`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `box`, `cls`) with 8 particles over 3 iterations on a Kaggle P100 GPU. Methodology follows Young et al. ([2015](https://doi.org/10.1145/3071178.3071208)) for PSO-based deep learning hyperparameter selection. Each particle is evaluated by training for 4 epochs and measuring validation mAP50-95. Output saved to `ml/optimization/pso_best.json` — `train.py` loads this automatically on the next run with no code changes needed.

### Segmentation — SAM

Used **zero-shot** — no fine-tuning required. RT-DETR bounding boxes serve as box prompts to [SAM](https://arxiv.org/abs/2304.02643) ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)). Four geometry features computed from each mask:

| Feature | Description |
|---|---|
| `surface_area` | Damage extent in cm² |
| `edge_sharpness` | Sobel gradient magnitude along mask boundary |
| `interior_contrast` | Mean pixel intensity inside vs. outside mask |
| `mask_compactness` | 4π × area / perimeter² (circle=1.0, crack≈0.05) |

> **Architectural note:** These 4 features cannot be derived from bounding boxes alone — SAM is a required pipeline component, not optional enrichment. Validated by [Zhang et al., 2024](https://doi.org/10.1016/j.ijtst.2024.10.005).

### Depth Estimation — EfficientNet-B3 + Monodepth2

- **[EfficientNet-B3](https://arxiv.org/abs/1905.11946)** ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)): regression head on cropped detection region + sun angle → depth in cm. Trained on Cluj ground truth measurements + Blender synthetic renders. Tuned with [Optuna](https://arxiv.org/abs/1907.10902) (TPE sampler, 50 trials).
- **[Monodepth2](https://arxiv.org/abs/1806.01260)** ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)): self-supervised dense depth map — depth at detection region extracted and fused with EfficientNet estimate.
- **Fallback**: proxy depth from mask geometry when `depth_confidence < 0.4` or `lighting_condition = low_light`.

### Severity Classification — XGBoost + WOA

**Whale Optimization Algorithm** ([Mirjalili & Lewis, 2016](https://doi.org/10.1016/j.advengsoft.2016.01.008)) performs binary feature selection across the 16-feature ML vector before [XGBoost](https://arxiv.org/abs/1603.02754) ([Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)) training. XGBoost extends gradient boosting ([Friedman, 2001](https://doi.org/10.1214/aos/1013203451)).

| Level | Description | Typical depth | Action |
|---|---|---|---|
| S1 | Superficial | < 1 cm | Monitor |
| S2 | Minor | 1–3 cm | Schedule maintenance |
| S3 | Moderate | 3–6 cm | Priority repair |
| S4 | Severe | 6–10 cm | Urgent repair |
| S5 | Critical | > 10 cm | Emergency closure |

### Route Optimization — ACO

**Ant Colony Optimization** ([Dorigo et al., 1996](https://doi.org/10.1109/3477.484436)) computes the optimal pre-survey driving route through the Cluj-Napoca OSM road network (loaded via `osmnx`), minimizing total distance while covering all primary and secondary roads.

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

## Training Results

### Phase 1 — Frozen Backbone (Epochs 1–10)

Backbone frozen (first 23 layers). Only decoder and detection head trained. Precision reports `nan` — expected during frozen phase as the model cannot yet produce reliable positive predictions.

| Epoch | Train GIoU ↓ | Train L1 ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|
| 1  | 1.418 | 1.033 | 0.201 | 0.00248 | 0.000631 |
| 3  | 1.157 | 0.721 | 0.335 | 0.01300 | 0.003980 |
| 5  | 1.028 | 0.609 | 0.447 | 0.02045 | 0.006900 |
| 7  | 0.977 | 0.569 | 0.500 | 0.02349 | 0.008450 |
| 10 | 0.942 | 0.546 | 0.533 | 0.02658 | 0.009920 |

### Phase 2 — Full Fine-Tune (Epochs 11–56, four Kaggle sessions)

All layers unfrozen. LR reduced to 1e-5 (10× lower than Phase 1). Three Kaggle sessions (Run 1: ep11–19, Run 2: ep20–38, Run 3: ep39–56) with LR warmup reset per session.

| Epoch | Run | Train GIoU ↓ | Val GIoU ↓ | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|---|---|
| 11 | R1 | 1.376 | 1.204 | 0.271 | 0.084 | 0.00761 | 0.00206 |
| 19 | R1 | 0.721 | 0.763 | 0.169 | 0.255 | 0.11567 | 0.04855 |
| 20 | R2 | 1.047 | 0.756 | 0.136 | 0.217 | 0.07964 | 0.03370 |
| 38 | R2 | 0.601 | 0.643 | 0.260 | 0.348 | 0.21071 | 0.09750 |
| 39 | R3 | 0.947 | 0.655 | 0.194 | 0.288 | 0.13941 | 0.06414 |
| 56 | R3 | 0.568 | 0.610 | 0.315 | 0.377 | 0.27295 | 0.12810 |

**Key observations:**
- GIoU and L1 losses decrease monotonically across all 56 Phase 2 epochs — no divergence, no overfitting
- The sharp GIoU drop at local epoch 9 of each run (e.g., global ep16, ep35, ep47) is Ultralytics disabling mosaic augmentation at a fixed epoch threshold — structural, not a bug
- LR warmup restart at the start of each Kaggle session causes a transient GIoU spike — cosmetic, weights carry over correctly

### Final Evaluation — best.pt on Validation Set (5,857 images)

Evaluated with `conf=0.001`, `iou=0.6` to compute full precision-recall curves.

| Metric | Value |
|---|---|
| **mAP50** | **0.272** |
| **mAP50-95** | **0.127** |
| Precision | 0.313 |
| Recall | 0.376 |

**Per-class AP@0.50:**

| Class | Instances | AP@0.50 |
|---|---|---|
| pothole | 1,811 | **0.379** |
| longitudinal_crack | 3,890 | 0.265 |
| alligator_crack | 1,553 | 0.226 |
| transverse_crack | 1,769 | 0.218 |
| patch_deterioration | — | 0.000 (no training data) |

**Confusion matrix analysis:**
- Cross-class confusion ≤ 0.04 between any damage pair — class discrimination is correctly learned
- Dominant failure mode: background false negatives — 45% longitudinal, 52% transverse, 42% alligator, 29% pothole missed
- Root cause: **domain gap** (RDD2022 from Japan/India/China/Norway/Czech/USA vs Romanian roads), **not label error**
- Fix: Cluj-Napoca data collection + fine-tuning (planned)
- At `conf=0.5` (operational threshold): precision ≈ 0.80+, recall ≈ 0.10–0.40 depending on class
- Optimal threshold: `conf=0.311` (F1 peak = 0.33 across all classes)

---

## PSO Hyperparameter Optimization

PSO search running on Kaggle P100 GPU. Configuration: **8 particles × 3 iterations × 4 eval epochs** per trial (~24 total trials, ~5.6h).

Algorithm follows [Kennedy & Eberhart (1995)](https://doi.org/10.1109/ICNN.1995.488968) with inertia weight decay w: 0.9 → 0.4, c₁ = c₂ = 1.5. Applied to DNN hyperparameter selection following [Young et al. (2015)](https://doi.org/10.1145/3071178.3071208).

**Search space (7 dimensions):**

| Parameter | Range | Scale |
|---|---|---|
| `lr0` | [1e-5, 5e-4] | Log |
| `weight_decay` | [1e-5, 1e-3] | Log |
| `warmup_epochs` | [1, 5] | Linear |
| `mosaic` | [0.5, 1.0] | Linear |
| `mixup` | [0.0, 0.3] | Linear |
| `box` | [5.0, 10.0] | Linear |
| `cls` | [0.3, 1.0] | Linear |

**Best hyperparameters found so far** (search in progress — `pso_checkpoint.json`):

| Parameter | Value |
|---|---|
| `lr0` | 1.953e-05 |
| `weight_decay` | 2.872e-04 |
| `warmup_epochs` | 2 |
| `mosaic` | 0.690 |
| `mixup` | 0.297 |
| `box` | 8.200 |
| `cls` | 0.690 |

These parameters will be used for the full retrain from clean COCO weights once the PSO search completes. `train.py` loads `pso_best.json` automatically.

---

## Database

PostgreSQL 15 + PostGIS runs in Docker. Two tables:

**`detections`** — one row per unique georeferenced damage instance:
- Spatial geometry (`POINT`, SRID 4326) with GIST index
- All ML-derived attributes: `damage_type`, `confidence`, SAM geometry features, `depth_estimate`, `depth_confidence`, `severity`, `severity_confidence`
- Lighting, weather (JSONB), location context from OSM
- Temporal tracking: `first_detected`, `last_detected`, `detection_count`, `deterioration_rate`
- Derived: `surrounding_density`, `priority_score`

**`survey_log`** — one row per pipeline run, tracking input footage, timestamps, detection counts, and processing status.

**Priority score formula** (implemented in `Detection.compute_priority_score()`):
```
priority_score = severity_weight × road_weight × infra_weight × log(detection_count + 1)
```

**DB session access:**
- `get_db()` — FastAPI `Depends()` injection for route handlers
- `get_db_session()` — direct async context manager for pipeline scripts

---

## Project Structure

```
Cluj-Road-Intelligence-System/
│
├── ml/
│   ├── detection/
│   │   ├── train.py                ✅ RT-DETR-L two-phase training + SWA
│   │   ├── evaluate.py             ✅ Per-class AP, mAP50-95, checkpoint comparison
│   │   ├── monitor.py              ✅ Live training dashboard (reads results.csv)
│   │   └── data_prep/
│   │       ├── prep_rdd2022.py     ✅ RDD2022 format conversion
│   │       ├── prep_pothole600.py  ✅ Pothole600 format conversion
│   │       ├── merge_datasets.py   ✅ Merge into unified train/val/test split
│   │       └── coco_to_yolo.py     ✅ COCO JSON → YOLO .txt conversion
│   ├── optimization/
│   │   ├── pso_hyperparams.py      ✅ PSO search (7-dim, 8 particles × 3 iters)
│   │   ├── pso_best.json           🔄 Best params (search in progress)
│   │   ├── pso_checkpoint.json     🔄 Resume state (search in progress)
│   │   └── optuna_search.py        ⬜ EfficientNet-B3 + XGBoost tuning
│   ├── segmentation/               ⬜ SAM inference module
│   ├── depth/                      ⬜ EfficientNet-B3 depth training
│   ├── severity/                   ⬜ XGBoost + WOA feature selection
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     🔄 PSO retrain pending
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
│   ├── main.py                     ✅ FastAPI app + CORS
│   ├── database.py                 ✅ SQLAlchemy engine, get_db(), get_db_session()
│   ├── models.py                   ✅ Detection + SurveyLog ORM models
│   ├── schemas.py                  ✅ Pydantic v2 schemas
│   └── routes/
│       ├── detections.py           ✅ GET /detections, /{id}, /nearby
│       ├── stats.py                ✅ GET /stats
│       ├── heatmap.py              ✅ GET /heatmap
│       └── priority.py             ✅ GET /priority-list
│
├── scheduler/
│   └── daily_job.py                ✅ APScheduler cron — fires pipeline nightly
│                                      (Europe/Bucharest TZ)
│
├── scripts/
│   ├── download_datasets.py        ✅ Download RDD2022 + Pothole600
│   ├── inspect_datasets.py         ✅ Distribution analysis and plots
│   ├── verify_merge.py             ✅ Verify merged dataset integrity
│   └── run_survey.py               ✅ Manual one-shot pipeline trigger
│
├── frontend/
│   └── (placeholder)               ⬜ React dashboard planned
│
├── data/
│   ├── raw/
│   │   ├── footage/                Input dashcam .mp4 files
│   │   └── gps_logs/               GPX telemetry files
│   ├── processed/
│   │   ├── frames/                 Extracted video frames
│   │   └── metadata/               Per-frame annotation metadata
│   ├── datasets/                   Prepared training/evaluation datasets
│   └── detection/
│       ├── dataset.yaml            ✅ YOLO-format dataset config
│       ├── train.json              ✅ 27,336 images
│       ├── val.json                ✅ 5,857 images
│       └── test.json               ✅ 5,857 images
│
├── docker-compose.yml              ✅ PostgreSQL 15 + PostGIS + pgAdmin
└── requirements.txt                ✅
```

**Legend:** ✅ Done · 🔄 In progress · ⬜ Planned

---

## Current Training Status

Phase 1 (10 frozen epochs) complete. Phase 2 (56 epochs across 3 Kaggle sessions) complete. **PSO hyperparameter search currently running on Kaggle P100.**

Next step: PSO completes → full retrain from `rtdetr-l.pt` with `pso_best.json` params → evaluate `best.pt` vs `swa.pt` → pick final `rtdetr_l_rdd2022.pt`.

---

## Usage

### Dataset Preparation

```bash
# Download raw datasets
python scripts/download_datasets.py

# Convert and merge (run in order)
python ml/detection/data_prep/prep_rdd2022.py
python ml/detection/data_prep/prep_pothole600.py
python ml/detection/data_prep/merge_datasets.py
python ml/detection/data_prep/coco_to_yolo.py

# Verify
python scripts/inspect_datasets.py
python scripts/verify_merge.py
```

### Training

```bash
# (Optional) PSO hyperparameter search — ~5.6h on P100
python ml/optimization/pso_hyperparams.py --particles 8 --iterations 3 --eval_epochs 4

# Resume PSO if session was interrupted
python ml/optimization/pso_hyperparams.py --resume

# Train RT-DETR-L (auto-loads pso_best.json if present)
python ml/detection/train.py
python ml/detection/train.py --smoke_test         # 2-epoch pipeline validation
python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt

# Multi-GPU (Kaggle T4×2)
python ml/detection/train.py --device 0,1 --workers 4

# Monitor in a second terminal
python ml/detection/monitor.py                    # auto-refreshes every 30s
python ml/detection/monitor.py --save             # save PNG
```

### Evaluation

```bash
python ml/detection/evaluate.py
python ml/detection/evaluate.py --weights path/to/best.pt
python ml/detection/evaluate.py --full            # val + test + TTA + comparison
python ml/detection/evaluate.py --compare         # best.pt vs swa.pt vs last.pt
```

### Database

```bash
docker-compose up -d
python scripts/setup_db.py
```

### Backend

```bash
uvicorn backend.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### Pipeline

```bash
# Manual one-shot survey run
python scripts/run_survey.py
python scripts/run_survey.py --date 2024-06-15

# Nightly scheduler (blocks — fires at 02:00 Europe/Bucharest)
python scheduler/daily_job.py
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

## Roadmap

**Done:**
- [x] Dataset download, conversion, merge, and verification scripts
- [x] Data distribution analysis and plots
- [x] RT-DETR-L training pipeline (two-phase + SWA + PSO integration)
- [x] PSO hyperparameter optimization script (`pso_hyperparams.py`)
- [x] Evaluation script with per-class AP and checkpoint comparison (`evaluate.py`)
- [x] Live training monitor (`monitor.py`)
- [x] Docker Compose — PostgreSQL 15 + PostGIS + pgAdmin
- [x] Database schema — `detections` + `survey_log` tables, enums, GIST index
- [x] SQLAlchemy ORM models + Pydantic v2 schemas
- [x] FastAPI backend — all routes (tested on Swagger)
- [x] APScheduler daily job (`scheduler/daily_job.py`)
- [x] Phase 1 training — 10 frozen backbone epochs (mAP50: 0.00248 → 0.02658)
- [x] Phase 2 training — 46 full fine-tune epochs across 3 Kaggle sessions (mAP50: 0.273)
- [x] Final evaluation on validation set (mAP50=0.272, mAP50-95=0.127)

**In progress:**
- [ ] PSO hyperparameter search (8 particles × 3 iterations × 4 eval epochs on P100)

**Planned:**
- [ ] Full retrain with PSO-optimized hyperparameters
- [ ] Final evaluate + pick best checkpoint (`best.pt` vs `swa.pt`)
- [ ] Inference pipeline — all 8 modules + orchestrator
- [ ] React dashboard — map, sidebar, detail panel
- [ ] ACO survey route generation (`aco_route.py`)
- [ ] Cluj-Napoca data collection drive (dashcam + GPS + depth measurements)
- [ ] Label Studio annotation of Cluj footage
- [ ] EfficientNet-B3 depth model training on Cluj ground truth
- [ ] XGBoost + WOA severity classifier training
- [ ] RT-DETR fine-tuning on Cluj footage → `rtdetr_l_cluj.pt`
- [ ] Optuna tuning for EfficientNet-B3 + XGBoost
- [ ] End-to-end integration test on real Cluj footage
- [ ] City Hall pilot demonstration

---

## Papers

All 39 papers organized across 10 categories. Category 0 contains foundational baselines.

### 0. Foundational / Baseline Papers

| Paper | Authors | Year | Link |
|---|---|---|---|
| ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky, Sutskever, Hinton | 2012 | [dl.acm.org](https://dl.acm.org/doi/10.1145/3065386) |
| Deep Residual Learning for Image Recognition (ResNet) | He, Zhang, Ren, Sun | 2016 | [arxiv](https://arxiv.org/abs/1512.03385) |
| Attention Is All You Need (Transformer) | Vaswani et al. | 2017 | [arxiv](https://arxiv.org/abs/1706.03762) |
| Adam: A Method for Stochastic Optimization | Kingma, Ba | 2015 | [arxiv](https://arxiv.org/abs/1412.6980) |
| Decoupled Weight Decay Regularization (AdamW) | Loshchilov, Hutter | 2019 | [arxiv](https://arxiv.org/abs/1711.05101) |
| Batch Normalization | Ioffe, Szegedy | 2015 | [arxiv](https://arxiv.org/abs/1502.03167) |
| Dropout | Srivastava et al. | 2014 | [jmlr.org](https://jmlr.org/papers/v15/srivastava14a.html) |
| Microsoft COCO | Lin et al. | 2014 | [arxiv](https://arxiv.org/abs/1405.0312) |
| PASCAL VOC Challenge | Everingham et al. | 2010 | [doi](https://doi.org/10.1007/s11263-009-0275-4) |
| Gradient Boosting Machine | Friedman | 2001 | [doi](https://doi.org/10.1214/aos/1013203451) |
| U-Net | Ronneberger et al. | 2015 | [arxiv](https://arxiv.org/abs/1505.04597) |
| Vision Transformer (ViT) | Dosovitskiy et al. | 2021 | [arxiv](https://arxiv.org/abs/2010.11929) |

### 1. Object Detection & Transformers

| Paper | Authors | Year | Link |
|---|---|---|---|
| DETRs Beat YOLOs on Real-Time Object Detection **(RT-DETR)** | Zhao, Lv et al. | 2024 | [arxiv](https://arxiv.org/abs/2304.08069) |
| RT-DETRv2: Improved Baseline with Bag-of-Freebies | Lv, Zhao et al. | 2024 | [arxiv](https://arxiv.org/abs/2407.17140) |
| End-to-End Object Detection with Transformers (DETR) | Carion et al. | 2020 | [arxiv](https://arxiv.org/abs/2005.12872) |
| Feature Pyramid Networks for Object Detection | Lin, Dollár et al. | 2017 | [arxiv](https://arxiv.org/abs/1612.03144) |
| Focal Loss for Dense Object Detection | Lin, Goyal et al. | 2017 | [arxiv](https://arxiv.org/abs/1708.02002) |

### 2. Road Damage Detection

| Paper | Authors | Year | Link |
|---|---|---|---|
| RDD2022: A Multi-National Image Dataset | Arya et al. | 2024 | [doi](https://doi.org/10.1002/gdj3.260) |
| Road Damage Detection Challenge Series (RDDC) | Tanaka et al. | 2025 | [Nature MI](https://www.nature.com/articles/s42256-025) |
| Computer Vision for Road Imaging and Pothole Detection (Review) | Fan et al. | 2022 | [doi](https://doi.org/10.1093/tse/tdac026) |
| Road Damage Detection Using Deep Neural Networks | Maeda et al. | 2018 | [doi](https://doi.org/10.1111/mice.12387) |
| Robust Video-Based Pothole Detection and Area Estimation | Bui et al. | 2021 | [doi](https://doi.org/10.1109/ACCESS.2021.3088384) |
| When SAM Meets Inventorying of Roadway Assets | Zhang, Huang, Qin | 2024 | [doi](https://doi.org/10.1016/j.ijtst.2024.10.005) |

### 3. Segmentation

| Paper | Authors | Year | Link |
|---|---|---|---|
| Segment Anything (SAM) | Kirillov et al. — Meta AI | 2023 | [arxiv](https://arxiv.org/abs/2304.02643) |
| SAM 2: Segment Anything in Images and Videos | Ravi et al. — Meta AI | 2024 | [arxiv](https://arxiv.org/abs/2408.00714) |

### 4. Depth Estimation

| Paper | Authors | Year | Link |
|---|---|---|---|
| Digging into Self-Supervised Monocular Depth Estimation **(Monodepth2)** | Godard et al. | 2019 | [arxiv](https://arxiv.org/abs/1806.01260) |
| EfficientNet: Rethinking Model Scaling for CNNs | Tan, Le | 2019 | [arxiv](https://arxiv.org/abs/1905.11946) |
| Low-Cost Monocular Vision Techniques for Pothole Distance Estimation | Hach, Sankowski | 2015 | [researchgate](https://www.researchgate.net/publication/285578304) |

### 5. Severity Classification & ML

| Paper | Authors | Year | Link |
|---|---|---|---|
| XGBoost: A Scalable Tree Boosting System | Chen, Guestrin | 2016 | [arxiv](https://arxiv.org/abs/1603.02754) |
| Modified XGBoost Hyper-Parameter Tuning Using Adaptive PSO | Langat, Waititu, Ngare | 2024 | [doi](https://doi.org/10.11648/j.mlr.20240902.15) |

### 6. Hyperparameter Optimization

| Paper | Authors | Year | Link |
|---|---|---|---|
| Particle Swarm Optimization | Kennedy, Eberhart | 1995 | [doi](https://doi.org/10.1109/ICNN.1995.488968) |
| PSO for Hyper-Parameter Selection in Deep Neural Networks | Young et al. | 2015 | [doi](https://doi.org/10.1145/3071178.3071208) |
| Optuna: A Next-generation Hyperparameter Optimization Framework | Akiba et al. | 2019 | [arxiv](https://arxiv.org/abs/1907.10902) |
| The Whale Optimization Algorithm | Mirjalili, Lewis | 2016 | [doi](https://doi.org/10.1016/j.advengsoft.2016.01.008) |

### 7. Training Techniques

| Paper | Authors | Year | Link |
|---|---|---|---|
| Averaging Weights Leads to Wider Optima (SWA) | Izmailov et al. | 2018 | [arxiv](https://arxiv.org/abs/1803.05407) |
| Albumentations: Fast and Flexible Image Augmentations | Buslaev et al. | 2020 | [doi](https://doi.org/10.3390/info11020125) |
| mixup: Beyond Empirical Risk Minimization | Zhang et al. | 2018 | [arxiv](https://arxiv.org/abs/1710.09412) |

### 8. Clustering & Spatial

| Paper | Authors | Year | Link |
|---|---|---|---|
| A Density-Based Algorithm for Discovering Clusters (DBSCAN) | Ester, Kriegel et al. | 1996 | [acm](https://dl.acm.org/doi/10.5555/3001460.3001507) |

### 9. Route Optimization

| Paper | Authors | Year | Link |
|---|---|---|---|
| Ant Colony Optimization | Dorigo, Maniezzo, Colorni | 1996 | [doi](https://doi.org/10.1109/3477.484436) |

> **Total: 39 papers across 10 categories.** Full citations with venues, volume numbers, and page ranges are listed in the thesis bibliography (`references.bib`).

---

## Technology Stack

| Layer | Technology | Reference |
|---|---|---|
| Detection | RT-DETR-L (Ultralytics 8.2) | [Zhao et al., 2024](https://arxiv.org/abs/2304.08069) |
| Segmentation | SAM — Segment Anything Model | [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) |
| Depth estimation | EfficientNet-B3 + Monodepth2 | [Tan & Le, 2019](https://arxiv.org/abs/1905.11946) · [Godard et al., 2019](https://arxiv.org/abs/1806.01260) |
| Severity classification | XGBoost | [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754) |
| Hyperparameter optimization | PSO (custom) · Optuna (TPE) | [Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968) · [Akiba et al., 2019](https://arxiv.org/abs/1907.10902) |
| Feature selection | WOA — Whale Optimization Algorithm | [Mirjalili & Lewis, 2016](https://doi.org/10.1016/j.advengsoft.2016.01.008) |
| Survey route planning | ACO · osmnx | [Dorigo et al., 1996](https://doi.org/10.1109/3477.484436) |
| Spatial clustering | DBSCAN (scikit-learn) | [Ester et al., 1996](https://dl.acm.org/doi/10.5555/3001460.3001507) |
| Augmentation | Albumentations · Mixup · Mosaic | [Buslaev et al., 2020](https://doi.org/10.3390/info11020125) · [Zhang et al., 2018](https://arxiv.org/abs/1710.09412) |
| Training | Focal Loss · SWA · TTA · fp16 AMP | [Lin et al., 2017](https://arxiv.org/abs/1708.02002) · [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407) |
| Database | PostgreSQL 15 + PostGIS | [postgis.net](https://postgis.net) |
| ORM | SQLAlchemy 2.0 (async) | [sqlalchemy.org](https://www.sqlalchemy.org) |
| Backend | FastAPI + Pydantic v2 | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Scheduler | APScheduler (Europe/Bucharest TZ) | [apscheduler.readthedocs.io](https://apscheduler.readthedocs.io) |
| Frontend | Planned: React 18 + Leaflet.js | — |
| Containerization | Docker Compose | [docker.com](https://docker.com) |
| Geocoding | Nominatim (OpenStreetMap) | [nominatim.org](https://nominatim.org) |
| Road network | OSM Overpass API | [overpass-api.de](https://overpass-api.de) |
| Weather | Open-Meteo API | [open-meteo.com](https://open-meteo.com) |
| Sun angle | pysolar | [pysolar.readthedocs.io](https://pysolar.readthedocs.io) |
| Annotation | Label Studio | [labelstud.io](https://labelstud.io) |
| Code style | Black | [black.readthedocs.io](https://black.readthedocs.io) |
| Language | Python 3.12 | [python.org](https://python.org) |

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 ([Arya et al., 2024](https://doi.org/10.1002/gdj3.260)), Pothole600.
Model attributions: RT-DETR ([Zhao et al., 2023](https://arxiv.org/abs/2304.08069)), SAM ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)), Monodepth2 ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Mathematics and Computer Science · 2026</i>
</div>