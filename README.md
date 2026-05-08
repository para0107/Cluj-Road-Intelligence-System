# Road Infrastructure Damage System (RIDS)

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
│  React 18 + Leaflet.js (in development)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline — Stage by Stage

```
┌─────────────────────────────────────────────────────────────────┐
│  PRE-SURVEY: ACO Route Planning                                 │
│  Ant Colony Optimization over Cluj-Napoca OSM road network      │
│  Finds minimum-distance route covering all primary/secondary    │
│  roads before the survey drive — deferred, future work          │
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
│  • sam_score (SAM self-predicted IoU) stored per detection      │
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
│  STAGE 5 — Severity Classifier       [severity_classifier.py]   │
│  • Rule-based classifier: normalised depth + SAM surface area   │
│  • Outputs S1–S5 severity levels deterministically              │
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
| Fine-tuned on | RDD2022 + Pothole600 (Run 1–2), N-RDD2024 (Run 3) |
| Output classes | 4 (RDD2022 run) → 10 (N-RDD2024 run) |
| Training hardware | Kaggle T4 × 2 (16 GB VRAM each) |

**Three training runs conducted:**

| Run | Dataset | Classes | mAP50 (val) | mAP50 (test) | Notes |
|---|---|---|---|---|---|
| 1 — Baseline | RDD2022 + Pothole600 | 4 | 0.272 | — | Default hyperparameters, 56 epochs |
| 2 — PSO-optimised | RDD2022 + Pothole600 | 4 | **0.465** | **0.458** | PSO best params, 50 epochs, +72% over baseline |
| 3 — N-RDD2024 | N-RDD2024 (re-annotation) | 10 | **0.577** | — | Best model, used in all pipeline validation |

**Training techniques:**
- Focal Loss γ=2.0 built into RT-DETR for class imbalance ([Lin et al., 2017](https://arxiv.org/abs/1708.02002))
- Mixup α=0.205 ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)) and Mosaic augmentation (p=0.860)
- Albumentations pipeline ([Buslaev et al., 2020](https://doi.org/10.3390/info11020125)): HSV jitter, rotation, scale, shear, horizontal flip
- Label smoothing 0.1 for N-RDD2024 run ([Szegedy et al., 2016](https://doi.org/10.1109/CVPR.2016.308)) — addresses 442:1 class imbalance
- TTA evaluated — **zero mAP gain** on this dataset — disabled in pipeline
- fp16 AMP, gradient accumulation ×4 (effective batch = 16)

**Hyperparameter optimization — PSO** ([Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968)):

PSO searched a 7-dimensional space with 10 particles × 4 iterations × 2 eval epochs (~9.3h on Kaggle P100). Methodology follows [Young et al. (2015)](https://doi.org/10.1145/3071178.3071208).

| Parameter | Default | PSO Best | Δ |
|---|---|---|---|
| `lr0` | 1.0e-04 | 4.47e-04 | +347% |
| `weight_decay` | 5.0e-04 | 5.27e-04 | +5.3% |
| `warmup_epochs` | 3 | 1 | −67% |
| `mosaic` | 1.0 | 0.860 | −14% |
| `mixup` | 0.15 | 0.205 | +37% |
| `box` | 7.5 | 7.685 | +2.5% |
| `cls` | 0.5 | 0.487 | −2.6% |

The most significant finding is the 4.5× increase in `lr0`, allowing the decoder to adapt faster during the frozen-backbone phase.

---

### Segmentation — SAM 2.1 Tiny

**SAM 2.1** ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)) is used **zero-shot** — no fine-tuning required. RT-DETR bounding boxes serve as box prompts. The implementation was validated on 4,107 real Cluj-Napoca dashcam frames, producing geometry features for 1,904 accepted detections across 10 classes.

**Model variant: `sam2.1_hiera_tiny.pt`** (38 MB)

The Tiny variant was selected for three reasons:
1. Sequential single-image inference — per-call latency is the bottleneck, not batch throughput
2. Box-prompted mode — spatial prior from RT-DETR bbox limits the benefit of larger variants
3. 4 GB VRAM constraint (RTX 2050) — Base/Large/Huge do not fit alongside RT-DETR-L

**Architecture (Ravi et al., 2024):** Hiera hierarchical vision transformer backbone producing multi-scale embeddings at four resolution levels → prompt encoder → two-layer transformer mask decoder → binary mask + predicted IoU score. Memory attention module (video propagation) is inactive in single-image mode.

**Four geometry features computed per mask:**

| Feature | Field | Description |
|---|---|---|
| Surface area | `surface_area_px` | Raw pixel count — ordinal damage-extent proxy |
| Edge sharpness | `edge_sharpness` | Mean Sobel gradient magnitude on mask boundary dilation ring |
| Interior contrast | `interior_contrast` | Mean intensity difference inside mask vs erosion ring |
| Mask compactness | `mask_compactness` | 4π·area/perimeter² (circle=1.0, rutting=0.027) |

**SAM score** (`sam_score`): SAM's self-predicted IoU — second return value of `sam_predictor.predict()`. Used as mask reliability signal; detections with `sam_score < min_sam_score` have geometry set to `None`.

**Empirical compactness ranking from Run 3 validation (1,904 detections):**

| Class | Mean compactness | Morphology confirmed |
|---|---|---|
| `pothole` | 0.500 | Most circular — bowl shape |
| `manhole_cover` | 0.395 | Near-rectangular |
| `alligator_crack` | 0.313 | Irregular network |
| `longitudinal_crack` | 0.293 | Elongated |
| `transverse_crack` | 0.160 | Wide but thin |
| `rutting` | 0.027 | Most elongated — long groove |

The compactness ranking matches visual morphology without any supervision — validated by [Zhang et al., 2024](https://doi.org/10.1016/j.ijtst.2024.10.005).

---

### Depth Estimation — Monodepth2

**Monodepth2** ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)) is a self-supervised monocular depth estimation model pretrained on KITTI outdoor driving data. It produces a dense relative disparity map per frame (higher disparity = closer to camera).

**Model variant: `mono_640x192`** — lightest checkpoint; VRAM ~0.8 GB; sufficient for ordinal severity proxy use. Validated on 5 Cluj-Napoca frames with 9 total detections, producing physically consistent depth maps (road surface near-field = bright, sky/buildings = dark in magma colormap).

**Three depth extraction paths per detection (in priority order):**

| Path | Condition | Method |
|---|---|---|
| `mask_region` | `geometry != None` and `low_sam_quality = False` | Mean disparity over approximate mask (ellipse from compactness) |
| `central_crop` | No geometry available | Mean disparity over central 60% of bounding box crop |
| `geometry_proxy` | `depth_confidence < 0.4` or `lighting == low_light` | Heuristic from `surface_area_px` × `mask_compactness` |

**Depth confidence** = 1 − clipped coefficient of variation of depth values in extraction region. High spatial variance → low confidence → geometry proxy fallback.

**Sample depth_norm values from Run 3 validation:**

| Class | depth_norm (mean) | Interpretation |
|---|---|---|
| `manhole_cover` | 0.700 | Near-field, flush with road |
| `pothole` | 0.541 | Mid-field, moderate depth |
| `longitudinal_crack` | 0.503 | Mid-field, surface-level |
| `patchy_road` | 0.489 | Mid-field, flat patch |
| `lane_line_blur` | 0.414–0.752 | Variable — marking position on road |

> **Note:** `depth_norm` is per-frame relative depth, not metric depth. Values are only comparable within the same frame.

---

### Severity Classification — Rule-based

Severity is assigned deterministically from normalised Monodepth2 depth and SAM surface area. No training data required.

| Level | Depth proxy (normalised) | Surface area | Action |
|---|---|---|---|
| S1 | < 0.1 | Any | Monitor |
| S2 | 0.1–0.3 | < 500 cm² | Schedule maintenance |
| S3 | 0.3–0.6 | 500–2000 cm² | Priority repair |
| S4 | 0.6–0.8 | > 2000 cm² | Urgent repair |
| S5 | > 0.8 | Any | Emergency closure |

Marking classes (`pedestrian_crossing_blur`, `lane_line_blur`) are always assigned S1 regardless of signals. Infrastructure class (`manhole_cover`) is stored as spatial reference only.

> **Design note:** XGBoost + WOA severity classification was evaluated as a design candidate but determined outside project scope — requires labelled per-detection severity measurements from a supervised field programme.

---

### Route Optimization — ACO (deferred)

**Ant Colony Optimization** ([Dorigo et al., 1996](https://doi.org/10.1109/3477.484436)) was evaluated for pre-survey route optimisation over the Cluj-Napoca OSM road network (via `osmnx`). Deferred as future work — depends on complete operational deployment.

---

## Detection Classes

**RDD2022 run (4 classes):**

| ID | Class | AP@50 (test) |
|---|---|---|
| 0 | `longitudinal_crack` | 0.174 |
| 1 | `transverse_crack` | 0.126 |
| 2 | `alligator_crack` | 0.231 |
| 3 | `pothole` | 0.313 |

**N-RDD2024 run (10 classes, `rtdetr_l_nrdd2024.pt`, mAP50=0.577):**

| ID | D-code | Class | Category | Train instances |
|---|---|---|---|---|
| 0 | D00 | `longitudinal_crack` | Damage | 18,163 |
| 1 | D10 | `transverse_crack` | Damage | 8,116 |
| 2 | D20 | `alligator_crack` | Damage | 6,354 |
| 3 | D30 | `repaired_crack` | Damage | 311 |
| 4 | D40 | `pothole` | Damage | 3,155 |
| 5 | D50 | `pedestrian_crossing_blur` | Marking (S1) | 711 |
| 6 | D60 | `lane_line_blur` | Marking (S1) | 3,966 |
| 7 | D70 | `manhole_cover` | Infrastructure | 3,013 |
| 8 | D80 | `patchy_road` | Damage | 704 |
| 9 | D90 | `rutting` | Damage | 41 |

---

## Training Results

### RDD2022 Baseline Run

**Final validation results (`best.pt`, 5,857 images):**

| Metric | Baseline | PSO-optimised | Δ |
|---|---|---|---|
| mAP50 | 0.272 | **0.465** | +71% |
| mAP50-95 | 0.127 | **0.215** | +69% |
| Precision | 0.313 | **0.545** | +74% |
| Recall | 0.376 | **0.464** | +23% |
| F1 | — | **0.501** | — |

**Test set (held-out, 5,758 images) — PSO `best.pt`:**

| Metric | Val | Test | Drop |
|---|---|---|---|
| mAP50 | 0.4650 | **0.4583** | −0.007 |
| mAP50-95 | 0.2151 | **0.2110** | −0.004 |
| Precision | 0.5450 | **0.5367** | −0.008 |
| Recall | 0.4639 | **0.4633** | −0.001 |

Small val→test drop confirms no overfitting to the validation set.

### N-RDD2024 Run (Chain Fine-Tune Ablation)

Three checkpoints were evaluated to determine the optimal training strategy:

| Checkpoint | Training data | mAP50 (val) | Notes |
|---|---|---|---|
| `rtdetr_l_rdd2022.pt` | RDD2022 + Pothole600 | 0.465 | PSO-optimised baseline |
| `rtdetr_l_nrdd2024.pt` | N-RDD2024 only | **0.577** | **Best model — used in pipeline** |
| `rtdetr_l_rdd2022_nrdd2024.pt` | RDD2022 → N-RDD2024 chain | 0.419 | Chain hypothesis refuted |

The chain fine-tune (RDD2022 → N-RDD2024) performed worse than training on N-RDD2024 directly, confirming that the RDD2022 weights act as a negative prior for the expanded 10-class schema. `rtdetr_l_nrdd2024.pt` is the selected operational checkpoint.

### Confusion Matrix Analysis (PSO RDD2022 run)

Background false-negative rates (domain gap signature):

| Class | Correctly detected | Missed as background | FN rate |
|---|---|---|---|
| `longitudinal_crack` | 2,093 | 1,747 | **45.5%** |
| `transverse_crack` | 828 | 913 | **52.4%** |
| `alligator_crack` | 830 | 640 | **43.5%** |
| `pothole` | 1,276 | 528 | **29.3%** |

Cross-class confusion ≤4% in all cases. Primary failure is damage-vs-background, not class-to-class. Root cause: RDD2022 sourced from Japan/India/China/Norway/Czech/USA — Romanian road surfaces differ substantially.

---

## Validation on Real-World Dashcam Footage

### Run 1 — RDD2022 Model (4 classes)

Applied `rtdetr_l_rdd2022.pt` (`best.pt`, mAP50=0.458 test) to Cluj-Napoca and Tokyo dashcam footage at 2 fps, `conf=0.35`.

| | Cluj-Napoca [`uBQtQbdbv4E`](https://www.youtube.com/watch?v=uBQtQbdbv4E) | Tokyo [`v7JZ9DSSRsY`](https://www.youtube.com/watch?v=v7JZ9DSSRsY) |
|---|---|---|
| Frames processed | 4,107 | 3,269 |
| Frames with detections | 165 (**4.0%**) | 196 (6.0%) |
| Total bounding boxes | 191 | 237 |
| Mean confidence | 0.421 | 0.450 |
| FP rate (manual review) | ~51% | ~95% |

**Key finding:** High FP rate — model not calibrated to Romanian road surfaces.

---

### Run 2 — N-RDD2024 Model (10 classes, manual 100-frame sample)

Applied `rtdetr_l_nrdd2024.pt` to the same Cluj footage. 100-frame manual review:

| | Run 1 (RDD2022) | Run 2 (N-RDD2024) |
|---|---|---|
| Detection rate | 4.0% | ~29.8% |
| Mean confidence | 0.421 | 0.510 |
| Damage FP rate | ~51% | ~25–29% |
| Marking classes | — | ✅ Detected (D50/D60/D70) |

Road markings (pedestrian crossings, lane lines) now correctly classified — Failure Mode 2 from Run 1 substantially addressed.

---

### Run 3 — Full Pipeline (RT-DETR + SAM 2.1, 4,107 frames)

First end-to-end execution of Stages 1–4 on real Cluj-Napoca footage using `rtdetr_l_nrdd2024.pt` + `sam2.1_hiera_tiny.pt`.

**Frame-level statistics:**

| Metric | Value |
|---|---|
| Frames processed | 4,107 |
| Frames with detections | 1,330 (**32.4%**) |
| Total boxes accepted | 1,904 |
| Total boxes dropped | 1,230,196 |
| Mean confidence (accepted) | 0.501 |
| Elapsed time | 3,729.8 s (~1.04 h) |

**Per-class distribution:**

| Class | Count | Share | Mean conf | Mean SAM score |
|---|---|---|---|---|
| `pedestrian_crossing_blur` | 682 | 35.8% | 0.491 | 0.721 |
| `longitudinal_crack` | 632 | 33.2% | 0.487 | 0.684 |
| `lane_line_blur` | 163 | 8.6% | 0.583 | 0.789 |
| `pothole` | 136 | 7.1% | 0.485 | 0.772 |
| `manhole_cover` | 115 | 6.0% | 0.530 | 0.833 |
| `transverse_crack` | 103 | 5.4% | 0.471 | 0.493 |
| `patchy_road` | 47 | 2.5% | 0.593 | 0.886 |
| `repaired_crack` | 15 | 0.8% | 0.512 | 0.748 |
| `alligator_crack` | 10 | 0.5% | 0.432 | 0.785 |
| `rutting` | 1 | 0.1% | 0.396 | 0.721 |

**Box categories:**
- Structural damage (`is_damage=True`): 944 boxes (49.6%)
- Road markings (`is_marking=True`): 845 boxes (44.4%)
- Infrastructure (`is_infrastructure=True`): 115 boxes (6.0%)

**Detection rate progression across all runs:**

| Run | Model | Detection rate | Classes |
|---|---|---|---|
| Run 1 | `rtdetr_l_rdd2022.pt` | 4.0% | 4 |
| Run 2 (sample) | `rtdetr_l_nrdd2024.pt` | ~29.8% | 10 |
| Run 3 (full) | `rtdetr_l_nrdd2024.pt` + SAM 2.1 | **32.4%** | 10 |

8× improvement from Run 1 to Run 3. Road marking classes (D50/D60) account for 44.4% of Run 3 detections — the primary driver of the detection rate increase.

> **GPS note:** GPS coordinates in Run 3 are mock placeholder data (San Francisco). Spatial deduplication and heatmap generation require a real GPS-synchronised survey run.

---

### Depth Validation — Monodepth2 (5-frame sample)

First execution of Stage 4 on 5 frames from the Cluj footage (9 total detections). Three-panel visualisations saved to `data/validation_nrdd_2024/depth_maps/`.

| Frame | Detections | Depth range | Notes |
|---|---|---|---|
| `frame_000000_t0.000` | 4 (3× long. crack + manhole) | [0.023, 0.669] | Road surface correctly brighter (near) in magma |
| `frame_000007_t3.500` | 1 (pothole, depth_norm=0.541) | [0.024, 0.594] | Pothole mid-field detection — plausible depth |
| `frame_000023_t11.500` | 2 (lane_line + patchy_road) | [0.013, 0.713] | patchy_road depth_norm=0.489, SAM score=0.932 |
| `frame_000025_t12.500` | 1 (lane_line, depth_norm=0.752) | [0.019, 0.648] | Near-field marking — high depth_norm correct |
| `frame_000027_t13.500` | 1 (lane_line, depth_norm=0.414) | [0.039, 0.498] | Further marking — lower depth_norm correct |

Depth maps show physically consistent disparity gradients: near-field road surface (bottom of frame) is bright yellow/orange, sky and distant buildings are deep purple/black. Monodepth2 working correctly on Romanian urban driving scenes.

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

**Priority score formula:**
```
priority_score = severity_weight × road_weight × infra_weight × log(detection_count + 1)
```

---

## Project Structure

```
Cluj-Road-Intelligence-System/
│
├── ml/
│   ├── detection/
│   │   ├── train.py                ✅ RT-DETR-L two-phase training + SWA + PSO
│   │   ├── evaluate.py             ✅ Per-class AP, mAP50-95, checkpoint comparison
│   │   ├── monitor.py              ✅ Live training dashboard (reads results.csv)
│   │   └── data_prep/
│   │       ├── prep_rdd2022.py     ✅ RDD2022 XML → COCO JSON conversion
│   │       ├── prep_pothole600.py  ✅ Pothole600 → COCO JSON conversion
│   │       ├── merge_datasets.py   ✅ Merge into unified train/val/test split
│   │       └── coco_to_yolo.py     ✅ COCO JSON → YOLO .txt + dataset.yaml
│   ├── optimization/
│   │   ├── pso_hyperparams.py      ✅ PSO search (7-dim, 10 particles × 4 iters)
│   │   ├── pso_best.json           ✅ Best params found (completed)
│   │   └── pso_checkpoint.json     ✅ Full swarm state (resume support)
│   ├── evaluation/
│   │   ├── checkpoint_comparison_*.json  ✅ best.pt vs last.pt — best.pt selected
│   │   ├── eval_val_*.json         ✅ Val set: mAP50=0.465, mAP50-95=0.215
│   │   ├── eval_val_tta_*.json     ✅ Val+TTA: identical to standard (TTA disabled)
│   │   ├── eval_test_*.json        ✅ Test set: mAP50=0.458, mAP50-95=0.211
│   │   └── eval_*.log              ✅ Full evaluation execution logs
│   ├── severity/                   ⬜ Rule-based severity classifier
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     ✅ PSO-optimised, mAP50=0.465/test=0.458
│       ├── rtdetr_l_nrdd2024.pt    ✅ N-RDD2024, mAP50=0.577 — operational checkpoint
│       ├── sam2.1_hiera_tiny.pt    ✅ SAM 2.1 Tiny — zero-shot segmentation
│       ├── mono_640x192/
│       │   ├── encoder.pth         ✅ Monodepth2 encoder (KITTI pretrained)
│       │   └── depth.pth           ✅ Monodepth2 depth decoder
│       └── networks/               ✅ Monodepth2 architecture files (from repo)
│
├── runs/
│   └── detect/
│       └── rtdetr_road/
│           └── final_fine_tune/    ✅ Consolidated best run outputs
│               ├── best.pt         ✅ Selected checkpoint (mAP50-95=0.2151 val)
│               ├── last.pt         ✅ Final epoch checkpoint
│               ├── confusion_matrix.png  ✅ Confusion matrix (PSO run)
│               ├── results.xlsx    ✅ Merged training history (all 60 epochs)
│               └── *.png           ✅ F1/P/R/PR curves
│
├── pipeline/
│   ├── preprocessor.py             ✅ Frame extraction + GPS sync + lighting
│   ├── detector.py                 ✅ RT-DETR inference + per-class thresholds
│   ├── segmentor.py                ✅ SAM 2.1 masks + 4 geometry features
│   ├── depth_estimator.py          ✅ Monodepth2 relative depth + proxy fallback
│   ├── severity_classifier.py      ⬜ Rule-based S1–S5 classifier
│   ├── enricher.py                 ⬜ OSM + Nominatim + Open-Meteo
│   ├── deduplicator.py             ⬜ DBSCAN + PostGIS upsert
│   └── orchestrator.py             ⬜ End-to-end coordinator
│
├── backend/
│   ├── main.py                     ✅ FastAPI app + CORS
│   ├── database.py                 ✅ SQLAlchemy async engine, get_db()
│   ├── models.py                   ✅ Detection + SurveyLog ORM models
│   ├── schemas.py                  ✅ Pydantic v2 schemas
│   └── routes/
│       ├── detections.py           ✅ GET /detections, /{id}, /nearby
│       ├── stats.py                ✅ GET /stats
│       ├── heatmap.py              ✅ GET /heatmap
│       └── priority.py             ✅ GET /priority-list
│
├── scheduler/
│   └── daily_job.py                ✅ APScheduler cron — 02:00 Europe/Bucharest
│
├── scripts/
│   ├── download_datasets.py        ✅ Download RDD2022 + Pothole600
│   ├── inspect_datasets.py         ✅ Distribution analysis and plots
│   ├── verify_merge.py             ✅ Verify merged dataset integrity
│   ├── setup_db.py                 ✅ Create tables, enums, GIST index
│   ├── generate_pso_comparison_plot.py  ✅ PSO vs baseline training curves
│   ├── detect_and_sam.py           ✅ RT-DETR + SAM 2.1 validation script
│   ├── validate_depth.py           ✅ Monodepth2 depth validation + 3-panel vis
│   └── run_survey.py               ✅ Manual one-shot pipeline trigger
│
├── data/
│   ├── raw/
│   │   ├── footage/                Input dashcam .mp4 files
│   │   └── gps_logs/               GPX telemetry files
│   ├── processed/
│   │   └── frames/                 Extracted video frames + manifest.json
│   ├── validation_nrdd_2024/
│   │   ├── bounding_boxes/cluj/    ✅ Annotated frames (RT-DETR boxes)
│   │   ├── bounding_boxes/bounding_boxes_annotations/  ✅ Per-frame bbox JSON
│   │   ├── sam_masks/cluj/         ✅ SAM 2.1 mask overlays (1,904 detections)
│   │   ├── depth_maps/             ✅ Monodepth2 3-panel visualisations
│   │   ├── detections_summary.json ✅ Full Run 3 pipeline output
│   │   └── depth_validation.json   ✅ Stage 4 depth extraction results
│   └── detection/
│       ├── dataset.yaml            ✅ YOLO-format dataset config
│       ├── train.json              ✅ 27,336 images
│       ├── val.json                ✅ 5,857 images
│       └── test.json               ✅ 5,857 images
│
├── docker-compose.yml              ✅ PostgreSQL 15 + PostGIS + pgAdmin
└── requirements.txt                ✅ Pinned versions for local inference
```

**Legend:** ✅ Done · 🔄 In progress · ⬜ Planned

---

## Usage

### Dataset Preparation

```bash
python scripts/download_datasets.py
python ml/detection/data_prep/prep_rdd2022.py
python ml/detection/data_prep/prep_pothole600.py
python ml/detection/data_prep/merge_datasets.py
python ml/detection/data_prep/coco_to_yolo.py
python scripts/inspect_datasets.py
python scripts/verify_merge.py
```

### Training

```bash
# PSO hyperparameter search (~9.3h on P100)
python ml/optimization/pso_hyperparams.py --particles 10 --iterations 4 --eval_epochs 2

# Train RT-DETR-L
python ml/detection/train.py
python ml/detection/train.py --smoke_test
python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt
python ml/detection/train.py --device 0,1 --workers 4   # Kaggle T4×2

# Monitor
python ml/detection/monitor.py --save --interval 60
```

### Evaluation

```bash
python ml/detection/evaluate.py                  # val set
python ml/detection/evaluate.py --split test     # held-out test set
python ml/detection/evaluate.py --compare        # best.pt vs last.pt
python ml/detection/evaluate.py --full           # all reports (~60 min on T4)
```

### Validation Scripts

```bash
# RT-DETR + SAM 2.1 on full Cluj footage (Stages 1–3)
python scripts/detect_and_sam.py
python scripts/detect_and_sam.py --limit 50      # test run
python scripts/detect_and_sam.py --damage_only   # skip marking/infra SAM

# Monodepth2 depth validation (Stage 4 sanity check)
python scripts/validate_depth.py --setup         # print setup instructions
python scripts/validate_depth.py --device cuda   # full run on GPU
python scripts/validate_depth.py --limit 100     # first 100 detected frames
```

### Pipeline (module usage)

```python
from pipeline.preprocessor    import Preprocessor, PreprocessorConfig
from pipeline.detector         import Detector, DetectorConfig
from pipeline.segmentor        import Segmentor, SegmentorConfig
from pipeline.depth_estimator  import DepthEstimator, DepthEstimatorConfig

frames   = Preprocessor(PreprocessorConfig()).run(video_path, gps_path, output_dir)
det_res  = Detector(DetectorConfig(weights="ml/weights/rtdetr_l_nrdd2024.pt")).run(frames)
seg_res  = Segmentor(SegmentorConfig(weights="ml/weights/sam2.1_hiera_tiny.pt")).run(det_res)
dep_res  = DepthEstimator(DepthEstimatorConfig(device="cuda")).run(seg_res)
```

### Database

```bash
docker-compose up -d
python scripts/setup_db.py
uvicorn backend.main:app --reload   # Swagger UI: http://localhost:8000/docs
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

**Completed:**
- [x] Dataset download, conversion, merge, and verification scripts
- [x] RT-DETR-L training pipeline (two-phase + PSO integration)
- [x] PSO hyperparameter search (10 particles × 4 iterations, ~9.3h on P100)
- [x] Baseline training — 56 epochs (mAP50=0.272)
- [x] PSO-optimised training — 50 epochs (mAP50=0.468, **+72% over baseline**)
- [x] Formal evaluation — val mAP50=0.465, test mAP50=0.458, test mAP50-95=0.211
- [x] TTA evaluated — zero gain, disabled in pipeline
- [x] N-RDD2024 chain fine-tune ablation — 3 checkpoints evaluated
- [x] **N-RDD2024 model completed** — `rtdetr_l_nrdd2024.pt`, mAP50=0.577, 10 classes
- [x] Docker Compose — PostgreSQL 15 + PostGIS + pgAdmin
- [x] Database schema — `detections` + `survey_log`, enums, GIST index
- [x] FastAPI backend — all routes (tested on Swagger)
- [x] APScheduler daily job
- [x] `preprocessor.py` — frame extraction, GPS sync, lighting classification
- [x] `detector.py` — RT-DETR-L inference, per-class thresholds, save/load
- [x] **`segmentor.py` — SAM 2.1 Tiny, 4 geometry features, save/load**
- [x] **`depth_estimator.py` — Monodepth2, 3 extraction paths, proxy fallback**
- [x] **`detect_and_sam.py` — full RT-DETR + SAM validation on 4,107 Cluj frames**
- [x] **`validate_depth.py` — Monodepth2 3-panel validation with depth JSON**
- [x] **Run 3 full pipeline — 1,904 detections, 10 classes, SAM geometry confirmed**
- [x] **Monodepth2 depth validation — physically consistent depth maps on Cluj footage**

**In progress:**
- [ ] React frontend — map, sidebar, detail panel (React 18 + Leaflet.js)

**Planned:**
- [ ] `severity_classifier.py` — rule-based S1–S5
- [ ] `enricher.py` — OSM + Nominatim + Open-Meteo
- [ ] `deduplicator.py` — DBSCAN + PostGIS upsert
- [ ] `orchestrator.py` — end-to-end pipeline coordinator
- [ ] Real GPS survey run — dashcam + GPX synchronised
- [ ] TRIB dataset annotation → Romanian domain fine-tuning ([Abrudan, 2025](https://doi.org/10.1007/s12145-025-01763-7))
- [ ] ACO survey route generation — deferred, future work
- [ ] End-to-end integration test on real survey footage
- [ ] City Hall pilot demonstration

---

## Known Issues & Compatibility Notes

- **PyTorch CUDA:** The venv must use a CUDA-enabled PyTorch build. Verify with `python -c "import torch; print(torch.version.cuda)"`. If `None`, force-reinstall: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`. RTX 2050 with driver 555.97 (CUDA 12.5) is compatible with cu121 wheels.
- **Kaggle environment (training):** Ultralytics 8.3.143 required for PyTorch 2.6 compatibility. Pillow must be force-reinstalled before importing Ultralytics in Kaggle kernels.
- **Pillow version (local):** `requirements.txt` pins `Pillow==9.5.0` for Ultralytics 8.2.18. `PIL._util.is_directory` removed in Pillow 10.0.
- **Monodepth2 import:** `networks/` must be copied from the cloned repo to `ml/weights/networks/`. The parent directory `ml/weights/` is injected into `sys.path` at runtime — no installation required.
- **GPS coordinates (Run 3):** All latitude/longitude values in `detections_summary.json` are mock San Francisco placeholder data. Do not use for spatial analysis.
- **TTA no-op:** `augment=True` in Ultralytics 8.2.x `model.val()` for RT-DETR produces identical results to standard inference.

---

## Papers

All 44 papers across 10 categories. Papers marked [EVALUATED, NOT DEPLOYED] were considered in design but are outside the operational pipeline scope.

### 0. Foundational / Baseline Papers

| Paper | Authors | Year | Link |
|---|---|---|---|
| ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky, Sutskever, Hinton | 2012 | [acm](https://dl.acm.org/doi/10.1145/3065386) |
| Deep Residual Learning for Image Recognition (ResNet) | He, Zhang, Ren, Sun | 2016 | [arxiv](https://arxiv.org/abs/1512.03385) |
| Attention Is All You Need (Transformer) | Vaswani et al. | 2017 | [arxiv](https://arxiv.org/abs/1706.03762) |
| Adam: A Method for Stochastic Optimization | Kingma, Ba | 2015 | [arxiv](https://arxiv.org/abs/1412.6980) |
| Decoupled Weight Decay Regularization (AdamW) | Loshchilov, Hutter | 2019 | [arxiv](https://arxiv.org/abs/1711.05101) |
| Batch Normalization | Ioffe, Szegedy | 2015 | [arxiv](https://arxiv.org/abs/1502.03167) |
| Dropout | Srivastava et al. | 2014 | [jmlr](https://jmlr.org/papers/v15/srivastava14a.html) |
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
| Computer Vision for Road Imaging and Pothole Detection | Fan et al. | 2022 | [doi](https://doi.org/10.1093/tse/tdac026) |
| Road Damage Detection Using Deep Neural Networks | Maeda et al. | 2018 | [doi](https://doi.org/10.1111/mice.12387) |
| Robust Video-Based Pothole Detection and Area Estimation | Bui et al. | 2021 | [doi](https://doi.org/10.1109/ACCESS.2021.3088384) |
| When SAM Meets Inventorying of Roadway Assets | Zhang, Huang, Qin | 2024 | [doi](https://doi.org/10.1016/j.ijtst.2024.10.005) |
| N-RDD2024: Road Damage and Defects | Kaya, Çodur | 2024 | [doi](https://doi.org/10.17632/27c8pwsd6v.3) |
| TRIB crack dataset: automatic recognition for road cracks detection | Abrudan | 2025 | [doi](https://doi.org/10.1007/s12145-025-01763-7) |

### 3. Segmentation

| Paper | Authors | Year | Link |
|---|---|---|---|
| Segment Anything (SAM) | Kirillov et al. — Meta AI | 2023 | [arxiv](https://arxiv.org/abs/2304.02643) |
| SAM 2: Segment Anything in Images and Videos | Ravi et al. — Meta AI | 2024 | [arxiv](https://arxiv.org/abs/2408.00714) |
| Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles | Bolya et al. | 2023 | [arxiv](https://arxiv.org/abs/2306.00989) |

### 4. Depth Estimation

| Paper | Authors | Year | Link |
|---|---|---|---|
| Digging into Self-Supervised Monocular Depth Estimation **(Monodepth2)** | Godard et al. | 2019 | [arxiv](https://arxiv.org/abs/1806.01260) |
| EfficientNet: Rethinking Model Scaling for CNNs [EVALUATED, NOT DEPLOYED] | Tan, Le | 2019 | [arxiv](https://arxiv.org/abs/1905.11946) |
| Low-Cost Monocular Vision Techniques for Pothole Distance Estimation | Hach, Sankowski | 2015 | [researchgate](https://www.researchgate.net/publication/285578304) |

### 5. Severity Classification & ML

| Paper | Authors | Year | Link |
|---|---|---|---|
| XGBoost: A Scalable Tree Boosting System [EVALUATED, NOT DEPLOYED] | Chen, Guestrin | 2016 | [arxiv](https://arxiv.org/abs/1603.02754) |
| Modified XGBoost Hyper-Parameter Tuning Using Adaptive PSO | Langat, Waititu, Ngare | 2024 | [doi](https://doi.org/10.11648/j.mlr.20240902.15) |

### 6. Hyperparameter Optimization

| Paper | Authors | Year | Link |
|---|---|---|---|
| Particle Swarm Optimization | Kennedy, Eberhart | 1995 | [doi](https://doi.org/10.1109/ICNN.1995.488968) |
| PSO for Hyper-Parameter Selection in Deep Neural Networks | Young et al. | 2015 | [doi](https://doi.org/10.1145/3071178.3071208) |
| Optuna: A Next-generation Hyperparameter Optimization Framework [EVALUATED, NOT DEPLOYED] | Akiba et al. | 2019 | [arxiv](https://arxiv.org/abs/1907.10902) |
| The Whale Optimization Algorithm [EVALUATED, NOT DEPLOYED] | Mirjalili, Lewis | 2016 | [doi](https://doi.org/10.1016/j.advengsoft.2016.01.008) |

### 7. Training Techniques

| Paper | Authors | Year | Link |
|---|---|---|---|
| Averaging Weights Leads to Wider Optima (SWA) | Izmailov et al. | 2018 | [arxiv](https://arxiv.org/abs/1803.05407) |
| Albumentations: Fast and Flexible Image Augmentations | Buslaev et al. | 2020 | [doi](https://doi.org/10.3390/info11020125) |
| mixup: Beyond Empirical Risk Minimization | Zhang et al. | 2018 | [arxiv](https://arxiv.org/abs/1710.09412) |
| Rethinking the Inception Architecture (label smoothing) | Szegedy et al. | 2016 | [doi](https://doi.org/10.1109/CVPR.2016.308) |

### 8. Clustering & Spatial

| Paper | Authors | Year | Link |
|---|---|---|---|
| A Density-Based Algorithm for Discovering Clusters (DBSCAN) | Ester, Kriegel et al. | 1996 | [acm](https://dl.acm.org/doi/10.5555/3001460.3001507) |

### 9. Route Optimization

| Paper | Authors | Year | Link |
|---|---|---|---|
| Ant Colony Optimization [DEFERRED] | Dorigo, Maniezzo, Colorni | 1996 | [doi](https://doi.org/10.1109/3477.484436) |

> **Total: 44 papers across 10 categories.** Full citations in `references.bib`.
> Added since last version: Hiera backbone paper (SAM 2.1 architecture dependency).

---

## Technology Stack

| Layer | Technology | Version / Notes | Reference |
|---|---|---|---|
| Detection | RT-DETR-L | Ultralytics 8.3.143 | [Zhao et al., 2024](https://arxiv.org/abs/2304.08069) |
| Segmentation | SAM 2.1 Tiny | `sam2.1_hiera_tiny.pt`, zero-shot | [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) |
| Depth estimation | Monodepth2 | `mono_640x192`, KITTI pretrained | [Godard et al., 2019](https://arxiv.org/abs/1806.01260) |
| Severity classification | Rule-based | S1–S5, normalised depth + SAM area | — |
| Hyperparameter optimization | PSO (custom) | 7-dim, 10 particles × 4 iters | [Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968) |
| Survey route planning | ACO · osmnx | Deferred | [Dorigo et al., 1996](https://doi.org/10.1109/3477.484436) |
| Spatial clustering | DBSCAN | scikit-learn, 2m radius | [Ester et al., 1996](https://dl.acm.org/doi/10.5555/3001460.3001507) |
| Augmentation | Albumentations · Mixup · Mosaic | — | [Buslaev et al., 2020](https://doi.org/10.3390/info11020125) |
| Training | Focal Loss · SWA · fp16 AMP | — | [Lin et al., 2017](https://arxiv.org/abs/1708.02002) |
| Database | PostgreSQL 15 + PostGIS | Docker | [postgis.net](https://postgis.net) |
| ORM | SQLAlchemy 2.0 (async) | — | [sqlalchemy.org](https://www.sqlalchemy.org) |
| Backend | FastAPI + Pydantic v2 | 0.111 | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Scheduler | APScheduler | Europe/Bucharest TZ | [apscheduler.readthedocs.io](https://apscheduler.readthedocs.io) |
| Frontend | React 18 + Leaflet.js | In development | — |
| Containerization | Docker Compose | — | [docker.com](https://docker.com) |
| Geocoding | Nominatim | OpenStreetMap | [nominatim.org](https://nominatim.org) |
| Road network | OSM Overpass API | — | [overpass-api.de](https://overpass-api.de) |
| Weather | Open-Meteo API | — | [open-meteo.com](https://open-meteo.com) |
| Sun angle | pysolar | — | [pysolar.readthedocs.io](https://pysolar.readthedocs.io) |
| Language | Python 3.12 | — | [python.org](https://python.org) |

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 ([Arya et al., 2024](https://doi.org/10.1002/gdj3.260)), Pothole600, N-RDD2024 ([Kaya & Çodur, 2024](https://doi.org/10.17632/27c8pwsd6v.3)).
Model attributions: RT-DETR ([Zhao et al., 2024](https://arxiv.org/abs/2304.08069)), SAM 2.1 ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)), Monodepth2 ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Mathematics and Computer Science · 2026</i>
</div>