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

Cluj Road Intelligence System  is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets through a 9-stage machine learning pipeline, enriches each detection with spatial, depth, severity, lighting, weather, and infrastructure context features, and exposes the results through a REST API backed by a PostGIS spatial database.

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
│  React 18 + Leaflet.js (in development)                         │
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
| Training hardware | Kaggle P100 (16 GB VRAM) · Evaluation on T4 |

**Training dataset:**

| Dataset | Images | Annotations | Countries | Reference |
|---|---|---|---|---|
| [RDD2022](https://doi.org/10.1002/gdj3.260) | 21,479 | 33,913 | India, Japan, Norway, China, Czech Republic, USA | Arya et al., 2024 |
| Pothole600 | 5,857 | 8,970 | Mixed | — |
| **Total** | **27,336** | **42,883** | 6 | |

**Two training runs were conducted:**

1. **Baseline run** (default hyperparameters): 10 frozen + 56 fine-tune epochs across 4 Kaggle sessions. Final mAP50 = 0.272.
2. **PSO-optimised run** (PSO best hyperparameters): 10 frozen + 50 fine-tune epochs. Final mAP50 = 0.468 (+72% over baseline).

**Training strategy:**
- **Phase 1** (10 epochs): Frozen backbone — only decoder and detection head trained
- **Phase 2** (50–56 epochs): Full fine-tune — backbone unfrozen, LR × 0.1. Early stopping patience = 20
- **SWA** ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)): planned but not applied in the final training run — `swa.pt` was not produced

Checkpoint priority at inference: `best.pt` > `last.pt` > `rtdetr_l_rdd2022.pt`

> **Selected checkpoint:** `best.pt` — confirmed winner by formal evaluation (see [Final Evaluation Results](#final-evaluation-results)).

**Training techniques:**
- Focal Loss γ=2.0 built into RT-DETR for class imbalance ([Lin et al., 2017](https://arxiv.org/abs/1708.02002))
- Mixup α=0.205 ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)) and Mosaic augmentation (p=0.860)
- Albumentations pipeline ([Buslaev et al., 2020](https://doi.org/10.3390/info11020125)): HSV jitter, rotation, scale, shear, horizontal flip
- Test Time Augmentation evaluated but provided **zero mAP gain** on this dataset (val TTA = val standard, both mAP50=0.465) — TTA is **disabled** in the inference pipeline
- fp16 AMP, gradient accumulation ×4 (effective batch = 16)
- `cudnn.benchmark = True`, `deterministic = False`

**Hyperparameter optimization — PSO** ([Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968)):

PSO searched a 7-dimensional space with 10 particles × 4 iterations × 2 eval epochs on a Kaggle P100 GPU (~9.3h). Methodology follows [Young et al. (2015)](https://doi.org/10.1145/3071178.3071208).

| Parameter | Default | PSO Best | Δ |
|---|---|---|---|
| `lr0` | 1.0e-04 | 4.47e-04 | +347% |
| `weight_decay` | 5.0e-04 | 5.27e-04 | +5.3% |
| `warmup_epochs` | 3 | 1 | −67% |
| `mosaic` | 1.0 | 0.860 | −14% |
| `mixup` | 0.15 | 0.205 | +37% |
| `box` | 7.5 | 7.685 | +2.5% |
| `cls` | 0.5 | 0.487 | −2.6% |

The most significant finding is the 4.5× increase in `lr0`, which allows the decoder to adapt faster during the frozen-backbone phase. Output saved to `ml/optimization/pso_best.json` — `train.py` loads this automatically.

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

- **[EfficientNet-B3](https://arxiv.org/abs/1905.11946)** ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)): regression head on cropped detection region + sun angle → depth in cm. To be trained on Cluj ground truth measurements + Blender synthetic renders. Tuned with [Optuna](https://arxiv.org/abs/1907.10902) (TPE sampler).
- **[Monodepth2](https://arxiv.org/abs/1806.01260)** ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)): self-supervised dense depth map — depth at detection region extracted and fused with EfficientNet estimate.
- **Fallback**: proxy depth from mask geometry when `depth_confidence < 0.4` or `lighting_condition = low_light`.

### Severity Classification — XGBoost + WOA

**Whale Optimization Algorithm** ([Mirjalili & Lewis, 2016](https://doi.org/10.1016/j.advengsoft.2016.01.008)) performs binary feature selection across the 16-feature ML vector before [XGBoost](https://arxiv.org/abs/1603.02754) ([Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)) training.

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

### Baseline Run — Default Hyperparameters

#### Phase 1 — Frozen Backbone (Epochs 1–10)

| Epoch | Train GIoU ↓ | Train L1 ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|
| 1  | 1.418 | 1.033 | 0.201 | 0.00248 | 0.000631 |
| 5  | 1.028 | 0.609 | 0.447 | 0.02045 | 0.006900 |
| 10 | 0.942 | 0.546 | 0.533 | 0.02658 | 0.009920 |

#### Phase 2 — Full Fine-Tune (Epochs 11–56, 3 Kaggle sessions)

| Epoch | Run | Train GIoU ↓ | Val GIoU ↓ | Precision ↑ | Recall ↑ | mAP50 ↑ |
|---|---|---|---|---|---|---|
| 11 | R1 | 1.376 | 1.204 | 0.271 | 0.084 | 0.00761 |
| 19 | R1 | 0.721 | 0.763 | 0.169 | 0.255 | 0.11567 |
| 38 | R2 | 0.601 | 0.643 | 0.260 | 0.348 | 0.21071 |
| 56 | R3 | 0.568 | 0.610 | 0.315 | 0.377 | 0.27295 |

**Final validation results (best.pt, 5,857 images, conf=0.001, iou=0.6):**

| Metric | Value |
|---|---|
| mAP50 | 0.272 |
| mAP50-95 | 0.127 |
| Precision | 0.313 |
| Recall | 0.376 |
| F1 peak | 0.33 at conf=0.311 |

**Per-class AP@0.50:**

| Class | Instances | AP@0.50 |
|---|---|---|
| pothole | 1,811 | 0.379 |
| longitudinal_crack | 3,890 | 0.265 |
| alligator_crack | 1,553 | 0.226 |
| transverse_crack | 1,769 | 0.218 |
| patch_deterioration | — | 0.000 |

---

### PSO-Optimised Run — Best Hyperparameters

#### Phase 1 — Frozen Backbone (Epochs 1–10)

| Epoch | Train GIoU ↓ | Train L1 ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|
| 1  | 0.942 | 0.491 | 0.217 | 0.0497 | 0.0189 |
| 5  | 0.731 | 0.361 | 0.220 | 0.0660 | 0.0277 |
| 10 | 0.695 | 0.335 | 0.238 | 0.0917 | 0.0408 |

Phase 1 comparison vs baseline: PSO achieves **26% lower GIoU** and **3.4× higher mAP50** in the same 10 frozen epochs, due to the 4.5× higher learning rate found by PSO.

#### Phase 2 — Full Fine-Tune (Epochs 11–50, 3 Kaggle sessions)

| Epoch | Session | Train GIoU ↓ | Val GIoU ↓ | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|---|---|
| 12 | S1 | 0.858 | 0.686 | 0.146 | 0.267 | 0.100 | 0.047 |
| 22 | S1 | 0.712 | 0.589 | 0.269 | 0.346 | 0.230 | 0.114 |
| 31 | S2 | 0.692 | 0.601 | 0.530 | 0.447 | 0.443 | 0.207 |
| 40 | S3 | 0.678 | 0.614 | 0.547 | 0.459 | 0.463 | 0.214 |
| 50 | S3 | 0.537 | 0.615 | 0.541 | 0.466 | 0.468 | 0.217 |

The GIoU drop at epoch 41 is the Ultralytics mosaic cutoff — structural, not a bug.

**Final validation results at epoch 50:**

| Metric | Baseline | PSO Retrain | Δ |
|---|---|---|---|
| mAP50 | 0.272 | **0.468** | **+72%** |
| mAP50-95 | 0.127 | **0.217** | **+71%** |
| Precision | 0.313 | **0.541** | **+73%** |
| Recall | 0.376 | **0.466** | **+24%** |

**Key findings:**
- mAP50 improvement is primarily precision-driven — the higher PSO learning rate produces more selective, better-localised predictions
- Recall improvement (+24%) confirms overall detection quality improved, not just precision
- Domain gap (international training data vs Romanian roads) remains the dominant limitation — background false-negative rates from confusion matrix: longitudinal 45.5%, transverse 52.4%, alligator 43.5%, pothole 29.3%
- Cross-class confusion is very low (≤4%) — the model distinguishes crack types reliably; the primary failure is damage vs clean road surface, not class-to-class mistakes
- Fix: Cluj-Napoca data collection + fine-tuning (planned)
- Recommended operational confidence threshold: `conf=0.35` (balances precision and recall for municipal survey use)

---

## Final Evaluation Results

Formal evaluation of the PSO-optimised RT-DETR-L checkpoint (`best.pt`) conducted on 21 April 2026 using `ml/detection/evaluate.py` on a Kaggle Tesla T4 GPU. All results use `conf=0.001` (standard for mAP computation per the COCO protocol, Lin et al., 2014) and `iou=0.6`. Per-class AP values follow the PASCAL VOC AP protocol (Everingham et al., 2010).

Evaluation outputs are stored in `ml/evaluation/` — JSON files for each run, log files for the full execution trace.

### Checkpoint Selection — best.pt vs last.pt

Both checkpoints were evaluated on the val set (5,758 images) to select the operational checkpoint for the inference pipeline.

| Checkpoint | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `best.pt` | **0.4650** | **0.2151** | **0.5450** | 0.4639 | **0.5012** |
| `last.pt` | 0.4651 | 0.2148 | 0.5413 | **0.4655** | 0.5005 |

**Decision: `best.pt` selected.** The margin is negligible (0.0003 mAP50-95) but `best.pt` has marginally higher mAP50-95 and precision, making it the tiebreaker. `best.pt` is the checkpoint Ultralytics saves at the epoch with the highest validation mAP50 during training — in this run, epoch 40 of Phase 2.

### Val Set — Final Numbers (best.pt)

| Metric | Value |
|---|---|
| mAP50 | **0.4650** |
| mAP50-95 | **0.2151** |
| Precision | **0.5450** |
| Recall | **0.4639** |
| F1 | **0.5012** |

### Test Set — Held-Out Generalisation (best.pt)

The test set (5,758 images, completely held out during training and hyperparameter search) gives the honest generalisation estimate reported in the thesis.

| Metric | Val | Test | Val→Test drop |
|---|---|---|---|
| mAP50 | 0.4650 | **0.4583** | −0.007 |
| mAP50-95 | 0.2151 | **0.2110** | −0.004 |
| Precision | 0.5450 | **0.5367** | −0.008 |
| Recall | 0.4639 | **0.4633** | −0.001 |
| F1 | 0.5012 | **0.4973** | −0.004 |

The val→test drop is consistently small across all metrics, confirming the model generalises well within the RDD2022 distribution. There is no evidence of overfitting to the validation set.

### Per-Class AP@50 (best.pt, test set)

Per-class AP is AP at IoU=0.50 as stored in `results.box.ap` by Ultralytics 8.2.x, following the PASCAL VOC AP protocol (Everingham et al., 2010).

| Class | Val AP@50 | Test AP@50 | Instances (val) | Notes |
|---|---|---|---|---|
| `pothole` | 0.3231 | **0.3130** | 1,533 | Best performing — visually distinctive shape and depth |
| `alligator_crack` | 0.2363 | **0.2312** | 781 | Second best — unique interconnected texture pattern |
| `longitudinal_crack` | 0.1739 | **0.1742** | 1,956 | Thin linear features, high domain gap sensitivity |
| `transverse_crack` | 0.1270 | **0.1257** | 868 | Lowest trained class — short perpendicular cracks hardest to localise |
| `patch_deterioration` | 0.0000 | **0.0000** | 0 | Expected — no training samples in RDD2022+Pothole600 |

The AP ranking is consistent between val and test sets and aligns with the visual distinctiveness of each damage type. Potholes are the easiest to detect due to their bowl-shaped 3D structure; transverse cracks are the hardest due to their short length and similarity to road joints at low resolution.

### Baseline vs PSO — Full Comparison (val set)

| Metric | Baseline | PSO best.pt | Δ (absolute) | Δ (relative) |
|---|---|---|---|---|
| mAP50 | 0.272 | **0.465** | +0.193 | **+71%** |
| mAP50-95 | 0.127 | **0.215** | +0.088 | **+69%** |
| Precision | 0.313 | **0.545** | +0.232 | **+74%** |
| Recall | 0.376 | **0.464** | +0.088 | **+23%** |
| F1 | — | **0.501** | — | — |

### TTA Evaluation

Test Time Augmentation (flip + rotate, predictions averaged) was evaluated on the val set.

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Standard | 0.4650 | 0.2151 | 0.5450 | 0.4639 |
| TTA | 0.4650 | 0.2151 | 0.5450 | 0.4639 |
| Δ | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

TTA produced **zero improvement** on this dataset. The `augment=True` flag in Ultralytics 8.2.x for RT-DETR appears to have no effect in this configuration. TTA is **disabled** in the CRIS inference pipeline — it adds ~3× inference time with no accuracy benefit.

### Confusion Matrix Analysis

The confusion matrix from the PSO-optimised run reveals the dominant failure pattern. Cross-class confusion between damage types is very low (≤4% in all cases). The primary failure is **background false negatives** — damage instances the model fails to detect and classifies as clean road surface:

| Class | Correctly detected | Missed as background | Background FN rate |
|---|---|---|---|
| `longitudinal_crack` | 2,093 | 1,747 | **45.5%** |
| `transverse_crack` | 828 | 913 | **52.4%** |
| `alligator_crack` | 830 | 640 | **43.5%** |
| `pothole` | 1,276 | 528 | **29.3%** |

This pattern is consistent with a **domain gap** rather than a model capacity or label quality issue. RDD2022 contains images from Japan, India, China, Norway, Czech Republic, and the USA — road surface textures, crack morphology, and lighting conditions differ substantially from Romanian urban roads. The model has learned to distinguish damage types from each other reliably, but its damage-vs-background boundary is calibrated to non-Romanian road surfaces.

**Implication for CRIS:** Cluj-Napoca fine-tuning data collection is the highest-priority next step for improving recall. Lowering the operational confidence threshold to `conf=0.35` partially compensates by recovering some missed detections at the cost of more false positives, which is the correct trade-off for a municipal survey system where missing damage is more costly than flagging clean road.

### Evaluation Artefacts

All evaluation outputs are committed to `ml/evaluation/`:

| File | Contents |
|---|---|
| `checkpoint_comparison_{ts}.json` | best.pt vs last.pt scores and full metrics |
| `eval_val_{ts}.json` | Val set metrics — mAP50, mAP50-95, P, R, F1, per-class AP50 |
| `eval_val_tta_{ts}.json` | Val set + TTA metrics |
| `eval_test_{ts}.json` | Test set metrics — primary thesis result |
| `eval_full_{ts}.log` | Full execution log with all intermediate output |
| `eval_compare_{ts}.log` | Checkpoint comparison execution log |

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
│   ├── segmentation/               ⬜ SAM inference module
│   ├── depth/                      ⬜ EfficientNet-B3 depth training
│   ├── severity/                   ⬜ XGBoost + WOA feature selection
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     ✅ PSO-optimised, mAP50=0.465/test=0.458 (best.pt)
│       ├── rtdetr_l_cluj.pt        ⬜ Future: fine-tuned on Cluj footage
│       ├── depth_effnet.pt         ⬜ Future: EfficientNet-B3 depth model
│       └── xgboost_severity.json   ⬜ Future: XGBoost classifier
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
│   └── run_survey.py               ✅ Manual one-shot pipeline trigger
│
├── frontend/
│   └── (in development)            🔄 React 18 + Leaflet.js dashboard
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
└── requirements.txt                ✅ Pillow==9.5.0 (Ultralytics 8.2.18 compat)
```

**Legend:** ✅ Done · 🔄 In progress · ⬜ Planned

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
# (Optional) PSO hyperparameter search — ~9.3h on P100
python ml/optimization/pso_hyperparams.py --particles 10 --iterations 4 --eval_epochs 2

# Train RT-DETR-L (auto-loads pso_best.json if present)
python ml/detection/train.py
python ml/detection/train.py --smoke_test         # 2-epoch pipeline validation
python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt

# Multi-GPU (Kaggle T4×2)
python ml/detection/train.py --device 0,1 --workers 4

# Monitor in a second terminal
python ml/detection/monitor.py
python ml/detection/monitor.py --save --interval 60
```

### Evaluation

```bash
# Evaluate best.pt on val set (auto-detected from final_fine_tune/)
python ml/detection/evaluate.py

# Evaluate on test set (held-out — use for thesis numbers)
python ml/detection/evaluate.py --split test

# Compare best.pt vs last.pt
python ml/detection/evaluate.py --compare

# Full report: val + val-TTA + test + checkpoint comparison (~60 min on T4)
python ml/detection/evaluate.py --full

# Merge all results.csv files into one xlsx (run locally after downloading Kaggle outputs)
python ml/detection/evaluate.py --merge-results
```

> **Kaggle note:** `evaluate.py` requires `ultralytics==8.2.18` and `Pillow==9.5.0`. On newer Kaggle environments (PyTorch 2.6+), patch `PIL._util.is_directory` before importing ultralytics — see `evaluate-best-model.ipynb` in the repo root.

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
- [x] RT-DETR-L training pipeline (two-phase + PSO integration)
- [x] PSO hyperparameter search (10 particles × 4 iterations, P100)
- [x] Evaluation script with per-class AP, checkpoint comparison, results merging
- [x] Live training monitor
- [x] Docker Compose — PostgreSQL 15 + PostGIS + pgAdmin
- [x] Database schema — `detections` + `survey_log` tables, enums, GIST index
- [x] SQLAlchemy ORM models + Pydantic v2 schemas
- [x] FastAPI backend — all routes (tested on Swagger)
- [x] APScheduler daily job
- [x] Baseline training — 56 epochs (mAP50=0.272, mAP50-95=0.127)
- [x] PSO-optimised training — 50 epochs (mAP50=0.468 training, **+72% over baseline**)
- [x] Confusion matrix analysis and domain gap diagnosis
- [x] **Formal evaluation completed** — val mAP50=0.465, test mAP50=0.458, test mAP50-95=0.211
- [x] **Checkpoint selected** — `best.pt` confirmed winner over `last.pt` (mAP50-95=0.2151 vs 0.2148)
- [x] **TTA evaluated** — zero gain on this dataset, disabled in inference pipeline

**In progress:**
- [ ] React frontend — map, sidebar, detail panel (React 18 + Leaflet.js)

**Planned:**
- [ ] Inference pipeline — all 8 modules + orchestrator
- [ ] Cluj-Napoca data collection drive (dashcam + GPS + depth measurements)
- [ ] Label Studio annotation of Cluj footage
- [ ] EfficientNet-B3 depth model training on Cluj ground truth
- [ ] XGBoost + WOA severity classifier training
- [ ] RT-DETR fine-tuning on Cluj footage → `rtdetr_l_cluj.pt`
- [ ] Optuna tuning for EfficientNet-B3 + XGBoost
- [ ] ACO survey route generation (`aco_route.py`)
- [ ] End-to-end integration test on real Cluj footage
- [ ] City Hall pilot demonstration

---

## Known Issues & Compatibility Notes

- **Pillow version:** `requirements.txt` pins `Pillow==9.5.0`. Ultralytics 8.2.18 uses `PIL._util.is_directory` which was removed in Pillow 10.0. Do not upgrade Pillow without upgrading Ultralytics first.
- **Kaggle PIL patch (newer environments):** Kaggle environments with PyTorch 2.6+ have a system-level Pillow 12.x that cannot be overridden by pip. Inject the missing symbol at runtime before any ultralytics import: `import PIL._util, os; PIL._util.is_directory = os.path.isdir`. See `evaluate-best-model.ipynb`.
- **Kaggle resume:** When resuming training after a session that reached its `--epochs` target, patch the checkpoint with `ckpt['epochs'] = current_epoch + extra_epochs` before calling `train.py --resume`. See `scripts/patch_checkpoint.py` (planned).
- **numpy/pandas incompatibility:** If `pandas` fails to import after training in the same kernel, reinstall with `pip install --force-reinstall numpy==1.26.4 pandas==2.2.2`.
- **TTA no-op:** `augment=True` in Ultralytics 8.2.x `model.val()` for RT-DETR produces identical results to standard inference. TTA is disabled in the pipeline.

---

## Papers

All 39 papers organized across 10 categories.

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

> **Total: 39 papers across 10 categories.** Full citations in `references.bib`.

---

## Technology Stack

| Layer | Technology | Reference |
|---|---|---|
| Detection | RT-DETR-L (Ultralytics 8.2.18) | [Zhao et al., 2024](https://arxiv.org/abs/2304.08069) |
| Segmentation | SAM — Segment Anything Model | [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) |
| Depth estimation | EfficientNet-B3 + Monodepth2 | [Tan & Le, 2019](https://arxiv.org/abs/1905.11946) · [Godard et al., 2019](https://arxiv.org/abs/1806.01260) |
| Severity classification | XGBoost | [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754) |
| Hyperparameter optimization | PSO (custom) · Optuna (TPE) | [Kennedy & Eberhart, 1995](https://doi.org/10.1109/ICNN.1995.488968) |
| Feature selection | WOA — Whale Optimization Algorithm | [Mirjalili & Lewis, 2016](https://doi.org/10.1016/j.advengsoft.2016.01.008) |
| Survey route planning | ACO · osmnx | [Dorigo et al., 1996](https://doi.org/10.1109/3477.484436) |
| Spatial clustering | DBSCAN (scikit-learn) | [Ester et al., 1996](https://dl.acm.org/doi/10.5555/3001460.3001507) |
| Augmentation | Albumentations · Mixup · Mosaic | [Buslaev et al., 2020](https://doi.org/10.3390/info11020125) |
| Training | Focal Loss · SWA · TTA · fp16 AMP | [Lin et al., 2017](https://arxiv.org/abs/1708.02002) · [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407) |
| Database | PostgreSQL 15 + PostGIS | [postgis.net](https://postgis.net) |
| ORM | SQLAlchemy 2.0 (async) | [sqlalchemy.org](https://www.sqlalchemy.org) |
| Backend | FastAPI + Pydantic v2 | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Scheduler | APScheduler (Europe/Bucharest TZ) | [apscheduler.readthedocs.io](https://apscheduler.readthedocs.io) |
| Frontend | React 18 + Leaflet.js (in development) | — |
| Containerization | Docker Compose | [docker.com](https://docker.com) |
| Geocoding | Nominatim (OpenStreetMap) | [nominatim.org](https://nominatim.org) |
| Road network | OSM Overpass API | [overpass-api.de](https://overpass-api.de) |
| Weather | Open-Meteo API | [open-meteo.com](https://open-meteo.com) |
| Sun angle | pysolar | [pysolar.readthedocs.io](https://pysolar.readthedocs.io) |
| Annotation | Label Studio | [labelstud.io](https://labelstud.io) |
| Language | Python 3.12 | [python.org](https://python.org) |

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 ([Arya et al., 2024](https://doi.org/10.1002/gdj3.260)), Pothole600.
Model attributions: RT-DETR ([Zhao et al., 2024](https://arxiv.org/abs/2304.08069)), SAM ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)), Monodepth2 ([Godard et al., 2019](https://arxiv.org/abs/1806.01260)).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Mathematics and Computer Science · 2026</i>
</div>
