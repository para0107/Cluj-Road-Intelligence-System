# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based pothole detection system using computer vision and deep learning. It processes video footage and GPS logs to detect and classify road damage in real time. The project is research/thesis-oriented.

## Environment Setup

Python 3.12 with a virtual environment in `.venv/`:

```bash
# Activate virtual environment
source .venv/Scripts/activate   # Windows (bash)

# Install dependencies
pip install -r requirements.txt
```

## Planned Tech Stack

Based on the research documentation collected:

- **Detection models**: RT-DETR (primary), YOLOv8 (comparison baseline)
- **Segmentation**: Segment Anything Model (SAM / SAM 2)
- **Image augmentation**: Albumentations
- **Hyperparameter optimization**: Optuna or Particle Swarm Optimization (PSO)
- **Deep learning framework**: PyTorch
- **Depth estimation**: Monocular self-supervised methods
- **Clustering**: DBSCAN (for GPS/spatial aggregation)

## Data Layout

```
data/
  raw/
    footage/       # Input video files
    gps_logs/      # GPS telemetry
  processed/
    frames/        # Frames extracted from video
    metadata/      # Per-frame annotation metadata
  datasets/        # Prepared training/evaluation datasets
```

## Code Style

- Formatter: **Black** (configured in IDE). Run `black .` before committing.
