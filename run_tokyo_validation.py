"""
run_tokyo_validation.py
=======================
Runs the CRIS Stage 1 + Stage 2 pipeline on the Tokyo dashcam video.
No GPS / .gpx file required — coordinates will be null.

This is the EXACT same pipeline applied to the Cluj video, with:
  - GPS sync removed (no .gpx)
  - Output directory pointed to data/processed/detections/tokyo/
  - Same model weights, same confidence threshold (0.35), same frame rate (1/0.5s)

Usage:
    python run_tokyo_validation.py

Outputs:
    data/processed/detections/tokyo/detections.json   -- all frame detections
    data/processed/frames/tokyo/                      -- extracted frames (JPEGs)

Author: CRIS pipeline — Paraschiv Tudor, BBU 2026
Reference for RT-DETR: Zhao et al. (2024) https://arxiv.org/abs/2304.08069
"""

import cv2
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from ultralytics import RTDETR

# ---------------------------------------------------------------------------
# Logging — always log, never print only
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tokyo_validation.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — actual values, no hardcoded mock data
# ---------------------------------------------------------------------------
VIDEO_PATH = Path("data/raw/footage/Tokyo Drive 4K ｜ Takanawa - Azabudai Hills - Shinjuku [v7JZ9DSSRsY].mp4")

# Model weights — same priority as Cluj pipeline: swa.pt > best.pt > rtdetr_l_rdd2022.pt
WEIGHT_CANDIDATES = [
    Path("ml/weights/swa.pt"),
    Path("ml/weights/best.pt"),
    Path("ml/weights/rtdetr_l_rdd2022.pt"),
    Path("ml/weights/rtdetr-l.pt"),  # Kaggle renamed version
]

FRAME_INTERVAL_S   = 0.5        # 1 frame every 0.5 seconds — same as Cluj
CONFIDENCE_THRESH  = 0.35       # same operational threshold as Cluj
IOU_THRESH         = 0.45       # NMS IoU threshold
IMG_SIZE           = 640        # RT-DETR input resolution

OUTPUT_FRAMES_DIR  = Path("data/processed/frames/tokyo")
OUTPUT_DETECTIONS  = Path("data/processed/detections/tokyo/detections.json")

# Class names — must match training order exactly
CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]

# Lighting classification thresholds (same as Cluj preprocessor)
# Based on mean luminance of the frame
DAYLIGHT_LUMA_THRESH  = 100   # > 100 → daylight
OVERCAST_LUMA_THRESH  = 60    # 60–100 → overcast
# < 60 → low_light


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_weights() -> Path:
    """Find the best available model checkpoint."""
    for candidate in WEIGHT_CANDIDATES:
        if candidate.exists():
            logger.info("Using model weights: %s", candidate)
            return candidate
    logger.error(
        "No model weights found. Checked: %s",
        [str(c) for c in WEIGHT_CANDIDATES],
    )
    sys.exit(1)


def classify_lighting(frame_bgr) -> str:
    """
    Classify lighting condition from mean luminance.
    Same logic as Cluj preprocessor — no sun angle available (no GPS/timestamp).
    Returns: 'daylight' | 'overcast' | 'low_light'
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    if mean_luma > DAYLIGHT_LUMA_THRESH:
        return "daylight"
    elif mean_luma > OVERCAST_LUMA_THRESH:
        return "overcast"
    else:
        return "low_light"


def extract_frames(video_path: Path, output_dir: Path, interval_s: float):
    """
    Stage 1 — Frame extraction (no GPS sync, GPS is null for Tokyo video).
    Extracts 1 frame every interval_s seconds.
    Returns list of dicts with frame metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0
    frame_step   = max(1, int(round(fps * interval_s)))

    logger.info("Video: %s", video_path.name)
    logger.info("  FPS: %.2f | Total frames: %d | Duration: %.1fs", fps, total_frames, duration_s)
    logger.info("  Extracting 1 frame every %.1fs (step=%d frames)", interval_s, frame_step)

    frame_metadata = []
    frame_idx      = 0      # video frame counter
    extracted      = 0      # extracted frame counter

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp_s  = frame_idx / fps if fps > 0 else 0
            lighting     = classify_lighting(frame)
            out_path     = output_dir / f"frame_{extracted:06d}.jpg"

            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            frame_metadata.append({
                "frame_index"  : extracted,
                "video_frame"  : frame_idx,
                "timestamp_s"  : round(timestamp_s, 1),
                "lighting"     : lighting,
                "frame_path"   : str(out_path),
                # GPS is null — no .gpx file for this video
                "latitude"     : None,
                "longitude"    : None,
            })
            extracted += 1

            if extracted % 200 == 0:
                elapsed = time.time() - t_start
                logger.info(
                    "  Extracted %d frames (t=%.1fs) | %.1fs elapsed",
                    extracted, timestamp_s, elapsed,
                )

        frame_idx += 1

    cap.release()
    elapsed_total = time.time() - t_start
    logger.info(
        "Stage 1 complete: %d frames extracted from %d video frames in %.1fs",
        extracted, frame_idx, elapsed_total,
    )
    return frame_metadata


def run_detection(frame_metadata: list, model, output_detections: Path) -> list:
    """
    Stage 2 — RT-DETR inference on each extracted frame.
    Reference: Zhao et al. (2024) https://arxiv.org/abs/2304.08069

    Returns list of frame dicts with detection results appended.
    Writes detections.json incrementally every 500 frames.
    """
    output_detections.parent.mkdir(parents=True, exist_ok=True)

    results_all   = []
    total_boxes   = 0
    frames_with_det = 0
    class_counts  = {c: 0 for c in CLASS_NAMES}

    t_start = time.time()
    logger.info("Stage 2 — RT-DETR inference on %d frames", len(frame_metadata))
    logger.info("  Confidence threshold: %.2f | IOU: %.2f | Image size: %d",
                CONFIDENCE_THRESH, IOU_THRESH, IMG_SIZE)

    for i, meta in enumerate(frame_metadata):
        frame_path = Path(meta["frame_path"])
        if not frame_path.exists():
            logger.warning("Frame not found, skipping: %s", frame_path)
            frame_record = {**meta, "boxes": [], "n_detections": 0}
            results_all.append(frame_record)
            continue

        # RT-DETR inference
        preds = model.predict(
            source=str(frame_path),
            conf=CONFIDENCE_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            verbose=False,
            device="cpu",   # change to 0 if CUDA available
        )

        boxes = []
        for pred in preds:
            if pred.boxes is None:
                continue
            for box in pred.boxes:
                cls_idx    = int(box.cls.item())
                cls_name   = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                confidence = float(box.conf.item())
                xyxy       = box.xyxy[0].tolist()   # [x1, y1, x2, y2] in pixels
                xywhn      = box.xywhn[0].tolist()  # [cx, cy, w, h] normalised

                boxes.append({
                    "class_idx"   : cls_idx,
                    "class_name"  : cls_name,
                    "confidence"  : round(confidence, 4),
                    "bbox_xyxy"   : [round(v, 2) for v in xyxy],
                    "bbox_xywhn"  : [round(v, 4) for v in xywhn],
                })
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_boxes += 1

        if boxes:
            frames_with_det += 1

        frame_record = {
            **meta,
            "boxes"         : boxes,
            "n_detections"  : len(boxes),
        }
        results_all.append(frame_record)

        # Progress log every 200 frames
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            fps_inf = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "  [%d/%d] boxes_so_far=%d | %.1f frames/s | %.1fs elapsed",
                i + 1, len(frame_metadata), total_boxes, fps_inf, elapsed,
            )

        # Incremental save every 500 frames — survive interruptions
        if (i + 1) % 500 == 0:
            _save_detections(results_all, output_detections)
            logger.info("  Checkpoint saved → %s (%d frames)", output_detections, len(results_all))

    # Final save
    _save_detections(results_all, output_detections)

    elapsed_total = time.time() - t_start
    logger.info("Stage 2 complete in %.1fs", elapsed_total)
    logger.info("  Frames processed      : %d", len(frame_metadata))
    logger.info("  Frames with detections: %d (%.1f%%)",
                frames_with_det, 100 * frames_with_det / max(1, len(frame_metadata)))
    logger.info("  Total bounding boxes  : %d", total_boxes)
    logger.info("  Class breakdown:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info("    %-30s: %d", cls, count)

    return results_all


def _save_detections(results: list, path: Path):
    """Save detections list to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: list):
    """Print final summary table — same format as Cluj pipeline."""
    total_frames   = len(results)
    frames_with    = sum(1 for r in results if r["n_detections"] > 0)
    total_boxes    = sum(r["n_detections"] for r in results)
    class_counts   = {}
    class_confs    = {}

    for r in results:
        for b in r["boxes"]:
            cn = b["class_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1
            class_confs.setdefault(cn, []).append(b["confidence"])

    all_confs = [b["confidence"] for r in results for b in r["boxes"]]

    print("\n" + "=" * 58)
    print("  CRIS — Tokyo Validation | RT-DETR Detection Summary")
    print("=" * 58)
    print(f"  Video                    : {VIDEO_PATH.name}")
    print(f"  Model                    : RT-DETR-L (Zhao et al., 2024)")
    print(f"  Confidence threshold     : {CONFIDENCE_THRESH}")
    print(f"  Frames processed         : {total_frames}")
    print(f"  Frames with detections   : {frames_with} ({100*frames_with/max(1,total_frames):.1f}%)")
    print(f"  Frames with no detection : {total_frames - frames_with}")
    print(f"  Total bounding boxes     : {total_boxes}")
    print(f"  Class breakdown")
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        if count > 0:
            mean_conf = sum(class_confs[cls]) / len(class_confs[cls])
            print(f"    {cls:<30}: {count:4d}  (mean conf {mean_conf:.2f})")
        else:
            print(f"    {cls:<30}:    0")
    if all_confs:
        print(f"  Confidence (all classes)")
        print(f"    min / mean / max : {min(all_confs):.3f} / "
              f"{sum(all_confs)/len(all_confs):.3f} / {max(all_confs):.3f}")
    print("=" * 58)
    logger.info("Summary printed. Detections saved to %s", OUTPUT_DETECTIONS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 58)
    logger.info("CRIS — Tokyo Dashcam Validation Pipeline")
    logger.info("  RT-DETR-L | Zhao et al. (2024) arxiv:2304.08069")
    logger.info("  No GPS — coordinates will be null")
    logger.info("=" * 58)

    # Validate video exists
    if not VIDEO_PATH.exists():
        logger.error("Video not found: %s", VIDEO_PATH)
        logger.error("Expected at: %s", VIDEO_PATH.resolve())
        sys.exit(1)
    logger.info("Video found: %s (%.1f MB)",
                VIDEO_PATH.name, VIDEO_PATH.stat().st_size / 1e6)

    # Load model
    weights = find_weights()
    logger.info("Loading RT-DETR-L model...")
    model = RTDETR(str(weights))
    logger.info("Model loaded.")

    # Stage 1 — Frame extraction
    logger.info("-" * 40)
    logger.info("STAGE 1 — Frame extraction (%.1fs interval)", FRAME_INTERVAL_S)
    frame_metadata = extract_frames(VIDEO_PATH, OUTPUT_FRAMES_DIR, FRAME_INTERVAL_S)

    # Stage 2 — Detection
    logger.info("-" * 40)
    logger.info("STAGE 2 — RT-DETR inference")
    results = run_detection(frame_metadata, model, OUTPUT_DETECTIONS)

    # Summary
    print_summary(results)
    logger.info("Pipeline complete. Run inspect_detector_tokyo.py to generate annotated frames.")


if __name__ == "__main__":
    main()