"""
scripts/detect_and_sam.py
--------------------------
Validation script: runs RT-DETR detection followed by SAM 2.1 segmentation
on all frames in the Cluj manifest.

Pipeline per frame:
  1. Run RT-DETR-L (N-RDD2024) with per-class confidence thresholds
  2. For every accepted bounding box:
     a. Save the frame image with the bounding box drawn on it
        -> data/validation_nrdd_2024/bounding_boxes/cluj/
     b. Save the bounding box coordinates to JSON
        -> data/validation_nrdd_2024/bounding_boxes/bounding_boxes_annotations/
     c. Run SAM 2.1 tiny with the bounding box as prompt
     d. Save the SAM mask overlay image
        -> data/validation_nrdd_2024/sam_masks/cluj/
     e. Compute and save the four geometry features

Outputs
-------
bounding_boxes/cluj/
    frame_000007_t3.500_box0_pothole.jpg      -- frame with bbox drawn

bounding_boxes/bounding_boxes_annotations/
    frame_000007_t3.500.json                  -- all boxes for that frame

sam_masks/cluj/
    frame_000007_t3.500_box0_pothole_mask.jpg -- SAM mask overlay

detections_summary.json                       -- all detections with geometry features

Usage
-----
    python scripts/detect_and_sam.py
    python scripts/detect_and_sam.py --limit 50   # process first 50 frames only (test)
    python scripts/detect_and_sam.py --damage_only # skip marking/infra classes for SAM

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("detect_and_sam")

# ---------------------------------------------------------------------------
# Absolute project root — everything relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH  = PROJECT_ROOT / "data" / "processed" / "frames" / "cluj" / "manifest.json"
RTDETR_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "nrdd_2024" / "best.pt"
SAM_WEIGHTS    = PROJECT_ROOT / "ml" / "weights" / "sam2.1_hiera_tiny.pt"
SAM_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"

OUT_BBOX_IMAGES = Path(r"C:\Facultate\pothole-detection\Pothole-Detection\data\validation_nrdd_2024\bounding_boxes\cluj")
OUT_BBOX_JSON   = Path(r"C:\Facultate\pothole-detection\Pothole-Detection\data\validation_nrdd_2024\bounding_boxes\bounding_boxes_annotations")
OUT_SAM_MASKS   = Path(r"C:\Facultate\pothole-detection\Pothole-Detection\data\validation_nrdd_2024\sam_masks\cluj")
OUT_SUMMARY     = Path(r"C:\Facultate\pothole-detection\Pothole-Detection\data\validation_nrdd_2024\detections_summary.json")

# ---------------------------------------------------------------------------
# N-RDD2024 class schema — must match training order
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "longitudinal_crack",        # 0  D00
    "transverse_crack",          # 1  D10
    "alligator_crack",           # 2  D20
    "repaired_crack",            # 3  D30
    "pothole",                   # 4  D40
    "pedestrian_crossing_blur",  # 5  D50
    "lane_line_blur",            # 6  D60
    "manhole_cover",             # 7  D70
    "patchy_road",               # 8  D80
    "rutting",                   # 9  D90
]

CLASS_CONF_THRESHOLDS: dict[str, float] = {
    "longitudinal_crack":        0.35,
    "transverse_crack":          0.35,
    "alligator_crack":           0.35,
    "repaired_crack":            0.35,
    "pothole":                   0.35,
    "pedestrian_crossing_blur":  0.35,
    "lane_line_blur":            0.50,
    "manhole_cover":             0.35,
    "patchy_road":               0.35,
    "rutting":                   0.35,
}

DAMAGE_CLASSES: set[str] = {
    "longitudinal_crack", "transverse_crack", "alligator_crack",
    "repaired_crack", "pothole", "patchy_road", "rutting",
}
INFRASTRUCTURE_CLASSES: set[str] = {"manhole_cover"}
MARKING_CLASSES: set[str] = {"lane_line_blur", "pedestrian_crossing_blur"}

# Colours per class for bounding box drawing (BGR)
CLASS_COLOURS: dict[str, tuple] = {
    "longitudinal_crack":        (0,   0,   255),   # red
    "transverse_crack":          (0,   165, 255),   # orange
    "alligator_crack":           (0,   255, 255),   # yellow
    "repaired_crack":            (255, 0,   255),   # magenta
    "pothole":                   (255, 0,   0),     # blue
    "pedestrian_crossing_blur":  (0,   165, 255),   # orange
    "lane_line_blur":            (0,   165, 255),   # orange
    "manhole_cover":             (128, 128, 128),   # grey
    "patchy_road":               (0,   255, 0),     # green
    "rutting":                   (255, 255, 0),     # cyan
}


# ---------------------------------------------------------------------------
# Geometry features from SAM mask
# ---------------------------------------------------------------------------

def compute_geometry_features(mask: np.ndarray, image: np.ndarray) -> dict:
    """
    Compute four geometry features from a binary SAM mask.

    Parameters
    ----------
    mask  : binary uint8 array (255 = damage, 0 = background), same HxW as image
    image : BGR image array

    Returns
    -------
    dict with keys: surface_area_px, edge_sharpness, interior_contrast, mask_compactness
    """
    binary = (mask > 127).astype(np.uint8)

    # 1. Surface area in pixels
    surface_area_px = int(np.sum(binary))

    # 2. Edge sharpness — Sobel magnitude along mask boundary
    gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag  = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Dilate mask to get boundary region
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated    = cv2.dilate(binary, kernel, iterations=1)
    boundary   = dilated - binary
    if boundary.sum() > 0:
        edge_sharpness = float(np.mean(sobel_mag[boundary > 0]))
    else:
        edge_sharpness = 0.0

    # 3. Interior contrast — mean inside mask vs mean of border strip outside
    eroded = cv2.erode(binary, kernel, iterations=2)
    outside_strip = binary - eroded   # thin ring inside the mask boundary
    if binary.sum() > 0 and outside_strip.sum() > 0:
        mean_inside  = float(np.mean(gray[binary > 0]))
        mean_outside = float(np.mean(gray[outside_strip > 0]))
        interior_contrast = abs(mean_inside - mean_outside)
    else:
        interior_contrast = 0.0

    # 4. Mask compactness — 4π × area / perimeter²
    # Circle = 1.0, thin crack ≈ 0.05
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if perimeter > 0:
            mask_compactness = float(4 * np.pi * surface_area_px / (perimeter ** 2))
        else:
            mask_compactness = 0.0
    else:
        mask_compactness = 0.0

    return {
        "surface_area_px":   surface_area_px,
        "edge_sharpness":    round(edge_sharpness, 4),
        "interior_contrast": round(interior_contrast, 4),
        "mask_compactness":  round(mask_compactness, 4),
    }


# ---------------------------------------------------------------------------
# SAM mask overlay helper
# ---------------------------------------------------------------------------

def overlay_mask(image: np.ndarray, mask: np.ndarray, colour: tuple) -> np.ndarray:
    """Blend a binary SAM mask onto the image with 40% opacity."""
    overlay = image.copy()
    overlay[mask > 127] = (
        np.array(colour, dtype=np.uint8) * 0.6
        + overlay[mask > 127] * 0.4
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int | None = None, damage_only: bool = False) -> None:

    # Create output directories
    OUT_BBOX_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_BBOX_JSON.mkdir(parents=True, exist_ok=True)
    OUT_SAM_MASKS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    logger.info("Loading manifest: %s", MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if limit:
        manifest = manifest[:limit]
        logger.info("Limit applied: processing first %d frames", limit)

    logger.info("Total frames to process: %d", len(manifest))

    # ------------------------------------------------------------------
    # Device selection -- use CUDA if available, fall back to CPU
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s  |  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ------------------------------------------------------------------
    # Load RT-DETR
    # ------------------------------------------------------------------
    logger.info("Loading RT-DETR from: %s", RTDETR_WEIGHTS)
    from ultralytics import RTDETR
    rtdetr = RTDETR(str(RTDETR_WEIGHTS))
    logger.info("RT-DETR loaded.")

    # ------------------------------------------------------------------
    # Load SAM 2.1 tiny
    # ------------------------------------------------------------------
    logger.info("Loading SAM 2.1 tiny from: %s", SAM_WEIGHTS)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_model     = build_sam2(SAM_CONFIG, str(SAM_WEIGHTS), device=device)
    sam_predictor = SAM2ImagePredictor(sam_model)
    logger.info("SAM 2.1 loaded.")

    # ------------------------------------------------------------------
    # Process frames
    # ------------------------------------------------------------------
    all_detections = []
    total_boxes    = 0
    total_dropped  = 0
    frames_with_hits = 0

    t_start = time.perf_counter()

    for frame_record in manifest:

        # Resolve frame path — manifest uses relative Windows paths
        raw_path   = frame_record["frame_path"]
        frame_path = PROJECT_ROOT / Path(raw_path.replace("\\", "/"))

        if not frame_path.exists():
            logger.warning("Frame not found: %s — skipping", frame_path)
            continue

        frame_stem = frame_path.stem   # e.g. frame_000007_t3.500

        # --------------------------------------------------------------
        # RT-DETR inference
        # --------------------------------------------------------------
        preds = rtdetr.predict(
            source=str(frame_path),
            conf=0.001,        # wide net — per-class filter below
            iou=0.6,
            imgsz=640,
            device=device,
            augment=False,
            verbose=False,
            save=False,
        )

        pred         = preds[0]
        img_h, img_w = pred.orig_shape

        # Read image for drawing and SAM
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            logger.warning("Could not read image: %s", frame_path)
            continue

        # Collect accepted boxes
        accepted_boxes = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for box in pred.boxes:
                coords    = box.xyxy[0].tolist()
                cls_id    = int(box.cls[0].item())
                conf      = float(box.conf[0].item())

                if cls_id >= len(CLASS_NAMES):
                    continue

                cls_name  = CLASS_NAMES[cls_id]
                threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.35)

                if conf < threshold:
                    total_dropped += 1
                    continue

                # Optionally skip non-damage classes for SAM
                if damage_only and cls_name not in DAMAGE_CLASSES:
                    continue

                accepted_boxes.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "x1": round(coords[0], 2),
                    "y1": round(coords[1], 2),
                    "x2": round(coords[2], 2),
                    "y2": round(coords[3], 2),
                    "threshold_applied": threshold,
                    "is_damage":         cls_name in DAMAGE_CLASSES,
                    "is_infrastructure": cls_name in INFRASTRUCTURE_CLASSES,
                    "is_marking":        cls_name in MARKING_CLASSES,
                })

        if not accepted_boxes:
            continue

        frames_with_hits += 1
        total_boxes      += len(accepted_boxes)

        # --------------------------------------------------------------
        # Draw bounding boxes and save annotated image
        # --------------------------------------------------------------
        image_with_boxes = image_bgr.copy()
        for i, box in enumerate(accepted_boxes):
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            colour          = CLASS_COLOURS.get(box["class_name"], (0, 255, 0))
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), colour, 2)
            label = f"{box['class_name']} {box['confidence']:.2f}"
            cv2.putText(
                image_with_boxes, label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )

        bbox_img_path = OUT_BBOX_IMAGES / f"{frame_stem}.jpg"
        cv2.imwrite(str(bbox_img_path), image_with_boxes)

        # --------------------------------------------------------------
        # Save bounding box coordinates JSON
        # --------------------------------------------------------------
        bbox_json = {
            "frame_path":    str(frame_path),
            "frame_stem":    frame_stem,
            "frame_index":   frame_record.get("frame_index"),
            "timestamp_s":   frame_record.get("timestamp_s"),
            "latitude":      frame_record.get("latitude"),
            "longitude":     frame_record.get("longitude"),
            "lighting":      frame_record.get("lighting"),
            "sun_elevation": frame_record.get("sun_elevation"),
            "image_width":   img_w,
            "image_height":  img_h,
            "boxes":         accepted_boxes,
        }
        bbox_json_path = OUT_BBOX_JSON / f"{frame_stem}.json"
        with open(bbox_json_path, "w", encoding="utf-8") as f:
            json.dump(bbox_json, f, indent=2, ensure_ascii=False)

        # --------------------------------------------------------------
        # SAM 2.1 — run once per image, prompt with all boxes at once
        # --------------------------------------------------------------
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_rgb)

        # Stack all bounding boxes as SAM prompts [x1, y1, x2, y2]
        sam_boxes = np.array([
            [b["x1"], b["y1"], b["x2"], b["y2"]] for b in accepted_boxes
        ], dtype=np.float32)

        try:
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=sam_boxes,
                multimask_output=False,
            )
            # masks shape: (N_boxes, 1, H, W) or (N_boxes, H, W)
            # Normalise to (N_boxes, H, W)
            if masks.ndim == 4:
                masks = masks[:, 0, :, :]
        except Exception:
            logger.exception("SAM failed on frame %s", frame_stem)
            # Still save the detections without geometry features
            for box in accepted_boxes:
                box["geometry"] = None
            all_detections.append(bbox_json)
            continue

        # --------------------------------------------------------------
        # Per-box: compute geometry features + save mask overlay
        # --------------------------------------------------------------
        for i, (box, mask, sam_score) in enumerate(
            zip(accepted_boxes, masks, scores)
        ):
            # mask is bool or float — convert to uint8 binary
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

            # Geometry features
            features = compute_geometry_features(mask_uint8, image_bgr)
            box["sam_score"]  = round(float(sam_score), 4)
            box["geometry"]   = features

            # Save SAM mask overlay
            colour      = CLASS_COLOURS.get(box["class_name"], (0, 255, 0))
            mask_overlay = overlay_mask(image_bgr, mask_uint8, colour)

            # Draw the bbox on the overlay too
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), colour, 2)
            label = f"{box['class_name']} {box['confidence']:.2f}"
            cv2.putText(
                mask_overlay, label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )

            mask_save_name = f"{frame_stem}_box{i}_{box['class_name']}_mask.jpg"
            cv2.imwrite(str(OUT_SAM_MASKS / mask_save_name), mask_overlay)

        # Update JSON with geometry features and SAM scores
        bbox_json["boxes"] = accepted_boxes
        with open(bbox_json_path, "w", encoding="utf-8") as f:
            json.dump(bbox_json, f, indent=2, ensure_ascii=False)

        all_detections.append(bbox_json)

        logger.info(
            "Frame %-40s | %d box(es) | %s",
            frame_stem,
            len(accepted_boxes),
            [f"{b['class_name']}({b['confidence']:.2f})" for b in accepted_boxes],
        )

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start

    summary = {
        "total_frames_in_manifest": len(manifest),
        "frames_with_detections":   frames_with_hits,
        "total_boxes_accepted":     total_boxes,
        "total_boxes_dropped":      total_dropped,
        "elapsed_seconds":          round(elapsed, 1),
        "rtdetr_weights":           str(RTDETR_WEIGHTS),
        "sam_weights":              str(SAM_WEIGHTS),
        "class_thresholds":         CLASS_CONF_THRESHOLDS,
        "detections":               all_detections,
    }

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== Done ===")
    logger.info("  Frames processed       : %d", len(manifest))
    logger.info("  Frames with detections : %d", frames_with_hits)
    logger.info("  Boxes accepted         : %d", total_boxes)
    logger.info("  Boxes dropped          : %d", total_dropped)
    logger.info("  Elapsed                : %.1f s", elapsed)
    logger.info("  Bbox images saved to   : %s", OUT_BBOX_IMAGES)
    logger.info("  Bbox JSON saved to     : %s", OUT_BBOX_JSON)
    logger.info("  SAM masks saved to     : %s", OUT_SAM_MASKS)
    logger.info("  Summary JSON           : %s", OUT_SUMMARY)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RT-DETR + SAM 2.1 validation on Cluj frames"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N frames (useful for testing)",
    )
    parser.add_argument(
        "--damage_only", action="store_true",
        help="Only run SAM on structural damage classes "
             "(skip lane_line_blur, pedestrian_crossing_blur, manhole_cover)",
    )
    args = parser.parse_args()
    main(limit=args.limit, damage_only=args.damage_only)