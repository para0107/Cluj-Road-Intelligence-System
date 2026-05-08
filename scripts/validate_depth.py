"""
scripts/validate_depth.py
--------------------------
Validation script for Stage 4 — Monodepth2 depth estimation.

For every frame that has at least one accepted detection in
detections_summary.json this script:

  1. Loads the original frame image.
  2. Runs Monodepth2 (mono_640x192, KITTI pretrained) to produce a dense
     relative depth map.
  3. For each accepted bounding box in that frame:
       - Extracts the mean relative depth over the central 60% of the bbox
         crop (robust to bbox border noise; see rationale in comments).
       - Normalises depth to [0, 1] within the frame
         (per-frame normalisation is correct because Monodepth2 produces
          relative, not metric, depth — values are only comparable within
          the same forward pass, not across frames).
  4. Saves a three-panel visualisation image:
       LEFT  — original frame with RT-DETR bounding boxes drawn
       MIDDLE — Monodepth2 depth map colourised with the 'magma' colormap
                (standard in depth estimation literature)
       RIGHT  — SAM 2.1 mask overlay image loaded from disk
  5. Writes depth_validation.json with all extracted depth values.

Central 60% crop rationale:
  The full bounding box often contains clean road surface at its edges
  (RT-DETR boxes are not pixel-tight). The central 60% crop reduces this
  border contamination. Using the SAM mask region directly would be more
  accurate but requires reconstructing binary masks from JPEG overlays
  (lossy — JPEG compression shifts the class colours). The central crop
  is the most robust approach given the data available at validation time.
  In the full pipeline, Stage 4 (depth_estimator.py) receives binary mask
  arrays in memory from Stage 3 and uses the exact mask pixels instead.

Monodepth2 reference:
  Godard et al., 2019. arXiv:1806.01260
  "Digging into Self-Supervised Monocular Depth Estimation"

Model variant: mono_640x192
  Monocular, KITTI pretrained, 640x192 input resolution.
  Lightest checkpoint; sufficient for ordinal severity proxy use.
  VRAM footprint ~0.8 GB — compatible with RTX 2050 (4 GB) alongside
  RT-DETR-L and SAM 2.1 Tiny.

Setup (run once before first use):
  python scripts/validate_depth.py --setup
  This prints the exact clone and download commands.

Usage:
  python scripts/validate_depth.py
  python scripts/validate_depth.py --limit 100
  python scripts/validate_depth.py --device cpu
  python scripts/validate_depth.py --setup

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_depth")

# ---------------------------------------------------------------------------
# Project paths — all absolute, no hardcoded relative assumptions
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")

# networks/ is copied from the Monodepth2 repo into the project weights folder.
# The script imports it from there — the full repo is NOT required at runtime.
NETWORKS_DIR   = PROJECT_ROOT / "ml" / "weights" / "networks"

SUMMARY_JSON   = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "detections_summary.json"
SAM_MASKS_DIR  = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "sam_masks" / "cluj"
OUT_DIR        = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "depth_maps"
OUT_JSON       = OUT_DIR / "depth_validation.json"

WEIGHTS_DIR    = PROJECT_ROOT / "ml" / "weights" / "mono_640x192"
ENCODER_PATH   = WEIGHTS_DIR / "encoder.pth"
DECODER_PATH   = WEIGHTS_DIR / "depth.pth"

# Monodepth2 inference resolution (fixed by the pretrained checkpoint)
MONO_WIDTH  = 640
MONO_HEIGHT = 192

# Central crop fraction for depth extraction
CROP_FRACTION = 0.6

# Class colours for drawing bounding boxes (BGR) — same as detector.py
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "longitudinal_crack":        (0,   0,   255),
    "transverse_crack":          (0,   165, 255),
    "alligator_crack":           (0,   255, 255),
    "repaired_crack":            (255, 0,   255),
    "pothole":                   (255, 0,   0),
    "pedestrian_crossing_blur":  (0,   165, 255),
    "lane_line_blur":            (0,   165, 255),
    "manhole_cover":             (128, 128, 128),
    "patchy_road":               (0,   255, 0),
    "rutting":                   (255, 255, 0),
}


# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------
SETUP_TEXT = """
=== Monodepth2 Setup Instructions ===

Step 1 — Clone the Monodepth2 repository somewhere outside the project:
  cd C:\\Facultate\\pothole-detection
  git clone https://github.com/nianticlabs/monodepth2.git Monodepth

Step 2 — Copy only the networks/ folder into the project weights directory:
  Source : C:\\Facultate\\pothole-detection\\Monodepth\\networks\\
  Dest   : C:\\Facultate\\pothole-detection\\Pothole-Detection\\ml\\weights\\networks\\

  Expected contents after copy:
    ml\\weights\\networks\\
        __init__.py
        depth_decoder.py
        pose_decoder.py
        pose_cnn.py
        resnet_encoder.py

  The full Monodepth2 repo is NOT needed at runtime — only these five files.

Step 3 — Download the mono_640x192 weights:
  https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip

  Extract so you have:
    ml\\weights\\mono_640x192\\
        encoder.pth
        depth.pth

Step 4 — Install dependencies (if not already installed):
  pip install tensorboardX

Step 5 — Verify:
  python scripts/validate_depth.py --limit 5
"""


# ---------------------------------------------------------------------------
# Monodepth2 loader
# ---------------------------------------------------------------------------

def load_monodepth2(device: torch.device) -> Tuple[object, object]:
    """
    Load Monodepth2 encoder and depth decoder from local weights.

    Imports the networks module from NETWORKS_DIR
    (ml/weights/networks/ — five .py files copied from the Monodepth2 repo).
    The full Monodepth2 repo is NOT required at runtime.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    (encoder, depth_decoder) — both in eval() mode on device
    """
    if not NETWORKS_DIR.exists():
        raise FileNotFoundError(
            f"networks/ folder not found at: {NETWORKS_DIR}\n"
            f"Run:  python scripts/validate_depth.py --setup\n"
            f"for setup instructions."
        )
    if not ENCODER_PATH.exists() or not DECODER_PATH.exists():
        raise FileNotFoundError(
            f"Monodepth2 weights not found in: {WEIGHTS_DIR}\n"
            f"Expected:\n"
            f"  {ENCODER_PATH}\n"
            f"  {DECODER_PATH}\n"
            f"Run:  python scripts/validate_depth.py --setup\n"
            f"for download instructions."
        )

    # Inject ml/weights/ into sys.path so that `import networks`
    # resolves to ml/weights/networks/
    weights_str = str(PROJECT_ROOT / "ml" / "weights")
    if weights_str not in sys.path:
        sys.path.insert(0, weights_str)
        logger.info("Added to sys.path: %s", weights_str)

    import networks  # noqa: E402 — imported here after path injection

    logger.info("Loading Monodepth2 encoder from: %s", ENCODER_PATH)
    encoder = networks.ResnetEncoder(18, False)

    # Filter out metadata keys (height, width) present in encoder.pth
    encoder_dict = torch.load(str(ENCODER_PATH), map_location=device)
    model_dict   = encoder.state_dict()
    encoder_dict = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder = encoder.to(device).eval()
    logger.info("Encoder loaded.")

    logger.info("Loading Monodepth2 depth decoder from: %s", DECODER_PATH)
    depth_decoder = networks.DepthDecoder(
        num_ch_enc = encoder.num_ch_enc,
        scales     = range(4),
    )
    depth_decoder.load_state_dict(
        torch.load(str(DECODER_PATH), map_location=device)
    )
    depth_decoder = depth_decoder.to(device).eval()
    logger.info("Depth decoder loaded.")

    return encoder, depth_decoder


# ---------------------------------------------------------------------------
# Depth inference
# ---------------------------------------------------------------------------

def run_depth(
    image_bgr:     np.ndarray,
    encoder:       object,
    depth_decoder: object,
    device:        torch.device,
) -> np.ndarray:
    """
    Run Monodepth2 on a single BGR image.

    The image is resized to MONO_WIDTH x MONO_HEIGHT (640x192) for inference,
    then the output disparity map is resized back to the original image
    dimensions so that bounding box coordinates from the JSON align directly
    with the depth map pixels without any coordinate remapping.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original frame in BGR format, any resolution.
    encoder, depth_decoder : Monodepth2 network objects
    device : torch.device

    Returns
    -------
    depth_map : np.ndarray  float32, shape (H, W)
        Relative depth map at the original image resolution.
        Values are disparity-derived: higher value = closer to camera.
        Not metric depth. Per-frame normalisation required for comparison.
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # Resize to Monodepth2 input resolution
    input_image = cv2.resize(image_bgr, (MONO_WIDTH, MONO_HEIGHT))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Convert to tensor: (1, 3, H, W), normalised to [0, 1]
    input_tensor = torch.from_numpy(input_image).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        features  = encoder(input_tensor)
        outputs   = depth_decoder(features)

    # Use the highest-resolution output (scale 0)
    disp = outputs[("disp", 0)]                    # shape: (1, 1, 192, 640)
    disp = disp.squeeze().cpu().numpy()            # shape: (192, 640)

    # Resize disparity back to original frame resolution
    depth_map = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return depth_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth extraction per bounding box
# ---------------------------------------------------------------------------

def extract_box_depth(
    depth_map:  np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    img_h: int, img_w: int,
) -> Dict:
    """
    Extract depth statistics from the central 60% crop of a bounding box.

    The central crop reduces contamination from adjacent clean road surface
    that often falls inside the bbox edges (RT-DETR boxes are not pixel-tight).

    Per-frame normalisation: depth_norm = (mean_raw - frame_min) / (frame_max - frame_min)
    This is correct because Monodepth2 disparity values are relative within a
    single forward pass and are not comparable across different frames.

    Parameters
    ----------
    depth_map : np.ndarray (H, W) float32 — full frame disparity
    x1, y1, x2, y2 : float — bounding box in pixel coordinates
    img_h, img_w : int — frame dimensions (for clamping)

    Returns
    -------
    dict with keys:
        crop_mean_raw      — mean disparity over central crop (raw, not normalised)
        crop_min_raw       — min disparity in central crop
        crop_max_raw       — max disparity in central crop
        frame_min_raw      — min disparity in full frame
        frame_max_raw      — max disparity in full frame
        depth_norm         — normalised depth [0, 1] within this frame
                             (1.0 = closest to camera, 0.0 = furthest)
        crop_px            — number of pixels in central crop
        crop_fraction_used — actual fraction (may differ if box is near edge)
    """
    frame_min = float(depth_map.min())
    frame_max = float(depth_map.max())
    frame_range = frame_max - frame_min if frame_max > frame_min else 1.0

    # Central crop coordinates
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * CROP_FRACTION / 2.0
    bh = (y2 - y1) * CROP_FRACTION / 2.0

    cx1 = max(0,     int(cx - bw))
    cy1 = max(0,     int(cy - bh))
    cx2 = min(img_w, int(cx + bw))
    cy2 = min(img_h, int(cy + bh))

    crop = depth_map[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        # Fallback: use single centre pixel if crop is degenerate
        px = min(max(int(cx), 0), img_w - 1)
        py = min(max(int(cy), 0), img_h - 1)
        crop_mean = float(depth_map[py, px])
        crop_min  = crop_mean
        crop_max  = crop_mean
        crop_px   = 1
    else:
        crop_mean = float(crop.mean())
        crop_min  = float(crop.min())
        crop_max  = float(crop.max())
        crop_px   = crop.size

    depth_norm = float((crop_mean - frame_min) / frame_range)

    # Actual fraction used (may be smaller near frame edges)
    actual_w = cx2 - cx1
    actual_h = cy2 - cy1
    box_w    = max(x2 - x1, 1.0)
    box_h    = max(y2 - y1, 1.0)
    crop_fraction_used = float((actual_w * actual_h) / (box_w * box_h))

    return {
        "crop_mean_raw":       round(crop_mean, 6),
        "crop_min_raw":        round(crop_min,  6),
        "crop_max_raw":        round(crop_max,  6),
        "frame_min_raw":       round(frame_min, 6),
        "frame_max_raw":       round(frame_max, 6),
        "depth_norm":          round(depth_norm, 4),
        "crop_px":             crop_px,
        "crop_fraction_used":  round(crop_fraction_used, 3),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_depth_colormap(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert a float32 disparity map to a BGR 'magma' colourised image.

    Normalises to [0, 255] uint8 first, then applies the magma colormap.
    Magma is standard in depth estimation literature (light = close,
    dark = far).
    """
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max > d_min:
        normalised = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalised = np.zeros_like(depth_map, dtype=np.uint8)

    # OpenCV COLORMAP_MAGMA available since OpenCV 4.1
    colourised = cv2.applyColorMap(normalised, cv2.COLORMAP_MAGMA)
    return colourised


def draw_boxes_on_frame(image_bgr: np.ndarray, boxes: List[dict]) -> np.ndarray:
    """
    Draw RT-DETR bounding boxes on a copy of the frame.
    Reads box coordinates and class names directly from the JSON dict.
    """
    canvas = image_bgr.copy()
    for box in boxes:
        x1  = int(box["x1"])
        y1  = int(box["y1"])
        x2  = int(box["x2"])
        y2  = int(box["y2"])
        cls = box["class_name"]
        conf = box["confidence"]

        colour = CLASS_COLOURS.get(cls, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(
            canvas, label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
        )
    return canvas


def load_sam_mask_panel(
    frame_stem: str,
    boxes:      List[dict],
    target_h:   int,
    target_w:   int,
) -> np.ndarray:
    """
    Load and composite all SAM mask overlays for a frame into a single panel.

    For frames with multiple detections, all mask overlays are blended
    together using maximum-intensity compositing so every mask is visible.
    If no mask file is found for a box, that box is skipped silently.

    The mask filename convention (confirmed from detections_summary.json):
        frame_stem : "frame_000023_t11.500"
        mask file  : "frame_000023_t11_500_box0_lane_line_blur_mask.jpg"
        transform  : frame_stem.replace(".", "_") + f"_box{i}_{cls}_mask.jpg"
    """
    # stem with dots replaced by underscores for filename lookup
    stem_underscored = frame_stem.replace(".", "_")

    composite = None

    for i, box in enumerate(boxes):
        cls      = box["class_name"]
        filename = f"{stem_underscored}_box{i}_{cls}_mask.jpg"
        mask_path = SAM_MASKS_DIR / filename

        if not mask_path.exists():
            logger.debug("Mask not found (skipping): %s", mask_path)
            continue

        mask_img = cv2.imread(str(mask_path))
        if mask_img is None:
            logger.debug("Cannot read mask image: %s", mask_path)
            continue

        # Resize to target dimensions if needed
        if mask_img.shape[:2] != (target_h, target_w):
            mask_img = cv2.resize(mask_img, (target_w, target_h))

        if composite is None:
            composite = mask_img.astype(np.float32)
        else:
            # Maximum compositing — brightest pixel wins per channel
            composite = np.maximum(composite, mask_img.astype(np.float32))

    if composite is None:
        # No mask files found — return a grey placeholder with a label
        placeholder = np.full((target_h, target_w, 3), 60, dtype=np.uint8)
        cv2.putText(
            placeholder, "No mask saved",
            (10, target_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return placeholder

    return composite.clip(0, 255).astype(np.uint8)


def add_panel_label(panel: np.ndarray, label: str) -> np.ndarray:
    """Add a text label at the top of a panel image."""
    labelled = panel.copy()
    cv2.rectangle(labelled, (0, 0), (panel.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(
        labelled, label,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return labelled


def build_three_panel(
    frame_bgr:    np.ndarray,
    depth_map:    np.ndarray,
    frame_stem:   str,
    boxes:        List[dict],
) -> np.ndarray:
    """
    Compose a three-panel validation image.

    LEFT   — original frame with RT-DETR bounding boxes
    MIDDLE — Monodepth2 depth map (magma colormap)
    RIGHT  — SAM 2.1 mask overlay composite

    All three panels are resized to the original frame height x width
    before horizontal concatenation.
    """
    h, w = frame_bgr.shape[:2]

    # Left panel
    left = draw_boxes_on_frame(frame_bgr, boxes)
    left = add_panel_label(left, "Original + RT-DETR boxes")

    # Middle panel
    middle = build_depth_colormap(depth_map)
    if middle.shape[:2] != (h, w):
        middle = cv2.resize(middle, (w, h), interpolation=cv2.INTER_LINEAR)
    middle = add_panel_label(middle, "Monodepth2 depth map (magma: light=near)")

    # Right panel
    right = load_sam_mask_panel(frame_stem, boxes, h, w)
    right = add_panel_label(right, "SAM 2.1 mask overlays")

    # Concatenate horizontally
    panel = np.concatenate([left, middle, right], axis=1)
    return panel


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def main(limit: Optional[int], device_str: str) -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load detections summary — real data from the pipeline run
    # ------------------------------------------------------------------
    logger.info("Loading detections summary: %s", SUMMARY_JSON)
    if not SUMMARY_JSON.exists():
        logger.error("detections_summary.json not found: %s", SUMMARY_JSON)
        sys.exit(1)

    with SUMMARY_JSON.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    all_frames: List[dict] = summary["detections"]

    # Keep only frames that have at least one detection
    detected_frames = [fr for fr in all_frames if fr.get("boxes")]
    logger.info(
        "Total frames in summary : %d  |  Frames with detections : %d",
        len(all_frames), len(detected_frames),
    )

    if limit is not None:
        detected_frames = detected_frames[:limit]
        logger.info("Limit applied: processing first %d detected frames", limit)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if device_str in ("auto", "cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            if device_str == "cuda":
                logger.error(
                    "CUDA requested but torch.cuda.is_available() returned False.\n"
                    "  Check:\n"
                    "    1. PyTorch CUDA build:  "
                    "python -c \"import torch; print(torch.version.cuda)\"\n"
                    "    2. NVIDIA driver is up to date.\n"
                    "    3. Correct CUDA toolkit version installed.\n"
                    "  Falling back to CPU."
                )
            else:
                logger.warning("CUDA not available — falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info("Device selected: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  VRAM: %.1f GB  |  CUDA: %s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            torch.version.cuda,
        )

    # ------------------------------------------------------------------
    # Load Monodepth2
    # ------------------------------------------------------------------
    encoder, depth_decoder = load_monodepth2(device)

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------
    depth_records: List[dict] = []   # accumulates per-frame depth data for JSON
    t_start = time.perf_counter()

    n_frames_ok      = 0
    n_frames_skipped = 0
    n_boxes_total    = 0

    for frame_record in detected_frames:

        frame_path = Path(frame_record["frame_path"])
        frame_stem = frame_record["frame_stem"]
        boxes      = frame_record["boxes"]

        if not frame_path.exists():
            logger.warning("Frame image not found — skipping: %s", frame_path)
            n_frames_skipped += 1
            continue

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            logger.warning("cv2.imread failed — skipping: %s", frame_path)
            n_frames_skipped += 1
            continue

        img_h, img_w = image_bgr.shape[:2]

        # ------------------------------------------------------------------
        # Run Monodepth2
        # ------------------------------------------------------------------
        try:
            depth_map = run_depth(image_bgr, encoder, depth_decoder, device)
        except Exception:
            logger.exception("Monodepth2 failed on frame: %s", frame_stem)
            n_frames_skipped += 1
            continue

        # ------------------------------------------------------------------
        # Extract per-box depth values
        # ------------------------------------------------------------------
        box_depth_records = []
        for box in boxes:
            depth_stats = extract_box_depth(
                depth_map,
                x1    = box["x1"],
                y1    = box["y1"],
                x2    = box["x2"],
                y2    = box["y2"],
                img_h = img_h,
                img_w = img_w,
            )
            box_depth_records.append({
                "class_id":   box["class_id"],
                "class_name": box["class_name"],
                "confidence": box["confidence"],
                "is_damage":  box["is_damage"],
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "sam_score":  box.get("sam_score"),
                "geometry":   box.get("geometry"),
                "depth":      depth_stats,
            })
            n_boxes_total += 1

        # ------------------------------------------------------------------
        # Build three-panel visualisation
        # ------------------------------------------------------------------
        panel = build_three_panel(image_bgr, depth_map, frame_stem, boxes)

        # Output filename: frame_000023_t11_500_depth.jpg
        out_stem     = frame_stem.replace(".", "_")
        out_img_name = f"{out_stem}_depth.jpg"
        out_img_path = OUT_DIR / out_img_name
        cv2.imwrite(str(out_img_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ------------------------------------------------------------------
        # Accumulate depth record for JSON
        # ------------------------------------------------------------------
        depth_records.append({
            "frame_path":    str(frame_path),
            "frame_stem":    frame_stem,
            "frame_index":   frame_record.get("frame_index"),
            "timestamp_s":   frame_record.get("timestamp_s"),
            "lighting":      frame_record.get("lighting"),
            "sun_elevation": frame_record.get("sun_elevation"),
            "image_width":   img_w,
            "image_height":  img_h,
            "depth_map_min": round(float(depth_map.min()), 6),
            "depth_map_max": round(float(depth_map.max()), 6),
            "depth_map_mean":round(float(depth_map.mean()), 6),
            "boxes":         box_depth_records,
            "panel_saved_to": str(out_img_path),
        })

        n_frames_ok += 1

        logger.info(
            "Frame %-35s | %d box(es) | depth [%.3f, %.3f] | saved: %s",
            frame_stem,
            len(boxes),
            float(depth_map.min()),
            float(depth_map.max()),
            out_img_name,
        )

    # ------------------------------------------------------------------
    # Write depth_validation.json
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start

    output_json = {
        "model":            "mono_640x192",
        "model_reference":  "Godard et al., 2019. arXiv:1806.01260",
        "inference_resolution": f"{MONO_WIDTH}x{MONO_HEIGHT}",
        "crop_fraction":    CROP_FRACTION,
        "device":           str(device),
        "elapsed_seconds":  round(elapsed, 1),
        "frames_processed": n_frames_ok,
        "frames_skipped":   n_frames_skipped,
        "boxes_total":      n_boxes_total,
        "encoder_path":     str(ENCODER_PATH),
        "decoder_path":     str(DECODER_PATH),
        "depth_maps_dir":   str(OUT_DIR),
        "frames":           depth_records,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    logger.info("=== Done ===")
    logger.info("  Frames processed : %d", n_frames_ok)
    logger.info("  Frames skipped   : %d", n_frames_skipped)
    logger.info("  Boxes processed  : %d", n_boxes_total)
    logger.info("  Elapsed          : %.1f s", elapsed)
    logger.info("  Panel images     : %s", OUT_DIR)
    logger.info("  Depth JSON       : %s", OUT_JSON)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_setup() -> None:
    print(SETUP_TEXT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monodepth2 depth estimation validation script (Stage 4 sanity check)."
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Print setup instructions (clone repo + download weights) and exit.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N detected frames (default: all).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0  (default: auto)",
    )
    args = parser.parse_args()

    if args.setup:
        _print_setup()
        sys.exit(0)

    main(limit=args.limit, device_str=args.device)