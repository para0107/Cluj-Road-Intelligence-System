"""
pipeline/segmentor.py
---------------------
Stage 3 of the road damage detection inference pipeline.

Responsibilities:
  - Load SAM 2.1 Tiny (sam2.1_hiera_tiny.pt) once and keep it in memory
  - Accept a list of DetectionResult objects from Stage 2 (Detector)
  - For every frame that has at least one accepted bounding box:
      * Set the image on the SAM predictor (one call per frame)
      * Prompt SAM with all bounding boxes for that frame simultaneously
      * For each returned binary mask:
          - Compute the four geometry features (surface_area_px,
            edge_sharpness, interior_contrast, mask_compactness)
          - Attach the SAM-predicted IoU score (sam_score)
          - Optionally save a mask overlay image to disk
  - Return a list of SegmentationResult objects, one per frame,
    each containing the enriched BoundingBox list

Geometry features (all computed from the binary mask via OpenCV):
  surface_area_px   -- pixel count of mask (damage extent proxy)
  edge_sharpness    -- mean Sobel gradient magnitude on mask boundary
  interior_contrast -- mean intensity difference inside vs outside mask
  mask_compactness  -- 4π·area/perimeter² (circle=1.0, crack≈0.03)

These features are the primary contribution of Stage 3 to Stage 5
(rule-based severity classifier). They cannot be derived from bounding
boxes alone. See: Zhang et al., 2024. doi:10.1016/j.ijtst.2024.10.005

SAM 2.1 reference:
  Ravi et al., 2024. arXiv:2408.00714

SAM 2.1 Tiny variant rationale:
  - Sequential single-image inference; per-call latency is the bottleneck
  - Box-prompted segmentation: spatial prior from RT-DETR bbox limits
    the benefit of larger model capacity
  - 4 GB VRAM constraint on local GPU (RTX 2050) precludes Base/Large/Huge
  - 3-5% mask IoU trade-off versus 4x inference time reduction vs Base

Usage (module):
    from pipeline.preprocessor import Preprocessor
    from pipeline.detector     import Detector, DetectorConfig
    from pipeline.segmentor    import Segmentor, SegmentorConfig

    frames  = Preprocessor.load_manifest("data/processed/frames/run/manifest.json")
    det_cfg = DetectorConfig(weights="ml/weights/rtdetr_l_nrdd2024.pt")
    det     = Detector(det_cfg)
    det_results = det.run(frames)

    seg_cfg = SegmentorConfig(weights="ml/weights/sam2.1_hiera_tiny.pt")
    seg     = Segmentor(seg_cfg)
    seg_results = seg.run(det_results)

Usage (CLI):
    python pipeline/segmentor.py
        --detections  data/processed/detections/run/detections.json
        --weights     ml/weights/sam2.1_hiera_tiny.pt
        --output      data/processed/segmentations/run/
        [--save_masks]
        [--damage_only]
        [--device cuda]
        [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Module-level logger
# Callers attach their own handlers; this module never calls basicConfig.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM 2.1 config path — relative to the sam2 package install root.
# This string is passed directly to build_sam2() and must match the filename
# in sam2/configs/sam2.1/.
# ---------------------------------------------------------------------------
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# ---------------------------------------------------------------------------
# Class colour map for mask overlay images (BGR, same as detector.py).
# Used only when save_masks=True.
# ---------------------------------------------------------------------------
_CLASS_COLOURS: Dict[str, tuple] = {
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

# Damage classes — geometry features are always computed for these.
# Marking and infrastructure classes are also segmented unless
# damage_only=True is passed to Segmentor.run().
_DAMAGE_CLASSES: set[str] = {
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "repaired_crack",
    "pothole",
    "patchy_road",
    "rutting",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SegmentorConfig:
    """
    All tunable parameters for Stage 3.

    weights:
        Path to the SAM 2.1 Tiny checkpoint (.pt file).
        Default: project-relative path used during development.

    sam_config:
        Hydra config name for build_sam2(). Must match the checkpoint variant.
        For the Tiny checkpoint this is "configs/sam2.1/sam2.1_hiera_t.yaml".

    device:
        "cpu", "cuda", "cuda:0", etc.
        SAM 2.1 Tiny fits within 4 GB VRAM alongside RT-DETR-L.

    save_masks:
        If True, write a mask overlay image for every accepted detection.
        Images are saved to output_dir/masks/<frame_stem>_box<i>_<class>_mask.jpg.
        Set False in production to save disk space; useful for debugging.

    damage_only:
        If True, run SAM only on structural damage classes (DAMAGE_CLASSES).
        Marking classes (pedestrian_crossing_blur, lane_line_blur) and
        infrastructure classes (manhole_cover) are passed through with
        geometry=None and sam_score=None.
        Set False (default) to segment all accepted detections.

    min_sam_score:
        Detections whose SAM-predicted IoU score falls below this threshold
        have their geometry fields set to None and are flagged as
        low_sam_quality=True. They are retained in the output but downstream
        stages (severity classifier, deduplicator) can choose to down-weight them.
        Default 0.0 disables filtering (keep all masks regardless of SAM score).
    """
    weights:      str   = "ml/weights/sam2.1_hiera_tiny.pt"
    sam_config:   str   = SAM_CONFIG
    device:       str   = "cpu"
    save_masks:   bool  = False
    damage_only:  bool  = False
    min_sam_score: float = 0.0
    save_debug:   bool  = False   # if True, write per-frame combined mask
                                  # overlays to output_dir/debug/


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MaskGeometry:
    """
    Four geometry features computed from one binary SAM mask.

    surface_area_px:
        Number of True pixels in the mask. Ordinal damage-extent proxy.
        Physical area conversion (cm²) requires a metric depth scale not
        available from Monodepth2 relative depth.

    edge_sharpness:
        Mean Sobel gradient magnitude (sqrt(Gx²+Gy²)) evaluated on the
        one-pixel-wide dilation ring surrounding the mask boundary.
        High values → structurally distinct edge (pothole, manhole cover).
        Low values  → diffuse boundary (faded road marking).

    interior_contrast:
        Absolute difference between mean greyscale intensity inside the mask
        and mean intensity of the thin erosion ring just inside the boundary.
        Captures the photometric shadow effect that distinguishes deep
        depressions (potholes) from flush surface markings.

    mask_compactness:
        4π·area/perimeter², where a perfect circle = 1.0 and an infinitely
        thin line → 0. Empirical class means from Run 3 full-pipeline
        validation: pothole 0.500, alligator_crack 0.313,
        longitudinal_crack 0.293, transverse_crack 0.160, rutting 0.027.
    """
    surface_area_px:   int
    edge_sharpness:    float
    interior_contrast: float
    mask_compactness:  float

    def to_dict(self) -> dict:
        return {
            "surface_area_px":   self.surface_area_px,
            "edge_sharpness":    round(self.edge_sharpness,    4),
            "interior_contrast": round(self.interior_contrast, 4),
            "mask_compactness":  round(self.mask_compactness,  4),
        }


@dataclass
class SegmentedBox:
    """
    One bounding box from Stage 2, enriched with SAM 2.1 outputs.

    All fields from BoundingBox are preserved (copied by value).
    Three additional fields are added:

    sam_score:
        SAM's self-predicted IoU for its own mask (range 0–1).
        This is the second return value of sam_predictor.predict().
        None if SAM was not run on this detection (damage_only=True and
        class is not in DAMAGE_CLASSES, or SAM raised an exception).

    geometry:
        MaskGeometry with the four features. None under the same conditions
        as sam_score=None, or if sam_score < min_sam_score.

    low_sam_quality:
        True when 0 < sam_score < min_sam_score. Signals to downstream
        stages that this mask's geometry features are unreliable.
    """
    # --- copied from BoundingBox ---
    x1:                float
    y1:                float
    x2:                float
    y2:                float
    class_id:          int
    class_name:        str
    confidence:        float
    threshold_applied: float
    severity_prior:    str
    is_damage:         bool
    is_infrastructure: bool
    is_marking:        bool
    # --- added by Stage 3 ---
    sam_score:         Optional[float]
    geometry:          Optional[MaskGeometry]
    low_sam_quality:   bool = False
    # --- runtime-only handoff to Stage 4 (NOT serialised to JSON) ---
    # The exact SAM binary mask, cropped tight to its nonzero bounding box,
    # as uint8 (0/255). mask_origin is the (y, x) of the crop top-left in the
    # full frame. These let Stage 4 use the real mask instead of an ellipse
    # approximation. They are dropped on save/load (JSON stores geometry only),
    # so a standalone Stage 4 run from segmentations.json still falls back to
    # the approximation.
    mask:        Optional["np.ndarray"] = field(default=None, repr=False, compare=False)
    mask_origin: Optional[tuple]        = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "x1":                round(self.x1, 2),
            "y1":                round(self.y1, 2),
            "x2":                round(self.x2, 2),
            "y2":                round(self.y2, 2),
            "class_id":          self.class_id,
            "class_name":        self.class_name,
            "confidence":        round(self.confidence, 4),
            "threshold_applied": self.threshold_applied,
            "severity_prior":    self.severity_prior,
            "is_damage":         self.is_damage,
            "is_infrastructure": self.is_infrastructure,
            "is_marking":        self.is_marking,
            "sam_score":         round(self.sam_score, 4) if self.sam_score is not None else None,
            "low_sam_quality":   self.low_sam_quality,
            "geometry":          self.geometry.to_dict() if self.geometry is not None else None,
        }


@dataclass
class SegmentationResult:
    """
    Stage 3 output for one frame: the enriched list of SegmentedBox objects
    plus the frame metadata forwarded from Stage 2 unchanged.
    """
    frame_path:    str
    frame_index:   int
    timestamp_s:   float
    latitude:      Optional[float]
    longitude:     Optional[float]
    lighting:      str
    sun_elevation: Optional[float]
    image_width:   int
    image_height:  int
    boxes:         List[SegmentedBox] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.boxes) > 0

    @property
    def n_detections(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> dict:
        return {
            "frame_path":    self.frame_path,
            "frame_index":   self.frame_index,
            "timestamp_s":   self.timestamp_s,
            "latitude":      self.latitude,
            "longitude":     self.longitude,
            "lighting":      self.lighting,
            "sun_elevation": self.sun_elevation,
            "image_width":   self.image_width,
            "image_height":  self.image_height,
            "boxes":         [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Geometry computation
# Ported directly from scripts/detect_and_sam.py — do not modify the logic
# without also updating the validation script and chapter 3 of the thesis.
# ---------------------------------------------------------------------------

def _compute_geometry(mask_uint8: np.ndarray, image_bgr: np.ndarray) -> MaskGeometry:
    """
    Compute four geometry features from a binary SAM mask.

    Parameters
    ----------
    mask_uint8 : np.ndarray
        Binary uint8 mask (255 = damage, 0 = background), same H×W as image.
        Typically produced by (sam_mask > 0.5).astype(np.uint8) * 255.
    image_bgr : np.ndarray
        Original frame in BGR format as returned by cv2.imread().

    Returns
    -------
    MaskGeometry

    Notes
    -----
    edge_sharpness uses a 5×5 dilation ring rather than erosion so that the
    boundary region falls *outside* the mask and captures the contrast between
    damage and clean road surface, not the internal texture of the damage.

    interior_contrast uses a 5×5 erosion with 2 iterations to produce a thin
    interior ring, then computes the absolute intensity difference between the
    mask interior and that ring. This avoids sensitivity to the overall scene
    brightness and instead measures local photometric step at the mask edge.
    """
    binary = (mask_uint8 > 127).astype(np.uint8)

    # ---- 1. Surface area ------------------------------------------------
    surface_area_px = int(np.sum(binary))

    # ---- 2. Edge sharpness ----------------------------------------------
    gray      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated  = cv2.dilate(binary, kernel, iterations=1)
    boundary = dilated - binary          # one-pixel ring outside the mask

    if boundary.sum() > 0:
        edge_sharpness = float(np.mean(sobel_mag[boundary > 0]))
    else:
        edge_sharpness = 0.0

    # ---- 3. Interior contrast -------------------------------------------
    eroded       = cv2.erode(binary, kernel, iterations=2)
    inside_ring  = binary - eroded       # thin ring inside the mask boundary

    if binary.sum() > 0 and inside_ring.sum() > 0:
        mean_inside  = float(np.mean(gray[binary > 0]))
        mean_ring    = float(np.mean(gray[inside_ring > 0]))
        interior_contrast = abs(mean_inside - mean_ring)
    else:
        interior_contrast = 0.0

    # ---- 4. Mask compactness -------------------------------------------
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
        if perimeter > 0:
            mask_compactness = float(4 * np.pi * surface_area_px / (perimeter ** 2))
        else:
            mask_compactness = 0.0
    else:
        mask_compactness = 0.0

    return MaskGeometry(
        surface_area_px   = surface_area_px,
        edge_sharpness    = round(edge_sharpness,    4),
        interior_contrast = round(interior_contrast, 4),
        mask_compactness  = round(mask_compactness,  4),
    )


def _overlay_mask(image_bgr: np.ndarray, mask_uint8: np.ndarray, colour: tuple) -> np.ndarray:
    """Blend a binary SAM mask onto the image at 40% opacity (same as detect_and_sam.py)."""
    overlay = image_bgr.copy()
    overlay[mask_uint8 > 127] = (
        np.array(colour, dtype=np.uint8) * 0.6
        + overlay[mask_uint8 > 127] * 0.4
    ).astype(np.uint8)
    return overlay


def _crop_mask(mask_uint8: np.ndarray) -> tuple:
    """
    Crop a full-frame binary mask to its nonzero bounding box.

    Returns (crop_uint8, (y0, x0)) where crop is the tight nonzero region and
    (y0, x0) is its top-left origin in the full frame. Returns (None, None)
    if the mask is empty. Storing the tight crop keeps the per-detection
    memory cost small while preserving the exact mask for Stage 4.
    """
    ys, xs = np.where(mask_uint8 > 127)
    if ys.size == 0:
        return None, None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return mask_uint8[y0:y1, x0:x1].copy(), (y0, x0)


def _combined_overlay(image_bgr: np.ndarray, items: list) -> np.ndarray:
    """
    Blend every mask for one frame onto a single image, then draw each
    bounding box and label on top.

    Parameters
    ----------
    image_bgr : np.ndarray
        The source frame (BGR).
    items : list of (mask_uint8, colour, (x1, y1, x2, y2), label)
        One entry per segmented box.

    Returns a new image; the input is not modified.
    """
    out = image_bgr.copy()
    for mask_uint8, colour, _bbox, _label in items:
        sel = mask_uint8 > 127
        if sel.any():
            out[sel] = (
                np.array(colour, dtype=np.uint8) * 0.6 + out[sel] * 0.4
            ).astype(np.uint8)

    for _mask, colour, bbox, label in items:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ly - th - base), (x1 + tw, ly + base), colour, -1)
        cv2.putText(
            out, label, (x1, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Segmentor
# ---------------------------------------------------------------------------

class Segmentor:
    """
    Stage 3 — SAM 2.1 segmentation and geometry feature extraction.

    Typical usage:
        cfg = SegmentorConfig(weights="ml/weights/sam2.1_hiera_tiny.pt",
                              device="cuda", save_masks=False)
        seg = Segmentor(cfg)
        results = seg.run(detection_results, output_dir="data/processed/segmentations/run/")
    """

    def __init__(self, cfg: SegmentorConfig) -> None:
        self.cfg = cfg
        self._sam_predictor = None   # loaded lazily on first run() call

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_sam(self) -> None:
        """Load SAM 2.1 Tiny. Called once; subsequent run() calls reuse it."""
        weights_path = Path(self.cfg.weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"SAM 2.1 weights not found: {weights_path}\n"
                f"Download with:\n"
                f"  python -c \"from sam2 import build_sam2; "
                f"build_sam2('{self.cfg.sam_config}', '{weights_path}', device='{self.cfg.device}')\""
            )

        logger.info(
            "Loading SAM 2.1 Tiny: %s  (device=%s)", weights_path, self.cfg.device
        )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_model = build_sam2(
            self.cfg.sam_config,
            str(weights_path),
            device=self.cfg.device,
        )
        self._sam_predictor = SAM2ImagePredictor(sam_model)

        logger.info("SAM 2.1 Tiny loaded successfully.")
        if self.cfg.device.startswith("cuda") or self.cfg.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                logger.info(
                    "GPU: %s  |  VRAM: %.1f GB",
                    props.name, props.total_memory / 1e9,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        detection_results: list,           # List[DetectionResult] from Stage 2
        output_dir: Optional[str] = None,
    ) -> List[SegmentationResult]:
        """
        Run SAM 2.1 segmentation on all frames with detections.

        Parameters
        ----------
        detection_results : list of DetectionResult
            Output from Detector.run(). Frames with no detections are
            forwarded as empty SegmentationResult objects.
        output_dir : str or None
            If provided: segmentations.json is saved here.
            If save_masks=True: mask overlays are saved to output_dir/masks/.

        Returns
        -------
        list of SegmentationResult
            One per frame. Frames without detections have boxes=[].
        """
        if self._sam_predictor is None:
            self._load_sam()

        masks_dir: Optional[Path] = None
        if output_dir and self.cfg.save_masks:
            masks_dir = Path(output_dir) / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Mask overlay images will be saved to: %s", masks_dir)

        debug_dir: Optional[Path] = None
        if output_dir and self.cfg.save_debug:
            debug_dir = Path(output_dir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Combined mask overlays will be saved to: %s", debug_dir)

        results: List[SegmentationResult] = []
        t_start = time.perf_counter()

        n_frames_total    = len(detection_results)
        n_frames_with_det = sum(1 for r in detection_results if r.has_detections)
        n_boxes_total     = sum(r.n_detections for r in detection_results)

        logger.info(
            "Starting SAM segmentation: %d frames total, %d with detections, %d boxes",
            n_frames_total, n_frames_with_det, n_boxes_total,
        )

        n_sam_run      = 0   # frames SAM was actually called on
        n_boxes_seg    = 0   # boxes with geometry features
        n_boxes_skipped = 0  # boxes where SAM was skipped (damage_only)
        n_boxes_failed  = 0  # boxes where SAM raised an exception
        n_low_quality   = 0  # boxes below min_sam_score

        for det_result in detection_results:

            # Build the SegmentationResult shell — metadata forwarded unchanged
            seg_result = SegmentationResult(
                frame_path    = det_result.frame_path,
                frame_index   = det_result.frame_index,
                timestamp_s   = det_result.timestamp_s,
                latitude      = det_result.latitude,
                longitude     = det_result.longitude,
                lighting      = det_result.lighting,
                sun_elevation = det_result.sun_elevation,
                image_width   = det_result.image_width,
                image_height  = det_result.image_height,
                boxes         = [],
            )

            # Frames without detections — forward unchanged
            if not det_result.has_detections:
                results.append(seg_result)
                continue

            # Decide which boxes to segment
            boxes_to_segment = []
            boxes_passthrough = []

            for bbox in det_result.boxes:
                if self.cfg.damage_only and bbox.class_name not in _DAMAGE_CLASSES:
                    boxes_passthrough.append(bbox)
                else:
                    boxes_to_segment.append(bbox)

            # Load image
            image_bgr = cv2.imread(det_result.frame_path)
            if image_bgr is None:
                logger.warning(
                    "Cannot read image for frame %d: %s — skipping SAM for this frame",
                    det_result.frame_index, det_result.frame_path,
                )
                # Attach all boxes as pass-through with no geometry
                for bbox in det_result.boxes:
                    seg_result.boxes.append(_passthrough_box(bbox))
                results.append(seg_result)
                n_boxes_skipped += len(det_result.boxes)
                continue

            # Attach pass-through boxes (damage_only mode)
            for bbox in boxes_passthrough:
                seg_result.boxes.append(_passthrough_box(bbox))
                n_boxes_skipped += 1

            if not boxes_to_segment:
                results.append(seg_result)
                continue

            # Run SAM on this frame
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                self._sam_predictor.set_image(image_rgb)
            except Exception:
                logger.exception(
                    "SAM set_image failed for frame %d: %s",
                    det_result.frame_index, det_result.frame_path,
                )
                for bbox in boxes_to_segment:
                    seg_result.boxes.append(_passthrough_box(bbox))
                results.append(seg_result)
                n_boxes_failed += len(boxes_to_segment)
                continue

            sam_boxes_np = np.array(
                [[b.x1, b.y1, b.x2, b.y2] for b in boxes_to_segment],
                dtype=np.float32,
            )

            try:
                masks, scores, _ = self._sam_predictor.predict(
                    point_coords   = None,
                    point_labels   = None,
                    box            = sam_boxes_np,
                    multimask_output = False,
                )
                # Normalise shape: (N, 1, H, W) → (N, H, W)
                if masks.ndim == 4:
                    masks = masks[:, 0, :, :]

            except Exception:
                logger.exception(
                    "SAM predict failed for frame %d: %s",
                    det_result.frame_index, det_result.frame_path,
                )
                for bbox in boxes_to_segment:
                    seg_result.boxes.append(_passthrough_box(bbox))
                results.append(seg_result)
                n_boxes_failed += len(boxes_to_segment)
                continue

            n_sam_run += 1
            frame_stem = Path(det_result.frame_path).stem

            # Items for the optional combined per-frame debug overlay
            debug_items: list = []

            # Per-box: geometry + optional mask save
            for i, (bbox, mask, sam_score) in enumerate(
                zip(boxes_to_segment, masks, scores)
            ):
                mask_uint8  = (mask > 0.5).astype(np.uint8) * 255
                score_float = float(sam_score)

                # Low SAM quality flag
                low_quality = (
                    self.cfg.min_sam_score > 0.0
                    and score_float < self.cfg.min_sam_score
                )

                if low_quality:
                    geometry = None
                    n_low_quality += 1
                    logger.debug(
                        "Frame %d box %d (%s): sam_score=%.3f < min_sam_score=%.3f — "
                        "geometry set to None",
                        det_result.frame_index, i, bbox.class_name,
                        score_float, self.cfg.min_sam_score,
                    )
                else:
                    geometry = _compute_geometry(mask_uint8, image_bgr)
                    n_boxes_seg += 1

                seg_box = SegmentedBox(
                    x1                = bbox.x1,
                    y1                = bbox.y1,
                    x2                = bbox.x2,
                    y2                = bbox.y2,
                    class_id          = bbox.class_id,
                    class_name        = bbox.class_name,
                    confidence        = bbox.confidence,
                    threshold_applied = bbox.threshold_applied,
                    severity_prior    = bbox.severity_prior,
                    is_damage         = bbox.is_damage,
                    is_infrastructure = bbox.is_infrastructure,
                    is_marking        = bbox.is_marking,
                    sam_score         = round(score_float, 4),
                    geometry          = geometry,
                    low_sam_quality   = low_quality,
                )

                # Keep the exact mask (tight crop) in memory for Stage 4.
                # Not serialised to JSON — dropped on save/load.
                crop, origin = _crop_mask(mask_uint8)
                seg_box.mask        = crop
                seg_box.mask_origin = origin

                seg_result.boxes.append(seg_box)

                # Collect for the combined debug overlay
                if debug_dir is not None:
                    colour = _CLASS_COLOURS.get(bbox.class_name, (0, 255, 0))
                    label  = f"{bbox.class_name} {bbox.confidence:.2f}"
                    debug_items.append(
                        (mask_uint8, colour, (bbox.x1, bbox.y1, bbox.x2, bbox.y2), label)
                    )

                # Optionally save per-box mask overlay image (legacy save_masks)
                if masks_dir is not None:
                    colour       = _CLASS_COLOURS.get(bbox.class_name, (0, 255, 0))
                    overlay      = _overlay_mask(image_bgr, mask_uint8, colour)
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
                    label = f"{bbox.class_name} {bbox.confidence:.2f}"
                    cv2.putText(
                        overlay, label,
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
                    )
                    mask_filename = f"{frame_stem}_box{i}_{bbox.class_name}_mask.jpg"
                    cv2.imwrite(str(masks_dir / mask_filename), overlay)

            # Combined per-frame mask overlay (one image per detection frame)
            if debug_dir is not None and debug_items:
                combined = _combined_overlay(image_bgr, debug_items)
                cv2.imwrite(str(debug_dir / f"{frame_stem}.jpg"), combined)

            results.append(seg_result)

            logger.info(
                "Frame %-40s | %d box(es) segmented | %s",
                frame_stem,
                len(boxes_to_segment),
                [f"{b.class_name}(sam={b.sam_score:.2f})" for b in seg_result.boxes
                 if b.sam_score is not None],
            )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t_start

        logger.info("=== Segmentation complete ===")
        logger.info("  Frames processed         : %d", n_frames_total)
        logger.info("  Frames SAM was called on : %d", n_sam_run)
        logger.info("  Boxes segmented          : %d", n_boxes_seg)
        logger.info("  Boxes skipped (pass-thru): %d", n_boxes_skipped)
        logger.info("  Boxes failed (SAM error) : %d", n_boxes_failed)
        logger.info("  Boxes low SAM quality    : %d  (sam_score < %.2f)",
                    n_low_quality, self.cfg.min_sam_score)
        logger.info("  Elapsed                  : %.1f s", elapsed)

        if output_dir:
            self.save_segmentations(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_segmentations(
        results: List[SegmentationResult],
        output_dir: str,
    ) -> str:
        """Save segmentation results to segmentations.json in output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "segmentations.json"

        n_boxes = sum(r.n_detections for r in results)
        n_with_geom = sum(
            1 for r in results for b in r.boxes if b.geometry is not None
        )

        payload = {
            "n_frames":           len(results),
            "n_boxes_total":      n_boxes,
            "n_boxes_with_geom":  n_with_geom,
            "frames":             [r.to_dict() for r in results],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Segmentations saved: %s  (%d frames, %d boxes, %d with geometry)",
            out_path, len(results), n_boxes, n_with_geom,
        )
        return str(out_path)

    @staticmethod
    def load_segmentations(segmentations_path: str) -> List[SegmentationResult]:
        """Load a saved segmentations.json back into SegmentationResult objects."""
        with open(segmentations_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results: List[SegmentationResult] = []
        for fr in payload["frames"]:
            boxes = []
            for b in fr["boxes"]:
                geom_dict = b.get("geometry")
                geometry = (
                    MaskGeometry(
                        surface_area_px   = geom_dict["surface_area_px"],
                        edge_sharpness    = geom_dict["edge_sharpness"],
                        interior_contrast = geom_dict["interior_contrast"],
                        mask_compactness  = geom_dict["mask_compactness"],
                    )
                    if geom_dict is not None else None
                )
                boxes.append(SegmentedBox(
                    x1                = b["x1"],
                    y1                = b["y1"],
                    x2                = b["x2"],
                    y2                = b["y2"],
                    class_id          = b["class_id"],
                    class_name        = b["class_name"],
                    confidence        = b["confidence"],
                    threshold_applied = b.get("threshold_applied", 0.35),
                    severity_prior    = b.get("severity_prior", "S2"),
                    is_damage         = b["is_damage"],
                    is_infrastructure = b["is_infrastructure"],
                    is_marking        = b["is_marking"],
                    sam_score         = b.get("sam_score"),
                    geometry          = geometry,
                    low_sam_quality   = b.get("low_sam_quality", False),
                ))
            results.append(SegmentationResult(
                frame_path    = fr["frame_path"],
                frame_index   = fr["frame_index"],
                timestamp_s   = fr["timestamp_s"],
                latitude      = fr.get("latitude"),
                longitude     = fr.get("longitude"),
                lighting      = fr.get("lighting", "unknown"),
                sun_elevation = fr.get("sun_elevation"),
                image_width   = fr["image_width"],
                image_height  = fr["image_height"],
                boxes         = boxes,
            ))

        logger.info(
            "Segmentations loaded: %s  (%d frames)", segmentations_path, len(results)
        )
        return results


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _passthrough_box(bbox) -> SegmentedBox:
    """
    Wrap a BoundingBox from Stage 2 into a SegmentedBox with no SAM output.
    Used for frames where SAM failed, or damage_only=True and class is not
    in DAMAGE_CLASSES.
    """
    return SegmentedBox(
        x1                = bbox.x1,
        y1                = bbox.y1,
        x2                = bbox.x2,
        y2                = bbox.y2,
        class_id          = bbox.class_id,
        class_name        = bbox.class_name,
        confidence        = bbox.confidence,
        threshold_applied = bbox.threshold_applied,
        severity_prior    = bbox.severity_prior,
        is_damage         = bbox.is_damage,
        is_infrastructure = bbox.is_infrastructure,
        is_marking        = bbox.is_marking,
        sam_score         = None,
        geometry          = None,
        low_sam_quality   = False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 -- SAM 2.1 Tiny segmentation and geometry extraction."
    )
    parser.add_argument(
        "--detections", required=True,
        help="detections.json produced by Stage 2 (Detector)",
    )
    parser.add_argument(
        "--weights",
        default="ml/weights/sam2.1_hiera_tiny.pt",
        help="Path to SAM 2.1 Tiny checkpoint (default: ml/weights/sam2.1_hiera_tiny.pt)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for segmentations.json (and masks/ if --save_masks)",
    )
    parser.add_argument(
        "--save_masks", action="store_true",
        help="Save per-box mask overlay images to output/masks/ (legacy)",
    )
    parser.add_argument(
        "--save_debug", action="store_true",
        help="Save one combined mask overlay per detection frame to output/debug/",
    )
    parser.add_argument(
        "--damage_only", action="store_true",
        help="Only run SAM on structural damage classes; skip marking/infrastructure",
    )
    parser.add_argument(
        "--min_sam_score", type=float, default=0.0,
        help="Minimum SAM-predicted IoU score to compute geometry (default 0.0 = disabled)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu | cuda | cuda:0  (default: cpu)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from pipeline.detector import Detector
    detection_results = Detector.load_detections(args.detections)
    logger.info(
        "Loaded %d detection results from: %s",
        len(detection_results), args.detections,
    )

    cfg = SegmentorConfig(
        weights       = args.weights,
        device        = args.device,
        save_masks    = args.save_masks,
        damage_only   = args.damage_only,
        min_sam_score = args.min_sam_score,
        save_debug    = args.save_debug,
    )
    segmentor = Segmentor(cfg)
    segmentor.run(detection_results, output_dir=args.output)


if __name__ == "__main__":
    main()