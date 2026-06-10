"""
pipeline/depth_estimator.py
----------------------------
Stage 4 of the road damage detection inference pipeline.

Responsibilities:
  - Load Monodepth2 (mono_640x192, KITTI pretrained) once and keep it
    in memory across all frames.
  - Accept a list of SegmentationResult objects from Stage 3 (Segmentor).
  - For every frame that has at least one accepted bounding box:
      * Run Monodepth2 to produce a dense relative disparity map.
      * For each detection in that frame:
          - If a valid MaskGeometry is present (geometry is not None):
              Use the SAM mask region for depth extraction.
              The binary mask is reconstructed from the surface_area_px
              and bounding box coordinates via a filled ellipse approximation
              (see _mask_from_geometry() — exact mask pixels are not stored
               in SegmentationResult to keep the JSON compact).
          - Fallback (geometry is None or low_sam_quality=True):
              Use the central 60% crop of the bounding box.
          - Additional fallback (lighting == "low_light" or
            depth_confidence < DEPTH_CONF_THRESHOLD):
              Use a SAM-geometry-based proxy depth derived from
              surface_area_px and mask_compactness (see _proxy_depth()).
      * Compute depth_confidence as the coefficient of variation (std/mean)
        of the depth values within the extraction region, inverted and
        clipped to [0, 1]. High spatial variance → low confidence.
      * Normalise depth to [0, 1] per frame (per-frame normalisation is
        correct because Monodepth2 disparity is relative within a single
        forward pass, not metric).
  - Return a list of DepthResult objects, one per frame.

Depth confidence rationale:
  Monodepth2 produces reliable relative depth on textured outdoor surfaces
  but struggles in textureless regions (uniform asphalt, overexposed sky)
  and in low-light conditions. The coefficient of variation of depth values
  within the extraction region is a proxy for this: high spatial variance
  means the disparity surface is unstable in that region.
  Threshold: depth_confidence < 0.4 → fallback to geometry proxy.
  (This matches the threshold documented in Chapter 3 of the thesis.)

Mask reconstruction rationale:
  SegmentationResult stores MaskGeometry (4 scalar features) but not the
  binary mask array — storing full masks in JSON would be impractical
  (640×360 bool array per detection). To use mask-region depth extraction
  in the pipeline, _mask_from_geometry() reconstructs an approximation of
  the mask from the bounding box + compactness value using a filled ellipse.
  Compactness ≈ 1.0 → circle; compactness → 0 → elongated horizontal strip.
  This approximation is good enough for mean depth extraction and avoids
  the JPEG colour-channel reconstruction fragility of the validation script.

Monodepth2 reference:
  Godard et al., 2019. arXiv:1806.01260
  "Digging into Self-Supervised Monocular Depth Estimation"

Monodepth2 repo (must be cloned separately — not a project dependency):
  https://github.com/nianticlabs/monodepth2
  Expected location: C:\\Facultate\\pothole-detection\\Monodepth
  Imported via sys.path injection — not installed as a package.

Model variant: mono_640x192
  Monocular, KITTI pretrained. Lightest checkpoint; sufficient for ordinal
  severity proxy. VRAM ~0.8 GB — compatible with RTX 2050 (4 GB) alongside
  RT-DETR-L and SAM 2.1 Tiny.

Usage (module):
    from pipeline.segmentor       import Segmentor, SegmentorConfig
    from pipeline.depth_estimator import DepthEstimator, DepthEstimatorConfig

    seg_results = Segmentor(SegmentorConfig(...)).run(det_results)

    de_cfg = DepthEstimatorConfig(
                 monodepth_root = r"C:\\Facultate\\pothole-detection\\Monodepth",
                 weights_dir    = "ml/weights/mono_640x192",
                 device         = "cuda",
             )
    de      = DepthEstimator(de_cfg)
    results = de.run(seg_results)

Usage (CLI):
    python pipeline/depth_estimator.py
        --segmentations  data/processed/segmentations/run/segmentations.json
        --monodepth_root C:\\Facultate\\pothole-detection\\Monodepth
        --weights_dir    ml/weights/mono_640x192
        --output         data/processed/depth/run/
        [--device cuda]
        [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monodepth2 inference constants
# ---------------------------------------------------------------------------
MONO_WIDTH  = 640    # fixed by the mono_640x192 checkpoint
MONO_HEIGHT = 192    # fixed by the mono_640x192 checkpoint

# Central crop fraction used when SAM mask is unavailable
CROP_FRACTION = 0.6

# Depth confidence threshold below which geometry proxy is used
DEPTH_CONF_THRESHOLD = 0.4

# Colormap for the disparity debug image. Monodepth2's own test_simple.py uses
# matplotlib 'magma'. cv2.COLORMAP_MAGMA exists in modern OpenCV (>= 3.4); fall
# back to COLORMAP_JET (always present) on older builds so this never crashes.
_DEPTH_COLORMAP = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DepthEstimatorConfig:
    """
    All tunable parameters for Stage 4.

    monodepth_root:
        Absolute path to the cloned Monodepth2 repository.
        The script imports networks/ from here via sys.path injection.
        Must be cloned from https://github.com/nianticlabs/monodepth2

    weights_dir:
        Directory containing encoder.pth and depth.pth for mono_640x192.
        Default: project-relative path used during development.

    device:
        "cpu", "cuda", "cuda:0", etc.
        mono_640x192 requires ~0.8 GB VRAM.

    depth_conf_threshold:
        Detections with depth_confidence below this value fall back to
        the SAM geometry proxy. Default 0.4 (documented in thesis Ch. 3).

    crop_fraction:
        Fraction of the bounding box used for central-crop depth extraction
        when no SAM mask geometry is available. Default 0.6.
    """
    monodepth_root:      str   = r"C:\Facultate\pothole-detection\Monodepth"
    weights_dir:         str   = r"C:\Facultate\pothole-detection\Pothole-Detection\ml\weights\mono_640x192"
    device:              str   = "cpu"
    depth_conf_threshold: float = DEPTH_CONF_THRESHOLD
    crop_fraction:       float = CROP_FRACTION
    save_debug:          bool  = False   # write colourised disparity images to
                                        # output_dir/debug/ for detection frames
    use_exact_mask_depth: bool = False   # if True and the real SAM mask is
                                        # available (orchestrator run), use it
                                        # for the depth extraction region instead
                                        # of the ellipse approximation. Default
                                        # False to keep validated numbers stable.
                                        # The debug overlay always uses the exact
                                        # mask when present, regardless of this.


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DepthEstimate:
    """
    Depth information for one bounding box detection.

    depth_raw:
        Mean Monodepth2 disparity value over the extraction region
        (mask region or central crop). Raw, not normalised.
        Higher disparity = closer to camera in Monodepth2.

    depth_norm:
        Depth normalised to [0, 1] within the frame.
        (depth_raw - frame_min) / (frame_max - frame_min).
        1.0 = closest to camera in this frame.
        0.0 = furthest from camera in this frame.

    depth_confidence:
        Reliability estimate for this depth measurement [0, 1].
        Computed as 1 - clipped_coefficient_of_variation.
        High spatial variance in the extraction region → low confidence.
        Values below DepthEstimatorConfig.depth_conf_threshold trigger
        fallback to geometry proxy.

    extraction_method:
        One of: "mask_region" | "central_crop" | "geometry_proxy"
        Documents which extraction path was used for this detection.

    used_proxy:
        True if depth_confidence < threshold or lighting == "low_light"
        and geometry proxy was used instead of Monodepth2 output.
    """
    depth_raw:          float
    depth_norm:         float
    depth_confidence:   float
    extraction_method:  str
    used_proxy:         bool
    frame_min_raw:      float
    frame_max_raw:      float
    region_px:          int       # number of pixels in extraction region

    def to_dict(self) -> dict:
        return {
            "depth_raw":         round(self.depth_raw,        6),
            "depth_norm":        round(self.depth_norm,        4),
            "depth_confidence":  round(self.depth_confidence,  4),
            "extraction_method": self.extraction_method,
            "used_proxy":        self.used_proxy,
            "frame_min_raw":     round(self.frame_min_raw,     6),
            "frame_max_raw":     round(self.frame_max_raw,     6),
            "region_px":         self.region_px,
        }


@dataclass
class DepthBox:
    """
    One bounding box from Stage 3, enriched with depth information.
    All Stage 3 fields are preserved unchanged.
    """
    # --- forwarded from SegmentedBox ---
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
    sam_score:         Optional[float]
    geometry:          Optional[object]   # MaskGeometry or None
    low_sam_quality:   bool
    # --- added by Stage 4 ---
    depth:             Optional[DepthEstimate]   # None if depth failed
    # --- runtime-only handoff from Stage 3 (NOT serialised) ---
    mask:        Optional["np.ndarray"] = field(default=None, repr=False, compare=False)
    mask_origin: Optional[tuple]        = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        geom = self.geometry
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
            "geometry":          geom.to_dict() if geom is not None else None,
            "depth":             self.depth.to_dict() if self.depth is not None else None,
        }


@dataclass
class DepthResult:
    """
    Stage 4 output for one frame.
    Metadata is forwarded from Stage 3 unchanged.
    """
    frame_path:      str
    frame_index:     int
    timestamp_s:     float
    latitude:        Optional[float]
    longitude:       Optional[float]
    lighting:        str
    sun_elevation:   Optional[float]
    image_width:     int
    image_height:    int
    depth_map_min:   Optional[float]   # None if Monodepth2 was not run
    depth_map_max:   Optional[float]
    depth_map_mean:  Optional[float]
    boxes:           List[DepthBox] = field(default_factory=list)

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
            "depth_map_min": round(self.depth_map_min, 6) if self.depth_map_min is not None else None,
            "depth_map_max": round(self.depth_map_max, 6) if self.depth_map_max is not None else None,
            "depth_map_mean":round(self.depth_map_mean,6) if self.depth_map_mean is not None else None,
            "boxes":         [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_monodepth2(
    monodepth_root: str,
    weights_dir:    str,
    device:         torch.device,
) -> Tuple[object, object]:
    """
    Load Monodepth2 encoder and depth decoder.

    Injects monodepth_root into sys.path and imports networks/.
    Both models are returned in eval() mode on device.

    Parameters
    ----------
    monodepth_root : str
        Path to the cloned monodepth2 repository.
    weights_dir : str
        Directory containing encoder.pth and depth.pth.
    device : torch.device

    Returns
    -------
    (encoder, depth_decoder)
    """
    root = Path(monodepth_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Monodepth2 repo not found: {root}\n"
            f"Clone with: git clone https://github.com/nianticlabs/monodepth2.git {root}"
        )

    encoder_path = Path(weights_dir) / "encoder.pth"
    decoder_path = Path(weights_dir) / "depth.pth"

    for p in (encoder_path, decoder_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Monodepth2 weight not found: {p}\n"
                f"Download mono_640x192.zip from:\n"
                f"  https://storage.googleapis.com/niantic-lon-static/research/"
                f"monodepth2/mono_640x192.zip\n"
                f"and extract encoder.pth + depth.pth to: {Path(weights_dir)}"
            )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        logger.info("sys.path: added Monodepth2 root: %s", root_str)

    import networks  # noqa: PLC0415

    logger.info("Loading Monodepth2 encoder: %s", encoder_path)
    encoder      = networks.ResnetEncoder(18, False)
    encoder_dict = torch.load(str(encoder_path), map_location=device)
    model_dict   = encoder.state_dict()
    # Filter out metadata keys (e.g. "height", "width") present in encoder.pth
    filtered     = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(filtered, strict=False)
    encoder = encoder.to(device).eval()
    logger.info("Encoder loaded  (%d / %d keys matched).", len(filtered), len(model_dict))

    logger.info("Loading Monodepth2 depth decoder: %s", decoder_path)
    depth_decoder = networks.DepthDecoder(
        num_ch_enc = encoder.num_ch_enc,
        scales     = range(4),
    )
    depth_decoder.load_state_dict(
        torch.load(str(decoder_path), map_location=device)
    )
    depth_decoder = depth_decoder.to(device).eval()
    logger.info("Depth decoder loaded.")

    return encoder, depth_decoder


def _run_monodepth2(
    image_bgr:     np.ndarray,
    encoder:       object,
    depth_decoder: object,
    device:        torch.device,
    orig_h:        int,
    orig_w:        int,
) -> np.ndarray:
    """
    Run Monodepth2 on a single BGR image.

    Resizes to MONO_WIDTH x MONO_HEIGHT for inference, then resizes the
    output disparity map back to (orig_h, orig_w) so that bounding box
    coordinates from Stage 3 align directly with depth map pixels.

    Returns
    -------
    np.ndarray float32, shape (orig_h, orig_w)
        Relative disparity map. Higher = closer to camera. Not metric.
    """
    input_img = cv2.resize(image_bgr, (MONO_WIDTH, MONO_HEIGHT))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(input_img).float() / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(tensor)
        outputs  = depth_decoder(features)

    disp = outputs[("disp", 0)].squeeze().cpu().numpy()   # (192, 640)
    disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return disp.astype(np.float32)


def _mask_from_geometry(
    x1: float, y1: float, x2: float, y2: float,
    compactness: float,
    img_h: int, img_w: int,
) -> np.ndarray:
    """
    Reconstruct an approximate binary mask from bounding box + compactness.

    Compactness (4π·area/perimeter²) encodes shape elongation:
      ≈ 1.0  → circle (pothole, manhole cover)
      ≈ 0.3  → irregular polygon (alligator crack)
      ≈ 0.05 → thin horizontal strip (crack, marking)

    The approximation maps compactness to the axes ratio of a filled ellipse:
      axes_ratio = sqrt(compactness / π) — derived from the compactness
      formula for an ellipse: C = 4π·πab / (π(a+b))² ≈ 4ab/(a+b)²
      For a circle (a=b): C = 1. For a:b → ∞: C → 0.

    The ellipse is drawn into a blank mask of the same size as the frame
    and then cropped to the bounding box. This gives a reasonable spatial
    prior for the mask-region depth extraction without requiring the
    original binary mask to be stored.

    This approximation is used only when geometry is available but the
    original mask array is not (which is always the case in the pipeline —
    masks are not serialised to JSON). When a more accurate mask is needed,
    Stage 3 should be modified to pass masks in memory rather than via JSON.

    Returns
    -------
    np.ndarray uint8, shape (img_h, img_w), values 0 or 255
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    bw = int((x2 - x1) / 2)
    bh = int((y2 - y1) / 2)

    if bw < 1 or bh < 1:
        return mask

    # Map compactness to axes ratio
    # Clamp compactness to (0, 1] to avoid sqrt of negative
    c = max(min(float(compactness), 1.0), 0.01)
    # For an ellipse: compactness ≈ 4ab/(a+b)²
    # Solve for ratio r = b/a given compactness c:
    # c·(1+r)² = 4r  →  c·r² + (2c-4)·r + c = 0
    discriminant = (2 * c - 4) ** 2 - 4 * c * c
    if discriminant < 0:
        axes_ratio = 1.0
    else:
        # Take the root closer to 1 (less extreme)
        r1 = (-(2 * c - 4) + math.sqrt(discriminant)) / (2 * c)
        r2 = (-(2 * c - 4) - math.sqrt(discriminant)) / (2 * c)
        # Pick the root in (0, 1]
        candidates = [r for r in (r1, r2) if 0 < r <= 1.0]
        axes_ratio = candidates[0] if candidates else 1.0

    # Semi-axes: longer axis along the direction the bbox is longer
    if bw >= bh:
        axis_x = bw
        axis_y = max(int(bh * axes_ratio), 1)
    else:
        axis_y = bh
        axis_x = max(int(bw * axes_ratio), 1)

    cv2.ellipse(
        mask,
        center=(cx, cy),
        axes=(axis_x, axis_y),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    return mask


def _extract_depth_mask_region(
    depth_map:  np.ndarray,
    mask:       np.ndarray,
    img_h:      int,
    img_w:      int,
    frame_min:  float,
    frame_max:  float,
) -> DepthEstimate:
    """Extract depth from valid mask pixels."""
    frame_range = frame_max - frame_min if frame_max > frame_min else 1.0
    pixels = depth_map[mask > 127]

    if pixels.size == 0:
        return None  # caller will fall back to central crop

    mean_d = float(pixels.mean())
    std_d  = float(pixels.std()) if pixels.size > 1 else 0.0

    # Coefficient of variation (std/mean), clipped to [0,1], inverted
    cv = (std_d / mean_d) if mean_d > 0 else 0.0
    cv_clipped   = min(cv, 1.0)
    depth_conf   = float(1.0 - cv_clipped)
    depth_norm   = float((mean_d - frame_min) / frame_range)

    return DepthEstimate(
        depth_raw         = round(mean_d,     6),
        depth_norm        = round(depth_norm,  4),
        depth_confidence  = round(depth_conf,  4),
        extraction_method = "mask_region",
        used_proxy        = False,
        frame_min_raw     = round(frame_min,   6),
        frame_max_raw     = round(frame_max,   6),
        region_px         = int(pixels.size),
    )


def _extract_depth_central_crop(
    depth_map:  np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    img_h:      int,
    img_w:      int,
    frame_min:  float,
    frame_max:  float,
    crop_frac:  float,
) -> DepthEstimate:
    """Extract depth from the central crop of the bounding box."""
    frame_range = frame_max - frame_min if frame_max > frame_min else 1.0

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) * crop_frac / 2.0
    hh = (y2 - y1) * crop_frac / 2.0

    cx1 = max(0,     int(cx - hw))
    cy1 = max(0,     int(cy - hh))
    cx2 = min(img_w, int(cx + hw))
    cy2 = min(img_h, int(cy + hh))

    crop = depth_map[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        # Degenerate box — single centre pixel
        px   = min(max(int(cx), 0), img_w - 1)
        py   = min(max(int(cy), 0), img_h - 1)
        mean_d = float(depth_map[py, px])
        std_d  = 0.0
        n_px   = 1
    else:
        mean_d = float(crop.mean())
        std_d  = float(crop.std()) if crop.size > 1 else 0.0
        n_px   = int(crop.size)

    cv          = (std_d / mean_d) if mean_d > 0 else 0.0
    depth_conf  = float(1.0 - min(cv, 1.0))
    depth_norm  = float((mean_d - frame_min) / frame_range)

    return DepthEstimate(
        depth_raw         = round(mean_d,     6),
        depth_norm        = round(depth_norm,  4),
        depth_confidence  = round(depth_conf,  4),
        extraction_method = "central_crop",
        used_proxy        = False,
        frame_min_raw     = round(frame_min,   6),
        frame_max_raw     = round(frame_max,   6),
        region_px         = n_px,
    )


def _geometry_proxy_depth(
    surface_area_px: int,
    mask_compactness: float,
    img_h: int,
    img_w: int,
) -> DepthEstimate:
    """
    Geometry-based depth proxy for low-confidence or low-light frames.

    Used when depth_confidence < DEPTH_CONF_THRESHOLD or lighting == "low_light".

    The proxy is derived from two SAM geometry features:
      - surface_area_px:  larger mask area → closer to camera (larger apparent size)
      - mask_compactness: lower compactness → crack (elongated, typically shallower)

    Proxy formula (heuristic, not metric):
      area_norm      = surface_area_px / (img_h * img_w)
      depth_proxy    = area_norm * (0.5 + 0.5 * mask_compactness)

    This is not a trained model — it is a transparent deterministic heuristic
    documented as such in the thesis (Chapter 3, Stage 4 fallback).
    Values are in [0, 1] by construction.

    depth_confidence is set to 0.0 to signal that this is a proxy,
    and used_proxy=True flags it for downstream stages.
    """
    frame_area = img_h * img_w
    area_norm  = min(surface_area_px / max(frame_area, 1), 1.0)
    proxy_val  = area_norm * (0.5 + 0.5 * mask_compactness)
    proxy_val  = float(min(max(proxy_val, 0.0), 1.0))

    return DepthEstimate(
        depth_raw         = proxy_val,
        depth_norm        = proxy_val,
        depth_confidence  = 0.0,
        extraction_method = "geometry_proxy",
        used_proxy        = True,
        frame_min_raw     = 0.0,
        frame_max_raw     = 1.0,
        region_px         = surface_area_px,
    )


# ---------------------------------------------------------------------------
# Debug / exact-mask helpers
# ---------------------------------------------------------------------------

def _reconstruct_mask(seg_box, img_h: int, img_w: int) -> Optional[np.ndarray]:
    """
    Rebuild a full-frame binary mask (uint8 0/255) from the tight crop that
    Stage 3 attached in memory (seg_box.mask + seg_box.mask_origin).

    Returns None when the exact mask is not available (e.g. a standalone
    Stage 4 run loaded from segmentations.json, where masks are not stored).
    """
    crop   = getattr(seg_box, "mask", None)
    origin = getattr(seg_box, "mask_origin", None)
    if crop is None or origin is None:
        return None
    full = np.zeros((img_h, img_w), dtype=np.uint8)
    y0, x0 = int(origin[0]), int(origin[1])
    h, w   = crop.shape[:2]
    y1, x1 = min(y0 + h, img_h), min(x0 + w, img_w)
    full[y0:y1, x0:x1] = crop[: y1 - y0, : x1 - x0]
    return full


def _colourise_disparity(depth_map: np.ndarray) -> np.ndarray:
    """
    Map a relative disparity array to a BGR colour image for inspection.

    Normalises with a 5th–95th percentile clip (same spirit as Monodepth2's
    test_simple.py, which clips vmax at the 95th percentile) so a few extreme
    pixels do not wash out the colour range. Higher disparity (closer to the
    camera) maps to the bright end of the colormap.
    """
    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    vmin = float(np.percentile(finite, 5))
    vmax = float(np.percentile(finite, 95))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    norm = np.clip((depth_map - vmin) / (vmax - vmin), 0.0, 1.0)
    u8   = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, _DEPTH_COLORMAP)


def _draw_depth_debug(
    depth_colour: np.ndarray,
    boxes: list,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Draw bounding boxes, the exact mask contour (when available), and the
    per-box depth_norm label on top of the colourised disparity image.

    Boxes here are DepthBox objects that still carry the Stage 3 mask handoff
    fields. White boxes and cyan contours read clearly on the magma colormap.
    """
    out = depth_colour.copy()
    for db in boxes:
        x1, y1, x2, y2 = int(db.x1), int(db.y1), int(db.x2), int(db.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

        mask = _reconstruct_mask(db, img_h, img_w)
        if mask is not None:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, (255, 255, 0), 1)

        if db.depth is not None:
            label = f"{db.class_name} d={db.depth.depth_norm:.2f}"
        else:
            label = db.class_name
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = max(y1 - 3, th + 3)
        cv2.rectangle(out, (x1, ly - th - base), (x1 + tw, ly + base), (0, 0, 0), -1)
        cv2.putText(
            out, label, (x1, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """
    Stage 4 — Monodepth2 relative depth estimation.

    Typical usage:
        cfg = DepthEstimatorConfig(
                  monodepth_root = r"C:\\Facultate\\pothole-detection\\Monodepth",
                  weights_dir    = r"ml\\weights\\mono_640x192",
                  device         = "cuda",
              )
        de      = DepthEstimator(cfg)
        results = de.run(segmentation_results,
                         output_dir="data/processed/depth/run/")
    """

    def __init__(self, cfg: DepthEstimatorConfig) -> None:
        self.cfg      = cfg
        self._encoder = None
        self._decoder = None
        self._device  = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load Monodepth2. Called once on first run()."""
        if self.cfg.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)

        logger.info(
            "DepthEstimator | device=%s | conf_threshold=%.2f | crop_fraction=%.2f",
            self._device, self.cfg.depth_conf_threshold, self.cfg.crop_fraction,
        )

        if self._device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                logger.info(
                    "GPU: %s  |  VRAM: %.1f GB",
                    props.name, props.total_memory / 1e9,
                )
            except Exception:
                pass

        self._encoder, self._decoder = _load_monodepth2(
            self.cfg.monodepth_root,
            self.cfg.weights_dir,
            self._device,
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        segmentation_results: list,    # List[SegmentationResult] from Stage 3
        output_dir: Optional[str] = None,
    ) -> List[DepthResult]:
        """
        Run Monodepth2 depth estimation on all frames with detections.

        Parameters
        ----------
        segmentation_results : list of SegmentationResult
            Output from Segmentor.run(). Frames without detections are
            forwarded as empty DepthResult objects.
        output_dir : str or None
            If provided, depth_estimates.json is saved here.

        Returns
        -------
        list of DepthResult
        """
        if self._encoder is None:
            self._load()

        debug_dir: Optional[Path] = None
        if output_dir and self.cfg.save_debug:
            debug_dir = Path(output_dir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Disparity debug images will be saved to: %s", debug_dir)

        results: List[DepthResult] = []
        t_start = time.perf_counter()

        n_frames_total     = len(segmentation_results)
        n_frames_with_det  = sum(1 for r in segmentation_results if r.has_detections)
        logger.info(
            "Starting depth estimation: %d frames, %d with detections",
            n_frames_total, n_frames_with_det,
        )

        n_mask_region  = 0
        n_central_crop = 0
        n_proxy        = 0
        n_failed       = 0
        n_low_light    = 0

        for seg_result in segmentation_results:

            depth_result = DepthResult(
                frame_path    = seg_result.frame_path,
                frame_index   = seg_result.frame_index,
                timestamp_s   = seg_result.timestamp_s,
                latitude      = seg_result.latitude,
                longitude     = seg_result.longitude,
                lighting      = seg_result.lighting,
                sun_elevation = seg_result.sun_elevation,
                image_width   = seg_result.image_width,
                image_height  = seg_result.image_height,
                depth_map_min = None,
                depth_map_max = None,
                depth_map_mean= None,
                boxes         = [],
            )

            # Forward frames with no detections unchanged
            if not seg_result.has_detections:
                results.append(depth_result)
                continue

            img_h = seg_result.image_height
            img_w = seg_result.image_width

            # ----------------------------------------------------------
            # Load image
            # ----------------------------------------------------------
            image_bgr = cv2.imread(seg_result.frame_path)
            if image_bgr is None:
                logger.warning(
                    "Cannot read image for frame %d: %s — depth skipped",
                    seg_result.frame_index, seg_result.frame_path,
                )
                for seg_box in seg_result.boxes:
                    depth_result.boxes.append(_seg_box_to_depth_box(seg_box, depth=None))
                results.append(depth_result)
                n_failed += 1
                continue

            # ----------------------------------------------------------
            # Low-light shortcut — skip Monodepth2, use geometry proxy
            # for all boxes in this frame
            # ----------------------------------------------------------
            if seg_result.lighting == "low_light":
                n_low_light += 1
                logger.debug(
                    "Frame %d: low_light — geometry proxy for all boxes",
                    seg_result.frame_index,
                )
                for seg_box in seg_result.boxes:
                    depth_est = _make_proxy_or_none(seg_box, img_h, img_w)
                    depth_result.boxes.append(
                        _seg_box_to_depth_box(seg_box, depth=depth_est)
                    )
                    if depth_est is not None and depth_est.used_proxy:
                        n_proxy += 1
                results.append(depth_result)
                continue

            # ----------------------------------------------------------
            # Run Monodepth2
            # ----------------------------------------------------------
            try:
                depth_map = _run_monodepth2(
                    image_bgr, self._encoder, self._decoder, self._device,
                    img_h, img_w,
                )
            except Exception:
                logger.exception(
                    "Monodepth2 failed on frame %d: %s",
                    seg_result.frame_index, seg_result.frame_path,
                )
                for seg_box in seg_result.boxes:
                    depth_result.boxes.append(_seg_box_to_depth_box(seg_box, depth=None))
                results.append(depth_result)
                n_failed += 1
                continue

            frame_min  = float(depth_map.min())
            frame_max  = float(depth_map.max())
            frame_mean = float(depth_map.mean())

            depth_result.depth_map_min  = round(frame_min,  6)
            depth_result.depth_map_max  = round(frame_max,  6)
            depth_result.depth_map_mean = round(frame_mean, 6)

            # ----------------------------------------------------------
            # Per-box depth extraction
            # ----------------------------------------------------------
            for seg_box in seg_result.boxes:

                depth_est: Optional[DepthEstimate] = None

                geom = seg_box.geometry
                use_mask = (
                    geom is not None
                    and not seg_box.low_sam_quality
                )

                if use_mask:
                    # Prefer the exact SAM mask when it was threaded through
                    # from Stage 3 and the operator opted in; otherwise fall
                    # back to the ellipse approximation (the validated default).
                    extract_mask = None
                    if self.cfg.use_exact_mask_depth:
                        extract_mask = _reconstruct_mask(seg_box, img_h, img_w)

                    if extract_mask is None:
                        extract_mask = _mask_from_geometry(
                            seg_box.x1, seg_box.y1, seg_box.x2, seg_box.y2,
                            compactness = geom.mask_compactness,
                            img_h       = img_h,
                            img_w       = img_w,
                        )
                    depth_est = _extract_depth_mask_region(
                        depth_map, extract_mask, img_h, img_w,
                        frame_min, frame_max,
                    )
                    if depth_est is not None:
                        n_mask_region += 1

                # Fallback 1: central crop
                if depth_est is None:
                    depth_est = _extract_depth_central_crop(
                        depth_map,
                        seg_box.x1, seg_box.y1, seg_box.x2, seg_box.y2,
                        img_h, img_w, frame_min, frame_max,
                        self.cfg.crop_fraction,
                    )
                    n_central_crop += 1

                # Fallback 2: geometry proxy when confidence is too low
                if depth_est.depth_confidence < self.cfg.depth_conf_threshold:
                    proxy = _make_proxy_or_none(seg_box, img_h, img_w)
                    if proxy is not None:
                        logger.debug(
                            "Frame %d box %s: depth_conf=%.3f < %.2f → geometry proxy",
                            seg_result.frame_index,
                            seg_box.class_name,
                            depth_est.depth_confidence,
                            self.cfg.depth_conf_threshold,
                        )
                        depth_est = proxy
                        n_proxy  += 1
                        # Subtract the central_crop count we just added above
                        if not use_mask:
                            n_central_crop -= 1

                depth_result.boxes.append(
                    _seg_box_to_depth_box(seg_box, depth=depth_est)
                )

            # Disparity debug image — one per detection frame.
            # Only produced here, where a real Monodepth2 map exists (low-light
            # and failed frames take the proxy/None paths above and are skipped).
            if debug_dir is not None and depth_result.boxes:
                depth_colour = _colourise_disparity(depth_map)
                annotated    = _draw_depth_debug(
                    depth_colour, depth_result.boxes, img_h, img_w
                )
                stem = Path(seg_result.frame_path).stem
                cv2.imwrite(str(debug_dir / f"{stem}.jpg"), annotated)

            results.append(depth_result)

            logger.info(
                "Frame %-40s | %d box(es) | depth [%.3f, %.3f]",
                Path(seg_result.frame_path).stem,
                seg_result.n_detections,
                frame_min,
                frame_max,
            )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t_start

        logger.info("=== Depth estimation complete ===")
        logger.info("  Frames processed        : %d", n_frames_total)
        logger.info("  Frames failed           : %d", n_failed)
        logger.info("  Frames low-light        : %d", n_low_light)
        logger.info("  Boxes — mask_region     : %d", n_mask_region)
        logger.info("  Boxes — central_crop    : %d", n_central_crop)
        logger.info("  Boxes — geometry_proxy  : %d", n_proxy)
        logger.info("  Elapsed                 : %.1f s", elapsed)

        if output_dir:
            self.save_depth_estimates(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_depth_estimates(
        results:    List[DepthResult],
        output_dir: str,
    ) -> str:
        """Save depth estimation results to depth_estimates.json."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "depth_estimates.json"

        n_boxes = sum(r.n_detections for r in results)
        n_with_depth = sum(
            1 for r in results for b in r.boxes if b.depth is not None
        )
        payload = {
            "model":             "mono_640x192",
            "model_reference":   "Godard et al., 2019. arXiv:1806.01260",
            "n_frames":          len(results),
            "n_boxes_total":     n_boxes,
            "n_boxes_with_depth":n_with_depth,
            "frames":            [r.to_dict() for r in results],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Depth estimates saved: %s  (%d frames, %d boxes, %d with depth)",
            out_path, len(results), n_boxes, n_with_depth,
        )
        return str(out_path)

    @staticmethod
    def load_depth_estimates(path: str) -> List[DepthResult]:
        """Load depth_estimates.json back into DepthResult objects."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results: List[DepthResult] = []
        for fr in payload["frames"]:
            boxes = []
            for b in fr["boxes"]:
                d = b.get("depth")
                depth_est = (
                    DepthEstimate(
                        depth_raw         = d["depth_raw"],
                        depth_norm        = d["depth_norm"],
                        depth_confidence  = d["depth_confidence"],
                        extraction_method = d["extraction_method"],
                        used_proxy        = d["used_proxy"],
                        frame_min_raw     = d["frame_min_raw"],
                        frame_max_raw     = d["frame_max_raw"],
                        region_px         = d["region_px"],
                    )
                    if d is not None else None
                )
                geom_d = b.get("geometry")
                from pipeline.segmentor import MaskGeometry
                geometry = (
                    MaskGeometry(
                        surface_area_px   = geom_d["surface_area_px"],
                        edge_sharpness    = geom_d["edge_sharpness"],
                        interior_contrast = geom_d["interior_contrast"],
                        mask_compactness  = geom_d["mask_compactness"],
                    )
                    if geom_d is not None else None
                )
                boxes.append(DepthBox(
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
                    depth             = depth_est,
                ))
            results.append(DepthResult(
                frame_path    = fr["frame_path"],
                frame_index   = fr["frame_index"],
                timestamp_s   = fr["timestamp_s"],
                latitude      = fr.get("latitude"),
                longitude     = fr.get("longitude"),
                lighting      = fr.get("lighting", "unknown"),
                sun_elevation = fr.get("sun_elevation"),
                image_width   = fr["image_width"],
                image_height  = fr["image_height"],
                depth_map_min = fr.get("depth_map_min"),
                depth_map_max = fr.get("depth_map_max"),
                depth_map_mean= fr.get("depth_map_mean"),
                boxes         = boxes,
            ))

        logger.info("Depth estimates loaded: %s (%d frames)", path, len(results))
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seg_box_to_depth_box(seg_box, depth: Optional[DepthEstimate]) -> DepthBox:
    """Wrap a SegmentedBox into a DepthBox, adding the depth field."""
    return DepthBox(
        x1                = seg_box.x1,
        y1                = seg_box.y1,
        x2                = seg_box.x2,
        y2                = seg_box.y2,
        class_id          = seg_box.class_id,
        class_name        = seg_box.class_name,
        confidence        = seg_box.confidence,
        threshold_applied = seg_box.threshold_applied,
        severity_prior    = seg_box.severity_prior,
        is_damage         = seg_box.is_damage,
        is_infrastructure = seg_box.is_infrastructure,
        is_marking        = seg_box.is_marking,
        sam_score         = seg_box.sam_score,
        geometry          = seg_box.geometry,
        low_sam_quality   = seg_box.low_sam_quality,
        depth             = depth,
        mask              = getattr(seg_box, "mask", None),
        mask_origin       = getattr(seg_box, "mask_origin", None),
    )


def _make_proxy_or_none(
    seg_box,
    img_h: int,
    img_w: int,
) -> Optional[DepthEstimate]:
    """
    Return a geometry proxy DepthEstimate if geometry is available,
    otherwise return None.
    """
    geom = seg_box.geometry
    if geom is None:
        return None
    return _geometry_proxy_depth(
        surface_area_px  = geom.surface_area_px,
        mask_compactness = geom.mask_compactness,
        img_h            = img_h,
        img_w            = img_w,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 -- Monodepth2 relative depth estimation."
    )
    parser.add_argument(
        "--segmentations", required=True,
        help="segmentations.json produced by Stage 3 (Segmentor)",
    )
    parser.add_argument(
        "--monodepth_root",
        default=r"C:\Facultate\pothole-detection\Monodepth",
        help="Path to cloned Monodepth2 repository",
    )
    parser.add_argument(
        "--weights_dir",
        default=r"C:\Facultate\pothole-detection\Pothole-Detection\ml\weights\mono_640x192",
        help="Directory containing encoder.pth and depth.pth",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for depth_estimates.json",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0  (default: auto)",
    )
    parser.add_argument(
        "--depth_conf_threshold", type=float, default=DEPTH_CONF_THRESHOLD,
        help=f"Confidence threshold for geometry proxy fallback (default: {DEPTH_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--save_debug", action="store_true",
        help="Save colourised disparity images to output/debug/ for detection frames",
    )
    parser.add_argument(
        "--exact_mask_depth", action="store_true",
        help="Use the exact SAM mask (when available) for depth extraction "
             "instead of the ellipse approximation. Changes depth_norm / "
             "depth_confidence vs the validated default; off by default.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from pipeline.segmentor import Segmentor
    seg_results = Segmentor.load_segmentations(args.segmentations)
    logger.info(
        "Loaded %d segmentation results from: %s",
        len(seg_results), args.segmentations,
    )

    cfg = DepthEstimatorConfig(
        monodepth_root       = args.monodepth_root,
        weights_dir          = args.weights_dir,
        device               = args.device,
        depth_conf_threshold = args.depth_conf_threshold,
        save_debug           = args.save_debug,
        use_exact_mask_depth = args.exact_mask_depth,
    )
    de = DepthEstimator(cfg)
    de.run(seg_results, output_dir=args.output)


if __name__ == "__main__":
    main()