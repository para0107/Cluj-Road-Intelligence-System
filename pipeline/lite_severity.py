"""
pipeline/lite_severity.py

Severity scoring for the LITE (per-user, real-time) pipeline.

The full survey pipeline derives four signals per detection:
    depth      — Monodepth2 (stage 4)                    ~2.5 GB VRAM, ~40 ms
    area       — SAM 2.1 mask pixel count (stage 3)      ~0.9 GB VRAM, ~90 ms
    contrast   — SAM mask interior vs ring gray delta
    sharpness  — Sobel magnitude on the SAM mask boundary

This module reproduces the SAME four signals with classical CV on the
detector's bounding-box crop — no SAM, no Monodepth2, no extra VRAM,
~1 ms per box on CPU:

    mask proxy — Otsu threshold on the gray crop (road damage is darker
                 than intact asphalt); falls back to the full box when the
                 threshold degenerates
    area       — proxy-mask pixel count             (same semantic as SAM's)
    sharpness  — mean Sobel magnitude on the proxy-mask boundary
                 (identical ops to segmentor._compute_geometry: 5×5 dilate)
    contrast   — |mean(gray inside eroded mask) − mean(gray in ring)|
                 (identical ops: 2× erode vs dilate ring)
    depth      — the full pipeline's own geometry fallback, verbatim:
                 area_norm · (0.5 + 0.5 · compactness)
                 (depth_estimator._geometry_proxy_depth — already a documented
                 thesis heuristic for frames where Monodepth2 is unreliable)

The signals then flow through the UNMODIFIED stage-5 formula
(severity_classifier.classify_box): same per-class weights, same S1–S5
bands, same marking-class caps. severity_confidence is computed with
used_proxy=True, so lite scores honestly report ~0.5 confidence — the full
pipeline remains the authoritative audit path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.severity_classifier import SeverityConfig, classify_box

_SEVERITY_CFG = SeverityConfig()
_LEVEL_TO_INT = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}

# Same structuring element as segmentor._compute_geometry
_KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


@dataclass
class LiteSeverity:
    """Result of the lite severity assessment for one detection box."""
    severity: int                 # 1..5
    severity_level: str           # "S1".."S5"
    severity_score: float         # [0, 1]
    severity_confidence: float    # ~0.5 — proxy-based by design
    surface_area_px: int
    edge_sharpness: float
    interior_contrast: float
    mask_compactness: float
    depth_proxy: float


def _proxy_mask(gray_crop: np.ndarray) -> np.ndarray:
    """
    Cheap damage-mask stand-in: Otsu-threshold the crop and keep the DARKER
    side (cracks/potholes read darker than intact asphalt). Falls back to the
    whole box when Otsu degenerates (uniform texture, over/under exposure).
    """
    _, binary = cv2.threshold(gray_crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    frac = float(binary.mean())
    if frac < 0.02 or frac > 0.90:
        return np.ones_like(gray_crop, dtype=np.uint8)
    return binary.astype(np.uint8)


def assess_box(
    frame_bgr: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    class_name: str,
) -> LiteSeverity:
    """
    Score one detection on the ORIGINAL frame. All heavy semantics reuse the
    survey pipeline's stage-5 formula; only the signal extractors are proxies.
    """
    img_h, img_w = frame_bgr.shape[:2]

    # Clamp + pad the crop a little so boundary ops have context
    pad = 4
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(img_w, int(x2) + pad)
    cy2 = min(img_h, int(y2) + pad)
    if cx2 - cx1 < 6 or cy2 - cy1 < 6:
        # Degenerate box — score conservatively from nothing (stage-5 treats
        # missing signals as 0.0 and never over-estimates)
        est = classify_box(
            class_name=class_name, depth_norm=None, surface_area_px=None,
            interior_contrast=None, edge_sharpness=None,
            used_proxy=True, depth_confidence=0.0, cfg=_SEVERITY_CFG,
        )
        return LiteSeverity(
            severity=_LEVEL_TO_INT[est.severity_level],
            severity_level=est.severity_level,
            severity_score=est.severity_score,
            severity_confidence=est.severity_confidence,
            surface_area_px=0, edge_sharpness=0.0,
            interior_contrast=0.0, mask_compactness=0.0, depth_proxy=0.0,
        )

    gray = cv2.cvtColor(frame_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
    binary = _proxy_mask(gray)

    # --- area (same semantic as segmentor: damage-mask pixel count) ---------
    surface_area_px = int(binary.sum())

    # --- sharpness: mean Sobel magnitude on the mask boundary ---------------
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    dilated = cv2.dilate(binary, _KERNEL_5, iterations=1)
    boundary = dilated - binary
    edge_sharpness = float(sobel_mag[boundary > 0].mean()) if boundary.any() else 0.0

    # --- contrast: eroded interior vs dilation ring -------------------------
    eroded = cv2.erode(binary, _KERNEL_5, iterations=2)
    ring = dilated - binary
    if eroded.any() and ring.any():
        interior_contrast = abs(float(gray[eroded > 0].mean()) - float(gray[ring > 0].mean()))
    else:
        interior_contrast = 0.0

    # --- compactness: 4πA / P² (same formula as segmentor) ------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    mask_compactness = (
        float(min(4.0 * math.pi * surface_area_px / (perimeter ** 2), 1.0))
        if perimeter > 0 else 0.0
    )

    # --- depth: the full pipeline's geometry fallback, verbatim -------------
    # (depth_estimator._geometry_proxy_depth: area over FULL frame area)
    area_norm = min(surface_area_px / max(img_h * img_w, 1), 1.0)
    depth_proxy = float(min(max(area_norm * (0.5 + 0.5 * mask_compactness), 0.0), 1.0))

    # --- unmodified stage-5 formula ------------------------------------------
    est = classify_box(
        class_name=class_name,
        depth_norm=depth_proxy,
        surface_area_px=surface_area_px,
        interior_contrast=interior_contrast,
        edge_sharpness=edge_sharpness,
        used_proxy=True,
        depth_confidence=0.0,
        cfg=_SEVERITY_CFG,
    )

    return LiteSeverity(
        severity=_LEVEL_TO_INT[est.severity_level],
        severity_level=est.severity_level,
        severity_score=est.severity_score,
        severity_confidence=est.severity_confidence,
        surface_area_px=surface_area_px,
        edge_sharpness=round(edge_sharpness, 4),
        interior_contrast=round(interior_contrast, 4),
        mask_compactness=round(mask_compactness, 4),
        depth_proxy=round(depth_proxy, 4),
    )
