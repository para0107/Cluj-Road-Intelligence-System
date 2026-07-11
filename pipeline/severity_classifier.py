"""
pipeline/severity_classifier.py
---------------------------------
Stage 5 of the road damage detection inference pipeline.

Responsibilities:
  - Accept a list of DepthResult objects from Stage 4 (DepthEstimator)
  - For every accepted bounding box in every frame, compute:
      1. Four normalised signal scores from available measurements
      2. A weighted raw score using per-class signal weights
      3. A severity_score in [0, 1] incorporating class importance weight
      4. A severity level S1–S5 mapped from the severity_score
      5. A severity_confidence reflecting measurement quality
  - Return a list of SeverityResult objects, one per frame

Severity formula
----------------
Step 1 — Signal normalisation (all to [0, 1]):
    S_depth     = depth_norm
    S_area      = min(surface_area_px / 1000, 1.0)
    S_contrast  = min(interior_contrast / 2.0, 1.0)
    S_sharpness = min(edge_sharpness / 60.0, 1.0)

    Denominators recalibrated after first validation run (severity_summary.json
    showed only 4/1919 detections at S4/S5 with original values):
    - 1000 px: ~p25 of structural damage area range. Pothole mean=327px gives
               S_area=0.327 (meaningful); alligator crack 5405px capped at 1.0.
               Original 5000px produced S_area=0.065 for potholes (near-zero).
    - 2.0:     midpoint of contrast range. Preserves differentiation between
               low-contrast cracks and high-contrast crossings/potholes.
               Original 3.0 squashed damage classes to <0.10.
    - 60.0:    approximate sharpness of rutting (58.28), the sharpest structural
               class. All damage classes now produce meaningful sharpness signals.
               Original 100.0 compressed all signals to <0.50.

Step 2 — Per-class weighted combination:
    raw_score = w_depth*S_depth + w_area*S_area
              + w_contrast*S_contrast + w_sharpness*S_sharpness

    Signal weights are class-specific (see SIGNAL_WEIGHTS) and sum to 1.0.
    Rationale: pothole severity is dominated by depth + contrast (bowl shape);
    alligator crack by area (spread of fatigue network); longitudinal crack
    by sharpness (active vs healed boundary).

Step 3 — Class importance weighting:
    severity_score = min(raw_score * class_weight * 2.0, 1.0)

    class_weight encodes how structurally critical the class is to fix.
    Pothole = 0.50 (highest hazard). Lane line blur = 0.10 (marking only).
    The ×2 rescaling ensures pothole with raw_score=1.0 → severity_score=1.0,
    while lane_line_blur with raw_score=1.0 → severity_score=0.20 (≤ S2).

Step 4 — Mapping to S1–S5:
    [0.00, 0.15) → S1  Monitor
    [0.15, 0.35) → S2  Schedule maintenance
    [0.35, 0.55) → S3  Priority repair
    [0.55, 0.75) → S4  Urgent repair
    [0.75, 1.00] → S5  Emergency closure

Design decisions
----------------
- No hardcoded severity overrides: marking classes (lane_line_blur,
  pedestrian_crossing_blur) cannot exceed S2 by construction of the formula
  (class_weight=0.10, max severity_score=0.20). No special-case if-branches
  are needed — the formula handles it.
- Repaired cracks: class_weight=0.15 → max severity_score=0.30 → S2 ceiling.
  A repaired crack in poor condition will approach S2 but never reach S3.
- Missing signals default to 0.0 (conservative). This under-estimates severity
  for detections without geometry (SAM failed) but never over-estimates.
- severity_confidence penalises proxy depth: when Monodepth2 confidence was
  below the 0.4 threshold and the geometry proxy was substituted, the severity
  confidence is reduced proportionally.

References
----------
  Godard et al., 2019. arXiv:1806.01260  (Monodepth2 — depth signal)
  Kirillov et al., 2023. arXiv:2304.02643 (SAM — geometry signals)
  Zhang et al., 2024. doi:10.1016/j.ijtst.2024.10.005 (SAM road asset validation)

Usage (module):
    from pipeline.depth_estimator     import DepthEstimator, DepthEstimatorConfig
    from pipeline.severity_classifier import SeverityClassifier, SeverityConfig

    depth_results    = DepthEstimator(DepthEstimatorConfig(...)).run(seg_results)
    sev_cfg          = SeverityConfig()
    classifier       = SeverityClassifier(sev_cfg)
    severity_results = classifier.run(depth_results)

Usage (CLI):
    python pipeline/severity_classifier.py
        --depth_estimates  data/processed/depth/run/depth_estimates.json
        --output           data/processed/severity/run/
        [--verbose]

Author: Paraschiv Tudor, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formula constants
# ---------------------------------------------------------------------------

AREA_NORM_DENOM      = 1000.0   # px -- recalibrated: ~p25 of structural damage area range
                                 # pothole (327px) → S_area=0.327; alligator (5405px) → capped 1.0
                                 # Original 5000.0 produced S_area≈0.065 for potholes (near-zero)
CONTRAST_NORM_DENOM  = 2.0      # -- midpoint of contrast range; preserves damage differentiation
                                 # Original 3.0 squashed low-contrast damage classes to <0.10
SHARPNESS_NORM_DENOM = 60.0     # -- approximate mean sharpness of rutting (58.28), the sharpest
                                 # structural damage class; original 100.0 compressed all signals

# Per-class signal weights [w_depth, w_area, w_contrast, w_sharpness]
# Each tuple sums to 1.0.
SIGNAL_WEIGHTS: Dict[str, tuple] = {
    "pothole":                   (0.35, 0.25, 0.30, 0.10),
    "alligator_crack":           (0.25, 0.40, 0.20, 0.15),
    "longitudinal_crack":        (0.30, 0.25, 0.15, 0.30),
    "transverse_crack":          (0.30, 0.20, 0.15, 0.35),
    "repaired_crack":            (0.20, 0.30, 0.10, 0.40),
    "patchy_road":               (0.25, 0.40, 0.15, 0.20),
    "rutting":                   (0.40, 0.25, 0.10, 0.25),
    "manhole_cover":             (0.30, 0.20, 0.25, 0.25),
    "pedestrian_crossing_blur":  (0.10, 0.50, 0.30, 0.10),
    "lane_line_blur":            (0.10, 0.50, 0.30, 0.10),
}

CLASS_WEIGHTS: Dict[str, float] = {
    "pothole":                   0.50,
    "alligator_crack":           0.45,
    "rutting":                   0.40,
    "transverse_crack":          0.35,
    "longitudinal_crack":        0.30,
    "patchy_road":               0.25,
    "manhole_cover":             0.20,
    "repaired_crack":            0.15,
    "pedestrian_crossing_blur":  0.10,
    "lane_line_blur":            0.10,
}

# [lo, hi) intervals → (level, action)
SEVERITY_INTERVALS = [
    (0.00, 0.15, "S1", "Monitor"),
    (0.15, 0.35, "S2", "Schedule maintenance"),
    (0.35, 0.55, "S3", "Priority repair"),
    (0.55, 0.75, "S4", "Urgent repair"),
    (0.75, 1.01, "S5", "Emergency closure"),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SeverityConfig:
    """
    Configuration for Stage 5 — severity classifier.

    All formula constants are exposed here so they can be recalibrated
    without modifying the classifier logic.

    area_norm_denom:
        Denominator for surface_area_px normalisation (pixels).
        Default 1000 — approximately the p25 of structural damage area range.
        Recalibrated from 5000 (which produced near-zero signals for potholes).

    contrast_norm_denom:
        Denominator for interior_contrast normalisation.
        Default 2.0 — midpoint of observed contrast range.
        Recalibrated from 3.0 (which squashed damage class contrasts to <0.10).

    sharpness_norm_denom:
        Denominator for edge_sharpness normalisation.
        Default 60.0 — approximate sharpness of rutting (58.28), the sharpest
        structural damage class. Recalibrated from 100.0.

    signal_weights:
        Per-class signal weight tuples. Can be replaced with a custom dict
        for recalibration without changing any other code.

    class_weights:
        Per-class importance weights. Can be replaced for recalibration.
    """
    area_norm_denom:      float = AREA_NORM_DENOM
    contrast_norm_denom:  float = CONTRAST_NORM_DENOM
    sharpness_norm_denom: float = SHARPNESS_NORM_DENOM
    signal_weights:       Dict[str, tuple] = field(
        default_factory=lambda: dict(SIGNAL_WEIGHTS)
    )
    class_weights:        Dict[str, float] = field(
        default_factory=lambda: dict(CLASS_WEIGHTS)
    )


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SeverityEstimate:
    """
    Full severity assessment for one bounding box detection.

    All intermediate values are stored for auditability and downstream
    use by the enricher, deduplicator, and priority scorer.

    signal_scores:
        Dict with keys S_depth, S_area, S_contrast, S_sharpness,
        each normalised to [0, 1].

    raw_score:
        Weighted combination of signal scores using per-class weights.
        Range [0, 1].

    class_weight:
        Class importance multiplier applied in Step 3.

    severity_score:
        Final score in [0, 1]. Feeds into S1–S5 mapping and
        priority_score formula (database schema).

    severity_level:
        "S1" | "S2" | "S3" | "S4" | "S5"

    severity_action:
        Human-readable action string for the assigned level.

    severity_confidence:
        Reliability of the severity estimate [0, 1].
        Reduced when depth proxy was used (geometry proxy substituted
        Monodepth2 because depth_confidence < 0.4 or lighting = low_light).
    """
    signal_scores:       Dict[str, float]
    signal_weights_used: Dict[str, float]
    raw_score:           float
    class_weight:        float
    severity_score:      float
    severity_level:      str
    severity_action:     str
    severity_confidence: float

    def to_dict(self) -> dict:
        return {
            "signal_scores":       self.signal_scores,
            "signal_weights_used": self.signal_weights_used,
            "raw_score":           self.raw_score,
            "class_weight":        self.class_weight,
            "severity_score":      self.severity_score,
            "severity_level":      self.severity_level,
            "severity_action":     self.severity_action,
            "severity_confidence": self.severity_confidence,
        }


@dataclass
class SeverityBox:
    """
    One bounding box from Stage 4, enriched with severity information.
    All Stage 4 fields are preserved unchanged.
    """
    # --- forwarded from DepthBox ---
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
    depth:             Optional[object]   # DepthEstimate or None
    # --- added by Stage 5 ---
    severity:          Optional[SeverityEstimate]

    def to_dict(self) -> dict:
        from pipeline.segmentor import MaskGeometry
        from pipeline.depth_estimator import DepthEstimate

        geom = self.geometry
        dep  = self.depth
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
            "sam_score":         round(self.sam_score, 4)
                                 if self.sam_score is not None else None,
            "low_sam_quality":   self.low_sam_quality,
            "geometry":          geom.to_dict() if geom is not None else None,
            "depth":             dep.to_dict()  if dep  is not None else None,
            "severity":          self.severity.to_dict()
                                 if self.severity is not None else None,
        }


@dataclass
class SeverityResult:
    """Stage 5 output for one frame."""
    frame_path:    str
    frame_index:   int
    timestamp_s:   float
    latitude:      Optional[float]
    longitude:     Optional[float]
    lighting:      str
    sun_elevation: Optional[float]
    image_width:   int
    image_height:  int
    depth_map_min: Optional[float]
    depth_map_max: Optional[float]
    depth_map_mean: Optional[float]
    boxes:         List[SeverityBox] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.boxes) > 0

    @property
    def n_detections(self) -> int:
        return len(self.boxes)

    @property
    def highest_severity(self) -> Optional[str]:
        """Return the highest severity level among all boxes in this frame."""
        order = {"S5": 5, "S4": 4, "S3": 3, "S2": 2, "S1": 1}
        levels = [
            b.severity.severity_level
            for b in self.boxes
            if b.severity is not None
        ]
        if not levels:
            return None
        return max(levels, key=lambda x: order.get(x, 0))

    def to_dict(self) -> dict:
        return {
            "frame_path":     self.frame_path,
            "frame_index":    self.frame_index,
            "timestamp_s":    self.timestamp_s,
            "latitude":       self.latitude,
            "longitude":      self.longitude,
            "lighting":       self.lighting,
            "sun_elevation":  self.sun_elevation,
            "image_width":    self.image_width,
            "image_height":   self.image_height,
            "depth_map_min":  self.depth_map_min,
            "depth_map_max":  self.depth_map_max,
            "depth_map_mean": self.depth_map_mean,
            "highest_severity": self.highest_severity,
            "boxes":          [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Core formula functions (module-level — used by validate_severity.py too)
# ---------------------------------------------------------------------------

def compute_signal_scores(
    depth_norm:        Optional[float],
    surface_area_px:   Optional[int],
    interior_contrast: Optional[float],
    edge_sharpness:    Optional[float],
    cfg:               SeverityConfig,
) -> Dict[str, float]:
    """Normalise raw signal values to [0, 1]. Missing values → 0.0."""
    s_depth     = float(depth_norm)        if depth_norm        is not None else 0.0
    s_area      = float(surface_area_px)   if surface_area_px   is not None else 0.0
    s_contrast  = float(interior_contrast) if interior_contrast  is not None else 0.0
    s_sharpness = float(edge_sharpness)    if edge_sharpness    is not None else 0.0

    return {
        "S_depth":     round(min(s_depth,                                    1.0), 4),
        "S_area":      round(min(s_area      / cfg.area_norm_denom,          1.0), 4),
        "S_contrast":  round(min(s_contrast  / cfg.contrast_norm_denom,      1.0), 4),
        "S_sharpness": round(min(s_sharpness / cfg.sharpness_norm_denom,     1.0), 4),
    }


def compute_raw_score(
    signal_scores: Dict[str, float],
    class_name:    str,
    cfg:           SeverityConfig,
) -> tuple[float, Dict[str, float]]:
    """
    Weighted combination using per-class signal weights.
    Returns (raw_score, weights_used_dict).
    """
    weights = cfg.signal_weights.get(class_name, (0.25, 0.25, 0.25, 0.25))
    w_depth, w_area, w_contrast, w_sharpness = weights

    raw = (
        w_depth     * signal_scores["S_depth"]     +
        w_area      * signal_scores["S_area"]      +
        w_contrast  * signal_scores["S_contrast"]  +
        w_sharpness * signal_scores["S_sharpness"]
    )

    weights_dict = {
        "w_depth":     w_depth,
        "w_area":      w_area,
        "w_contrast":  w_contrast,
        "w_sharpness": w_sharpness,
    }
    return round(raw, 4), weights_dict


def compute_severity_score(raw_score: float, class_name: str, cfg: SeverityConfig) -> float:
    """Apply class importance weight and rescale to [0, 1]."""
    class_weight = cfg.class_weights.get(class_name, 0.25)
    return round(min(raw_score * class_weight * 2.0, 1.0), 4)


def score_to_level(severity_score: float) -> tuple[str, str]:
    """Map severity_score to (level, action) using SEVERITY_INTERVALS."""
    for lo, hi, level, action in SEVERITY_INTERVALS:
        if lo <= severity_score < hi:
            return level, action
    return "S5", "Emergency closure"


def compute_severity_confidence(
    used_proxy:       bool,
    depth_confidence: Optional[float],
) -> float:
    """
    Estimate reliability of the severity assessment.

    Full confidence (≥0.85) when all signals come from real measurements
    and depth_confidence is high.
    Reduced confidence when the geometry proxy was substituted for Monodepth2
    (depth_confidence < 0.4 or low_light frame).
    """
    if used_proxy:
        # Geometry proxy was used — depth signal is heuristic
        base = 0.5
        bonus = 0.5 * (depth_confidence or 0.0)
        return round(base + bonus, 4)
    else:
        # Real Monodepth2 depth — confidence driven by depth_confidence
        base = 0.6
        bonus = 0.4 * (depth_confidence if depth_confidence is not None else 1.0)
        return round(min(base + bonus, 1.0), 4)


def classify_box(
    class_name:        str,
    depth_norm:        Optional[float],
    surface_area_px:   Optional[int],
    interior_contrast: Optional[float],
    edge_sharpness:    Optional[float],
    used_proxy:        bool,
    depth_confidence:  Optional[float],
    cfg:               SeverityConfig,
) -> SeverityEstimate:
    """Classify one detection. All formula steps documented in module docstring."""
    signal_scores          = compute_signal_scores(
        depth_norm, surface_area_px, interior_contrast, edge_sharpness, cfg
    )
    raw_score, weights_dict = compute_raw_score(signal_scores, class_name, cfg)
    severity_score          = compute_severity_score(raw_score, class_name, cfg)
    level, action           = score_to_level(severity_score)
    sev_confidence          = compute_severity_confidence(used_proxy, depth_confidence)

    return SeverityEstimate(
        signal_scores       = signal_scores,
        signal_weights_used = weights_dict,
        raw_score           = raw_score,
        class_weight        = cfg.class_weights.get(class_name, 0.25),
        severity_score      = severity_score,
        severity_level      = level,
        severity_action     = action,
        severity_confidence = sev_confidence,
    )


# ---------------------------------------------------------------------------
# SeverityClassifier
# ---------------------------------------------------------------------------

class SeverityClassifier:
    """
    Stage 5 — Rule-based weighted severity classification.

    Typical usage:
        cfg        = SeverityConfig()
        classifier = SeverityClassifier(cfg)
        results    = classifier.run(depth_results, output_dir="data/processed/severity/run/")
    """

    def __init__(self, cfg: SeverityConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        depth_results: list,              # List[DepthResult] from Stage 4
        output_dir: Optional[str] = None,
    ) -> List[SeverityResult]:
        """
        Run severity classification on all frames.

        Parameters
        ----------
        depth_results : list of DepthResult
            Output from DepthEstimator.run(). Frames without detections
            are forwarded as empty SeverityResult objects.
        output_dir : str or None
            If provided, severity_estimates.json is saved here.

        Returns
        -------
        list of SeverityResult
        """
        results: List[SeverityResult] = []
        t_start = time.perf_counter()

        n_total   = len(depth_results)
        n_boxes   = sum(r.n_detections for r in depth_results)

        logger.info(
            "Starting severity classification: %d frames, %d boxes",
            n_total, n_boxes,
        )

        level_counts: Dict[str, int] = {}

        for depth_result in depth_results:

            sev_result = SeverityResult(
                frame_path    = depth_result.frame_path,
                frame_index   = depth_result.frame_index,
                timestamp_s   = depth_result.timestamp_s,
                latitude      = depth_result.latitude,
                longitude     = depth_result.longitude,
                lighting      = depth_result.lighting,
                sun_elevation = depth_result.sun_elevation,
                image_width   = depth_result.image_width,
                image_height  = depth_result.image_height,
                depth_map_min = depth_result.depth_map_min,
                depth_map_max = depth_result.depth_map_max,
                depth_map_mean= depth_result.depth_map_mean,
                boxes         = [],
            )

            if not depth_result.has_detections:
                results.append(sev_result)
                continue

            for depth_box in depth_result.boxes:
                # Extract signals from DepthBox
                geom = depth_box.geometry
                dep  = depth_box.depth

                depth_norm        = dep.depth_norm        if dep  is not None else None
                depth_confidence  = dep.depth_confidence  if dep  is not None else None
                used_proxy        = dep.used_proxy         if dep  is not None else False
                surface_area_px   = geom.surface_area_px  if geom is not None else None
                interior_contrast = geom.interior_contrast if geom is not None else None
                edge_sharpness    = geom.edge_sharpness   if geom is not None else None

                severity = classify_box(
                    class_name        = depth_box.class_name,
                    depth_norm        = depth_norm,
                    surface_area_px   = surface_area_px,
                    interior_contrast = interior_contrast,
                    edge_sharpness    = edge_sharpness,
                    used_proxy        = used_proxy,
                    depth_confidence  = depth_confidence,
                    cfg               = self.cfg,
                )

                level_counts[severity.severity_level] = (
                    level_counts.get(severity.severity_level, 0) + 1
                )

                sev_result.boxes.append(SeverityBox(
                    x1                = depth_box.x1,
                    y1                = depth_box.y1,
                    x2                = depth_box.x2,
                    y2                = depth_box.y2,
                    class_id          = depth_box.class_id,
                    class_name        = depth_box.class_name,
                    confidence        = depth_box.confidence,
                    threshold_applied = depth_box.threshold_applied,
                    severity_prior    = depth_box.severity_prior,
                    is_damage         = depth_box.is_damage,
                    is_infrastructure = depth_box.is_infrastructure,
                    is_marking        = depth_box.is_marking,
                    sam_score         = depth_box.sam_score,
                    geometry          = depth_box.geometry,
                    low_sam_quality   = depth_box.low_sam_quality,
                    depth             = depth_box.depth,
                    severity          = severity,
                ))

                logger.debug(
                    "Frame %d | %-30s | score=%.3f | level=%s",
                    depth_result.frame_index,
                    depth_box.class_name,
                    severity.severity_score,
                    severity.severity_level,
                )

            results.append(sev_result)

        elapsed = time.perf_counter() - t_start

        logger.info("=== Severity classification complete ===")
        logger.info("  Frames processed : %d", n_total)
        logger.info("  Boxes classified : %d", n_boxes)
        logger.info("  Elapsed          : %.2f s", elapsed)
        logger.info("  S1–S5 distribution:")
        for level in ["S1", "S2", "S3", "S4", "S5"]:
            cnt = level_counts.get(level, 0)
            pct = cnt / max(n_boxes, 1) * 100
            logger.info("    %s : %4d  (%.1f%%)", level, cnt, pct)

        if output_dir:
            self.save_severity(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_severity(
        results:    List[SeverityResult],
        output_dir: str,
    ) -> str:
        """Save severity results to severity_estimates.json."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "severity_estimates.json"

        n_boxes = sum(r.n_detections for r in results)
        level_counts: Dict[str, int] = {}
        for r in results:
            for b in r.boxes:
                if b.severity:
                    lvl = b.severity.severity_level
                    level_counts[lvl] = level_counts.get(lvl, 0) + 1

        payload = {
            "formula": {
                "signal_normalisers": {
                    "area_denom":      AREA_NORM_DENOM,
                    "contrast_denom":  CONTRAST_NORM_DENOM,
                    "sharpness_denom": SHARPNESS_NORM_DENOM,
                },
                "severity_intervals": [
                    {"lo": lo, "hi": hi, "level": lvl, "action": act}
                    for lo, hi, lvl, act in SEVERITY_INTERVALS
                ],
                "class_weights":    CLASS_WEIGHTS,
                "signal_weights":   {
                    cls: dict(zip(
                        ["w_depth", "w_area", "w_contrast", "w_sharpness"], w
                    ))
                    for cls, w in SIGNAL_WEIGHTS.items()
                },
            },
            "n_frames":      len(results),
            "n_boxes_total": n_boxes,
            "level_counts":  level_counts,
            "frames":        [r.to_dict() for r in results],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Severity estimates saved: %s  (%d frames, %d boxes)",
            out_path, len(results), n_boxes,
        )
        return str(out_path)

    @staticmethod
    def load_severity(path: str) -> List[SeverityResult]:
        """Load severity_estimates.json back into SeverityResult objects."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        from pipeline.segmentor import MaskGeometry
        from pipeline.depth_estimator import DepthEstimate

        results: List[SeverityResult] = []
        for fr in payload["frames"]:
            boxes = []
            for b in fr["boxes"]:

                geom_d = b.get("geometry")
                geometry = (
                    MaskGeometry(
                        surface_area_px   = geom_d["surface_area_px"],
                        edge_sharpness    = geom_d["edge_sharpness"],
                        interior_contrast = geom_d["interior_contrast"],
                        mask_compactness  = geom_d["mask_compactness"],
                    ) if geom_d else None
                )

                dep_d = b.get("depth")
                depth_est = (
                    DepthEstimate(
                        depth_raw         = dep_d["depth_raw"],
                        depth_norm        = dep_d["depth_norm"],
                        depth_confidence  = dep_d["depth_confidence"],
                        extraction_method = dep_d["extraction_method"],
                        used_proxy        = dep_d["used_proxy"],
                        frame_min_raw     = dep_d["frame_min_raw"],
                        frame_max_raw     = dep_d["frame_max_raw"],
                        region_px         = dep_d["region_px"],
                    ) if dep_d else None
                )

                sev_d = b.get("severity")
                severity_est = (
                    SeverityEstimate(
                        signal_scores       = sev_d["signal_scores"],
                        signal_weights_used = sev_d["signal_weights_used"],
                        raw_score           = sev_d["raw_score"],
                        class_weight        = sev_d["class_weight"],
                        severity_score      = sev_d["severity_score"],
                        severity_level      = sev_d["severity_level"],
                        severity_action     = sev_d["severity_action"],
                        severity_confidence = sev_d["severity_confidence"],
                    ) if sev_d else None
                )

                boxes.append(SeverityBox(
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
                    is_infrastructure = b.get("is_infrastructure", False),
                    is_marking        = b.get("is_marking", False),
                    sam_score         = b.get("sam_score"),
                    geometry          = geometry,
                    low_sam_quality   = b.get("low_sam_quality", False),
                    depth             = depth_est,
                    severity          = severity_est,
                ))

            results.append(SeverityResult(
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

        logger.info("Severity estimates loaded: %s (%d frames)", path, len(results))
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5 -- Rule-based severity classification."
    )
    parser.add_argument(
        "--depth_estimates", required=True,
        help="depth_estimates.json produced by Stage 4 (DepthEstimator)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for severity_estimates.json",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from pipeline.depth_estimator import DepthEstimator
    depth_results = DepthEstimator.load_depth_estimates(args.depth_estimates)
    logger.info(
        "Loaded %d depth results from: %s",
        len(depth_results), args.depth_estimates,
    )

    cfg        = SeverityConfig()
    classifier = SeverityClassifier(cfg)
    classifier.run(depth_results, output_dir=args.output)


if __name__ == "__main__":
    main()