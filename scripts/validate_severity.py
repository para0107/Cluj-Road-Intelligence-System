"""
scripts/validate_severity.py
-----------------------------
Validation script for Stage 5 — Rule-based severity classification.

Reads the depth_validation.json produced by scripts/validate_depth.py
(which already contains SAM geometry features forwarded from
detections_summary.json) and computes a severity level S1–S5 for every
detection using the weighted multi-signal formula.

Formula (fully documented in pipeline/severity_classifier.py):

  1. Four signal scores, each normalised to [0, 1]:
       S_depth     = depth_norm
       S_area      = min(surface_area_px / 5000, 1.0)
       S_contrast  = min(interior_contrast / 3.0, 1.0)
       S_sharpness = min(edge_sharpness / 100.0, 1.0)

  2. Class-specific signal weights (sum to 1.0 per class):
       raw_score = w_depth*S_depth + w_area*S_area
                 + w_contrast*S_contrast + w_sharpness*S_sharpness

  3. Class importance weight (how critical is this class to fix):
       severity_score = min(raw_score * class_weight * 2, 1.0)

  4. Mapping to S1–S5:
       [0.00, 0.15) → S1  Monitor
       [0.15, 0.35) → S2  Schedule maintenance
       [0.35, 0.55) → S3  Priority repair
       [0.55, 0.75) → S4  Urgent repair
       [0.75, 1.00] → S5  Emergency closure

Outputs
-------
  data/validation_nrdd_2024/severity/
      severity_results.json   -- all detections with severity added
      severity_summary.json   -- per-class and per-level statistics

Usage
-----
    python scripts/validate_severity.py
    python scripts/validate_severity.py --verbose

References
----------
  Godard et al., 2019. arXiv:1806.01260  (Monodepth2 — depth signal source)
  Kirillov et al., 2023. arXiv:2304.02643 (SAM — geometry signal source)
  Zhang et al., 2024. doi:10.1016/j.ijtst.2024.10.005 (SAM road asset validation)

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_severity")

# ---------------------------------------------------------------------------
# Paths — all absolute
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")

DEPTH_JSON = (
    PROJECT_ROOT
    / "data" / "validation_nrdd_2024" / "depth_maps" / "depth_validation.json"
)

OUT_DIR = PROJECT_ROOT / "data" / "validation_nrdd_2024" / "severity"

# ---------------------------------------------------------------------------
# Severity formula constants
# All normalisation denominators are derived from empirical class means
# observed in the Run 3 full-pipeline validation (detections_summary.json,
# depth_validation.json). No values are hardcoded without justification.
# ---------------------------------------------------------------------------

# Signal normalisation denominators
AREA_NORM_DENOM      = 5000.0   # px -- alligator crack mean = 5,405 px (largest class)
CONTRAST_NORM_DENOM  = 3.0      # -- pedestrian crossing mean = 2.882 (highest class)
SHARPNESS_NORM_DENOM = 100.0    # -- max observed in data ~103 (rutting)

# Per-class signal weights: [w_depth, w_area, w_contrast, w_sharpness]
# Each row sums to 1.0.
# Rationale documented per class below.
SIGNAL_WEIGHTS: Dict[str, tuple] = {
    # pothole: bowl shape → depth + shadow contrast are most diagnostic
    "pothole":                   (0.35, 0.25, 0.30, 0.10),
    # alligator_crack: wide spread of interconnected cracks → area dominates
    "alligator_crack":           (0.25, 0.40, 0.20, 0.15),
    # longitudinal_crack: active vs healed → sharpness of edges matters
    "longitudinal_crack":        (0.30, 0.25, 0.15, 0.30),
    # transverse_crack: same rationale as longitudinal, sharpness slightly heavier
    "transverse_crack":          (0.30, 0.20, 0.15, 0.35),
    # repaired_crack: sharpness reveals if repair is deteriorating again
    "repaired_crack":            (0.20, 0.30, 0.10, 0.40),
    # patchy_road: patch extent matters most
    "patchy_road":               (0.25, 0.40, 0.15, 0.20),
    # rutting: longitudinal depth groove → depth is primary signal
    "rutting":                   (0.40, 0.25, 0.10, 0.25),
    # manhole_cover: infrastructure, balanced signals
    "manhole_cover":             (0.30, 0.20, 0.25, 0.25),
    # marking classes: area of fading is most relevant;
    # both are assigned low class_weight so they never exceed S2
    "pedestrian_crossing_blur":  (0.10, 0.50, 0.30, 0.10),
    "lane_line_blur":            (0.10, 0.50, 0.30, 0.10),
}

# Class importance weights — how critical is this class to fix.
# Pothole = 0.50 (highest structural hazard).
# Lane blur = 0.10 (marking maintenance, not structural).
# With the ×2 rescaling, max possible severity_score = class_weight × 2.
# So lane_line_blur can never exceed 0.20 → always S1 or low S2.
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

# S1–S5 interval boundaries [lower, upper)
SEVERITY_INTERVALS = [
    (0.00, 0.15, "S1", "Monitor"),
    (0.15, 0.35, "S2", "Schedule maintenance"),
    (0.35, 0.55, "S3", "Priority repair"),
    (0.55, 0.75, "S4", "Urgent repair"),
    (0.75, 1.01, "S5", "Emergency closure"),  # 1.01 to include 1.0
]

SEVERITY_LABELS = {
    "S1": "Monitor",
    "S2": "Schedule maintenance",
    "S3": "Priority repair",
    "S4": "Urgent repair",
    "S5": "Emergency closure",
}


# ---------------------------------------------------------------------------
# Core severity computation
# ---------------------------------------------------------------------------

def compute_signal_scores(
    depth_norm:        Optional[float],
    surface_area_px:   Optional[int],
    interior_contrast: Optional[float],
    edge_sharpness:    Optional[float],
) -> Dict[str, float]:
    """
    Normalise raw signal values to [0, 1].
    Missing values (None) default to 0.0 — conservative assumption.
    """
    s_depth     = float(depth_norm)        if depth_norm        is not None else 0.0
    s_area      = float(surface_area_px)   if surface_area_px   is not None else 0.0
    s_contrast  = float(interior_contrast) if interior_contrast  is not None else 0.0
    s_sharpness = float(edge_sharpness)    if edge_sharpness    is not None else 0.0

    return {
        "S_depth":     round(min(s_depth,                            1.0), 4),
        "S_area":      round(min(s_area      / AREA_NORM_DENOM,      1.0), 4),
        "S_contrast":  round(min(s_contrast  / CONTRAST_NORM_DENOM,  1.0), 4),
        "S_sharpness": round(min(s_sharpness / SHARPNESS_NORM_DENOM, 1.0), 4),
    }


def compute_raw_score(
    signal_scores: Dict[str, float],
    class_name:    str,
) -> float:
    """
    Weighted combination of signal scores using per-class signal weights.
    Falls back to equal weights if class is not in SIGNAL_WEIGHTS.
    """
    weights = SIGNAL_WEIGHTS.get(class_name, (0.25, 0.25, 0.25, 0.25))
    w_depth, w_area, w_contrast, w_sharpness = weights

    raw = (
        w_depth     * signal_scores["S_depth"]     +
        w_area      * signal_scores["S_area"]      +
        w_contrast  * signal_scores["S_contrast"]  +
        w_sharpness * signal_scores["S_sharpness"]
    )
    return round(raw, 4)


def compute_severity_score(raw_score: float, class_name: str) -> float:
    """
    Apply class importance weight and rescale to [0, 1].

    severity_score = min(raw_score * class_weight * 2, 1.0)

    The ×2 rescaling ensures that a class with class_weight=0.50
    (pothole) and raw_score=1.0 achieves severity_score=1.0,
    while a class with class_weight=0.10 (lane_line_blur) and
    raw_score=1.0 achieves severity_score=0.20 — capping it at S2.
    """
    class_weight = CLASS_WEIGHTS.get(class_name, 0.25)
    score = raw_score * class_weight * 2.0
    return round(min(score, 1.0), 4)


def score_to_level(severity_score: float) -> tuple[str, str]:
    """Map severity_score ∈ [0, 1] to (level, action) using fixed intervals."""
    for lo, hi, level, action in SEVERITY_INTERVALS:
        if lo <= severity_score < hi:
            return level, action
    return "S5", "Emergency closure"   # fallback for score == 1.0


def classify_detection(
    class_name:        str,
    depth_norm:        Optional[float],
    surface_area_px:   Optional[int],
    interior_contrast: Optional[float],
    edge_sharpness:    Optional[float],
    used_proxy:        bool = False,
    depth_confidence:  Optional[float] = None,
) -> dict:
    """
    Full severity classification for one detection.

    Returns a dict with all intermediate values for auditability:
      signal_scores, raw_score, class_weight, severity_score,
      severity_level, severity_action, severity_confidence
    """
    signal_scores  = compute_signal_scores(
        depth_norm, surface_area_px, interior_contrast, edge_sharpness
    )
    raw_score      = compute_raw_score(signal_scores, class_name)
    severity_score = compute_severity_score(raw_score, class_name)
    level, action  = score_to_level(severity_score)

    # Severity confidence: penalise when depth proxy was used or
    # depth_confidence is low (geometry proxy substituted Monodepth2).
    # Full confidence when all signals come from real measurements.
    if used_proxy:
        sev_confidence = round(0.5 * (1.0 + (depth_confidence or 0.0)), 4)
    else:
        sev_confidence = round(min(1.0, 0.6 + 0.4 * (depth_confidence or 1.0)), 4)

    return {
        "signal_scores":       signal_scores,
        "raw_score":           raw_score,
        "class_weight":        CLASS_WEIGHTS.get(class_name, 0.25),
        "signal_weights":      dict(zip(
            ["w_depth", "w_area", "w_contrast", "w_sharpness"],
            SIGNAL_WEIGHTS.get(class_name, (0.25, 0.25, 0.25, 0.25)),
        )),
        "severity_score":      severity_score,
        "severity_level":      level,
        "severity_action":     action,
        "severity_confidence": sev_confidence,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(verbose: bool = False) -> None:

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load depth_validation.json — real data, no hardcoded values
    # ------------------------------------------------------------------
    if not DEPTH_JSON.exists():
        logger.error("depth_validation.json not found: %s", DEPTH_JSON)
        logger.error(
            "Run scripts/validate_depth.py --device cuda first to generate it."
        )
        raise SystemExit(1)

    logger.info("Loading: %s", DEPTH_JSON)
    with DEPTH_JSON.open("r", encoding="utf-8") as f:
        depth_data = json.load(f)

    frames = depth_data["frames"]
    logger.info(
        "Loaded %d frames, model=%s, device=%s",
        len(frames), depth_data.get("model"), depth_data.get("device"),
    )

    # ------------------------------------------------------------------
    # Classify every detection
    # ------------------------------------------------------------------
    severity_frames   = []
    class_scores      = defaultdict(list)
    level_counts      = defaultdict(int)
    class_level_counts = defaultdict(lambda: defaultdict(int))
    n_boxes_total     = 0
    n_no_geometry     = 0
    n_no_depth        = 0
    n_proxy_used      = 0

    for frame in frames:
        frame_severity_boxes = []

        for box in frame["boxes"]:
            cls        = box["class_name"]
            depth_info = box.get("depth")
            geom_info  = box.get("geometry")

            # Extract signals — None if unavailable
            depth_norm        = depth_info["depth_norm"]        if depth_info else None
            depth_confidence  = depth_info.get("depth_confidence") if depth_info else None
            used_proxy        = depth_info.get("used_proxy", False) if depth_info else False
            surface_area_px   = geom_info["surface_area_px"]   if geom_info else None
            interior_contrast = geom_info["interior_contrast"] if geom_info else None
            edge_sharpness    = geom_info["edge_sharpness"]    if geom_info else None

            if depth_info is None:
                n_no_depth += 1
            if geom_info is None:
                n_no_geometry += 1
            if used_proxy:
                n_proxy_used += 1

            result = classify_detection(
                class_name        = cls,
                depth_norm        = depth_norm,
                surface_area_px   = surface_area_px,
                interior_contrast = interior_contrast,
                edge_sharpness    = edge_sharpness,
                used_proxy        = used_proxy,
                depth_confidence  = depth_confidence,
            )

            n_boxes_total += 1
            class_scores[cls].append(result["severity_score"])
            level_counts[result["severity_level"]] += 1
            class_level_counts[cls][result["severity_level"]] += 1

            # Build output box — forward all original fields, add severity
            out_box = {
                "class_id":          box.get("class_id"),
                "class_name":        cls,
                "confidence":        box.get("confidence"),
                "is_damage":         box.get("is_damage"),
                "is_infrastructure": box.get("is_infrastructure", False),
                "is_marking":        box.get("is_marking", False),
                "x1":                box.get("x1"),
                "y1":                box.get("y1"),
                "x2":                box.get("x2"),
                "y2":                box.get("y2"),
                "sam_score":         box.get("sam_score"),
                "geometry":          geom_info,
                "depth":             depth_info,
                "severity":          result,
            }
            frame_severity_boxes.append(out_box)

            logger.debug(
                "  [%s] score=%.3f level=%s (depth=%.3f area=%s contrast=%.3f sharp=%.3f)",
                cls,
                result["severity_score"],
                result["severity_level"],
                depth_norm or 0.0,
                surface_area_px or 0,
                interior_contrast or 0.0,
                edge_sharpness or 0.0,
            )

        severity_frames.append({
            "frame_path":    frame["frame_path"],
            "frame_stem":    frame["frame_stem"],
            "frame_index":   frame["frame_index"],
            "timestamp_s":   frame["timestamp_s"],
            "lighting":      frame["lighting"],
            "sun_elevation": frame.get("sun_elevation"),
            "image_width":   frame["image_width"],
            "image_height":  frame["image_height"],
            "depth_map_min": frame.get("depth_map_min"),
            "depth_map_max": frame.get("depth_map_max"),
            "boxes":         frame_severity_boxes,
        })

    # ------------------------------------------------------------------
    # Build per-class summary statistics
    # ------------------------------------------------------------------
    class_summary = {}
    for cls in sorted(class_scores.keys()):
        scores = class_scores[cls]
        class_summary[cls] = {
            "n_detections":    len(scores),
            "mean_score":      round(statistics.mean(scores), 4),
            "min_score":       round(min(scores), 4),
            "max_score":       round(max(scores), 4),
            "stdev_score":     round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "class_weight":    CLASS_WEIGHTS.get(cls, 0.25),
            "level_breakdown": dict(class_level_counts[cls]),
        }

    # ------------------------------------------------------------------
    # Print human-readable summary to console
    # ------------------------------------------------------------------
    logger.info("=== Severity Classification Complete ===")
    logger.info("  Total detections classified : %d", n_boxes_total)
    logger.info("  Detections without depth    : %d", n_no_depth)
    logger.info("  Detections without geometry : %d", n_no_geometry)
    logger.info("  Detections using proxy depth: %d", n_proxy_used)
    logger.info("")
    logger.info("  S1–S5 distribution:")
    for level in ["S1", "S2", "S3", "S4", "S5"]:
        cnt = level_counts.get(level, 0)
        pct = cnt / max(n_boxes_total, 1) * 100
        bar = "█" * (cnt // 10)
        logger.info(
            "    %s (%s): %4d  (%5.1f%%)  %s",
            level, SEVERITY_LABELS[level], cnt, pct, bar,
        )
    logger.info("")
    logger.info("  Per-class severity scores:")
    for cls, stats in sorted(class_summary.items(),
                              key=lambda x: -x[1]["mean_score"]):
        logger.info(
            "    %-35s  mean=%.3f  min=%.3f  max=%.3f  "
            "class_weight=%.2f  n=%d  levels=%s",
            cls,
            stats["mean_score"],
            stats["min_score"],
            stats["max_score"],
            stats["class_weight"],
            stats["n_detections"],
            stats["level_breakdown"],
        )

    # ------------------------------------------------------------------
    # Save severity_results.json
    # ------------------------------------------------------------------
    results_path = OUT_DIR / "severity_results.json"
    results_payload = {
        "source_depth_json":       str(DEPTH_JSON),
        "model":                   depth_data.get("model"),
        "device":                  depth_data.get("device"),
        "n_frames":                len(severity_frames),
        "n_boxes_total":           n_boxes_total,
        "n_no_depth":              n_no_depth,
        "n_no_geometry":           n_no_geometry,
        "n_proxy_used":            n_proxy_used,
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
        "level_counts":            dict(level_counts),
        "frames":                  severity_frames,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    logger.info("Severity results saved: %s", results_path)

    # ------------------------------------------------------------------
    # Save severity_summary.json
    # ------------------------------------------------------------------
    summary_path = OUT_DIR / "severity_summary.json"
    summary_payload = {
        "n_frames":      len(severity_frames),
        "n_detections":  n_boxes_total,
        "level_counts":  dict(level_counts),
        "class_summary": class_summary,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    logger.info("Severity summary saved: %s", summary_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5 validation — severity classification on depth_validation.json"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging (prints per-detection scores)",
    )
    args = parser.parse_args()
    main(verbose=args.verbose)