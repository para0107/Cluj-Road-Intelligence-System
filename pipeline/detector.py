"""
pipeline/detector.py
---------------------
Stage 2 of the road damage detection inference pipeline.

Responsibilities:
  - Load RT-DETR-L weights (N-RDD2024 fine-tuned checkpoint, Run 2)
  - Run inference on every frame in the preprocessor manifest
  - Apply per-class confidence thresholds as post-processing
    (global Ultralytics threshold is set to 0.001; per-class filtering
    is applied here so no box is silently discarded before we see it)
  - Return one DetectionResult per frame, containing all accepted bounding boxes

Per-class thresholds (Section 4.10 of thesis):
  - All damage classes:         0.35  (near F1 peak for Run 2)
  - lane_line_blur:             0.50  (interim — replace with F1-curve peak
                                       from model.val() on N-RDD2024 val split)
  - pedestrian_crossing_blur:   0.35  (global; severity S1 cap handles priority)

Severity overrides:
  - lane_line_blur and pedestrian_crossing_blur are always assigned S1
    regardless of confidence — they are marking maintenance issues, not
    structural road damage (Section 4.10).

TTA note:
  Test Time Augmentation was evaluated during training and produced zero
  accuracy improvement for RT-DETR on this dataset. Disabled here.
  (Ref: Section 4.5 of thesis)

N-RDD2024 class mapping (10-class schema, Kaya & Codur 2024):
  0 -> longitudinal_crack        (D00)
  1 -> transverse_crack          (D10)
  2 -> alligator_crack           (D20)
  3 -> repaired_crack            (D30)
  4 -> pothole                   (D40)
  5 -> pedestrian_crossing_blur  (D50)
  6 -> lane_line_blur            (D60)
  7 -> manhole_cover             (D70)
  8 -> patchy_road               (D80)
  9 -> rutting                   (D90)

Usage (module):
    from pipeline.preprocessor import Preprocessor
    from pipeline.detector import Detector, DetectorConfig

    frames  = Preprocessor.load_manifest("data/processed/frames/run/manifest.json")
    cfg     = DetectorConfig(
                  weights="ml/weights/rtdetr_l_nrdd2024.pt",
                  conf=0.35,
              )
    det     = Detector(cfg)
    results = det.run(frames)

Usage (CLI):
    python pipeline/detector.py
        --manifest  data/processed/frames/run/manifest.json
        --weights   ml/weights/rtdetr_l_nrdd2024.pt
        --output    data/processed/detections/run/
        [--conf 0.35] [--device cpu] [--verbose]

References:
    RT-DETR:    Zhao et al., 2024. arXiv:2304.08069
    N-RDD2024:  Kaya & Codur, 2024. doi:10.17632/27c8pwsd6v.3
    Focal Loss: Lin et al., 2017. arXiv:1708.02002
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2   # used only for the optional debug overlay; already a project dependency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class colour map for debug overlays (BGR).
# Mirrors _CLASS_COLOURS in segmentor.py so the detection and segmentation
# debug images use the same colour per class. Kept local here to avoid a
# module-level import dependency on segmentor.
# ---------------------------------------------------------------------------
_CLASS_COLOURS: dict[str, tuple] = {
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
# N-RDD2024 10-class schema
# Order must match the class IDs used during training.
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "longitudinal_crack",        # 0  D00 -- structural damage
    "transverse_crack",          # 1  D10 -- structural damage
    "alligator_crack",           # 2  D20 -- structural damage
    "repaired_crack",            # 3  D30 -- structural damage (repaired)
    "pothole",                   # 4  D40 -- structural damage
    "pedestrian_crossing_blur",  # 5  D50 -- marking (S1 override)
    "lane_line_blur",            # 6  D60 -- marking (S1 override, conf >= 0.50)
    "manhole_cover",             # 7  D70 -- infrastructure annotation
    "patchy_road",               # 8  D80 -- structural damage
    "rutting",                   # 9  D90 -- structural damage
]

# ---------------------------------------------------------------------------
# Per-class confidence thresholds.
#
# Ultralytics model.predict() is called with conf=0.001 so that no box is
# discarded before reaching this filter. The thresholds below are applied
# as post-processing in the detection loop.
#
# 0.35 -- operational threshold for all damage classes. Sits near the
#         macro-averaged F1 peak for Run 2 (Section 4.5).
# 0.50 -- interim threshold for lane_line_blur only. Replace this value
#         with the F1-peak confidence from model.val() on the N-RDD2024
#         validation split once that calibration run is complete.
# ---------------------------------------------------------------------------
CLASS_CONF_THRESHOLDS: dict[str, float] = {
    "longitudinal_crack":        0.35,
    "transverse_crack":          0.35,
    "alligator_crack":           0.35,
    "repaired_crack":            0.35,
    "pothole":                   0.35,
    "pedestrian_crossing_blur":  0.35,
    "lane_line_blur":            0.50,   # interim -- calibrate from F1 curve
    "manhole_cover":             0.35,
    "patchy_road":               0.35,
    "rutting":                   0.35,
}

# ---------------------------------------------------------------------------
# Severity priors -- used by the rule-based severity classifier (Stage 5)
# as a starting point before depth and area signals are applied.
# Marking classes are always S1 regardless of confidence or surface condition.
# ---------------------------------------------------------------------------
CLASS_SEVERITY_PRIOR: dict[str, str] = {
    "longitudinal_crack":        "S2",
    "transverse_crack":          "S2",
    "alligator_crack":           "S3",
    "repaired_crack":            "S1",
    "pothole":                   "S3",
    "pedestrian_crossing_blur":  "S1",   # marking -- always S1
    "lane_line_blur":            "S1",   # marking -- always S1
    "manhole_cover":             "S2",
    "patchy_road":               "S2",
    "rutting":                   "S3",
}

# ---------------------------------------------------------------------------
# Semantic class sets.
# Used by downstream stages (segmentor, severity classifier, deduplicator,
# priority scorer) to route each detection without re-checking class names.
# ---------------------------------------------------------------------------
DAMAGE_CLASSES: set[str] = {
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "repaired_crack",
    "pothole",
    "patchy_road",
    "rutting",
}

INFRASTRUCTURE_CLASSES: set[str] = {
    "manhole_cover",
}

MARKING_CLASSES: set[str] = {
    "lane_line_blur",
    "pedestrian_crossing_blur",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    """
    Configuration for Stage 2 -- RT-DETR inference.

    weights:
        Path to the trained RT-DETR-L checkpoint (.pt file).
        Default: N-RDD2024 fine-tuned checkpoint (Run 2, mAP50=0.577 val).

    conf:
        Reference confidence threshold documented in the thesis (0.35).
        Note: Ultralytics model.predict() is actually called with conf=0.001
        and per-class filtering is applied as post-processing using
        CLASS_CONF_THRESHOLDS. This field is retained for documentation,
        logging, and CLI help text.

    iou:
        IoU threshold for internal duplicate suppression.
        0.6 is the standard value used during mAP evaluation.

    imgsz:
        Inference resolution. Must match training resolution (640).

    device:
        "cpu", "cuda", "cuda:0", "0", etc.

    batch_size:
        Frames per forward pass. Keep at 1 for CPU; increase to 4-8 on GPU.
    """
    weights:    str   = "ml/weights/rtdetr_l_nrdd2024.pt"
    conf:       float = 0.35
    iou:        float = 0.6
    imgsz:      int   = 640
    device:     str   = "cpu"
    batch_size: int   = 1
    save_debug: bool  = False   # if True, write annotated frames to output_dir/debug/


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    """A single detection bounding box in pixel coordinates."""
    x1:               float
    y1:               float
    x2:               float
    y2:               float
    class_id:         int
    class_name:       str
    confidence:       float
    threshold_applied: float       # per-class threshold that was applied
    severity_prior:   str          # from CLASS_SEVERITY_PRIOR
    is_damage:        bool         # structural road damage
    is_infrastructure: bool        # infrastructure annotation (manhole etc.)
    is_marking:       bool         # road marking annotation (D50, D60)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "x1":               round(self.x1, 2),
            "y1":               round(self.y1, 2),
            "x2":               round(self.x2, 2),
            "y2":               round(self.y2, 2),
            "class_id":         self.class_id,
            "class_name":       self.class_name,
            "confidence":       round(self.confidence, 4),
            "threshold_applied": self.threshold_applied,
            "severity_prior":   self.severity_prior,
            "is_damage":        self.is_damage,
            "is_infrastructure": self.is_infrastructure,
            "is_marking":       self.is_marking,
        }


@dataclass
class DetectionResult:
    """All detections for one frame, including GPS and lighting context."""
    frame_path:    str
    frame_index:   int
    timestamp_s:   float
    latitude:      Optional[float]
    longitude:     Optional[float]
    wall_time:     Optional[str]
    lighting:      str
    sun_elevation: Optional[float]
    image_width:   int
    image_height:  int
    boxes:         List[BoundingBox] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return len(self.boxes)

    @property
    def has_detections(self) -> bool:
        return len(self.boxes) > 0

    @property
    def damage_boxes(self) -> List[BoundingBox]:
        return [b for b in self.boxes if b.is_damage]

    @property
    def marking_boxes(self) -> List[BoundingBox]:
        return [b for b in self.boxes if b.is_marking]

    @property
    def infrastructure_boxes(self) -> List[BoundingBox]:
        return [b for b in self.boxes if b.is_infrastructure]

    def to_dict(self) -> dict:
        return {
            "frame_path":    self.frame_path,
            "frame_index":   self.frame_index,
            "timestamp_s":   self.timestamp_s,
            "latitude":      self.latitude,
            "longitude":     self.longitude,
            "wall_time":     self.wall_time,
            "lighting":      self.lighting,
            "sun_elevation": self.sun_elevation,
            "image_width":   self.image_width,
            "image_height":  self.image_height,
            "n_detections":  self.n_detections,
            "boxes":         [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Debug overlay helper
# ---------------------------------------------------------------------------

def _draw_detections(
    image_bgr,
    boxes: List["BoundingBox"],
):
    """
    Draw bounding boxes and class/confidence labels onto a copy of the frame.

    Colours follow _CLASS_COLOURS. The returned image is a new array; the
    input is not modified.
    """
    out = image_bgr.copy()
    for b in boxes:
        colour = _CLASS_COLOURS.get(b.class_name, (0, 255, 0))
        x1, y1 = int(b.x1), int(b.y1)
        x2, y2 = int(b.x2), int(b.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = f"{b.class_name} {b.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(
            out, (x1, ly - th - base), (x1 + tw, ly + base), colour, -1
        )
        cv2.putText(
            out, label, (x1, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """
    Stage 2 -- RT-DETR-L inference wrapper.

    The model is loaded once at construction time (lazy, on first call to
    run()) and reused across all frames in the batch.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg    = config or DetectorConfig()
        self._model = None
        logger.info(
            "Detector config | weights=%s | ref_conf=%.2f | iou=%.2f | "
            "imgsz=%d | device=%s | batch=%d",
            self.cfg.weights, self.cfg.conf, self.cfg.iou,
            self.cfg.imgsz, self.cfg.device, self.cfg.batch_size,
        )
        logger.info(
            "Per-class thresholds: %s",
            {k: v for k, v in CLASS_CONF_THRESHOLDS.items()},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the RT-DETR model on first call to run()."""
        if self._model is not None:
            return

        weights_path = Path(self.cfg.weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.cfg.weights}\n"
                f"Place the checkpoint at the path above or update "
                f"DetectorConfig.weights."
            )

        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Install with: pip install ultralytics==8.3.143"
            )

        logger.info("Loading RT-DETR-L from: %s", self.cfg.weights)
        self._model = RTDETR(str(weights_path))
        logger.info("Model loaded | classes: %s", CLASS_NAMES)

    def _validate_class_names(self) -> None:
        """
        Warn if the checkpoint's built-in class names differ from CLASS_NAMES.
        A mismatch means the weights were trained on a different schema.
        """
        if self._model is None:
            return
        model_names = self._model.names or {}
        for idx, expected in enumerate(CLASS_NAMES):
            actual = model_names.get(idx, "<missing>")
            if actual != expected:
                logger.warning(
                    "Class ID %d mismatch -- checkpoint says '%s', "
                    "CLASS_NAMES says '%s'. "
                    "Ensure the weights match the N-RDD2024 10-class schema.",
                    idx, actual, expected,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        frames,                         # List[FrameResult] from preprocessor
        output_dir: Optional[str] = None,
    ) -> List[DetectionResult]:
        """
        Run RT-DETR inference on all frames in the manifest.

        The global Ultralytics confidence threshold is set to 0.001 so that
        all raw predictions reach this method. Per-class filtering using
        CLASS_CONF_THRESHOLDS is applied in the inner loop.

        Args:
            frames:     List of FrameResult objects from Stage 1 (Preprocessor).
            output_dir: If provided, saves detections.json here.

        Returns:
            List of DetectionResult, one per successfully processed frame.
            Frames with zero accepted detections are included so downstream
            stages can correlate by frame_index.
        """
        self._load_model()
        self._validate_class_names()

        debug_dir: Optional[Path] = None
        if output_dir and self.cfg.save_debug:
            debug_dir = Path(output_dir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug overlays will be saved to: %s", debug_dir)

        results: List[DetectionResult] = []
        n_frames      = len(frames)
        total_boxes   = 0
        total_dropped = 0

        logger.info(
            "=== Detector.run === %d frames | device=%s",
            n_frames, self.cfg.device,
        )

        for idx, frame in enumerate(frames):

            if not Path(frame.frame_path).exists():
                logger.warning(
                    "Frame %d: image not found at %s -- skipping",
                    frame.frame_index, frame.frame_path,
                )
                continue

            # ----------------------------------------------------------
            # Inference.
            # conf=0.001 -- cast the widest possible net; per-class
            # filtering happens below.  TTA is disabled (augment=False).
            # ----------------------------------------------------------
            try:
                preds = self._model.predict(
                    source=frame.frame_path,
                    conf=0.001,
                    iou=self.cfg.iou,
                    imgsz=self.cfg.imgsz,
                    device=self.cfg.device,
                    augment=False,    # TTA disabled -- zero gain (Section 4.5)
                    verbose=False,
                    save=False,
                )
            except Exception:
                logger.exception(
                    "Frame %d: inference failed on %s",
                    frame.frame_index, frame.frame_path,
                )
                continue

            pred         = preds[0]
            img_h, img_w = pred.orig_shape

            # ----------------------------------------------------------
            # Per-class confidence filter
            # ----------------------------------------------------------
            boxes: List[BoundingBox] = []

            if pred.boxes is not None and len(pred.boxes) > 0:
                for box in pred.boxes:
                    coords    = box.xyxy[0].tolist()
                    cls_id    = int(box.cls[0].item())
                    conf      = float(box.conf[0].item())

                    if cls_id >= len(CLASS_NAMES):
                        logger.warning(
                            "Frame %d: unknown class_id %d -- skipping box",
                            frame.frame_index, cls_id,
                        )
                        continue

                    cls_name  = CLASS_NAMES[cls_id]
                    threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.35)

                    if conf < threshold:
                        total_dropped += 1
                        logger.debug(
                            "DROPPED  %-28s  conf=%.3f  threshold=%.2f  "
                            "frame=%d",
                            cls_name, conf, threshold, frame.frame_index,
                        )
                        continue

                    boxes.append(BoundingBox(
                        x1                = coords[0],
                        y1                = coords[1],
                        x2                = coords[2],
                        y2                = coords[3],
                        class_id          = cls_id,
                        class_name        = cls_name,
                        confidence        = round(conf, 4),
                        threshold_applied = threshold,
                        severity_prior    = CLASS_SEVERITY_PRIOR[cls_name],
                        is_damage         = cls_name in DAMAGE_CLASSES,
                        is_infrastructure = cls_name in INFRASTRUCTURE_CLASSES,
                        is_marking        = cls_name in MARKING_CLASSES,
                    ))

            total_boxes += len(boxes)

            result = DetectionResult(
                frame_path    = frame.frame_path,
                frame_index   = frame.frame_index,
                timestamp_s   = frame.timestamp_s,
                latitude      = frame.latitude,
                longitude     = frame.longitude,
                wall_time     = (
                    frame.wall_time.isoformat() if frame.wall_time else None
                ),
                lighting      = frame.lighting,
                sun_elevation = frame.sun_elevation,
                image_width   = img_w,
                image_height  = img_h,
                boxes         = boxes,
            )
            results.append(result)

            # Debug overlay — only for frames that actually have detections
            if debug_dir is not None and boxes:
                dbg_img = cv2.imread(frame.frame_path)
                if dbg_img is None:
                    logger.warning(
                        "Debug: cannot read frame %s for overlay",
                        frame.frame_path,
                    )
                else:
                    annotated = _draw_detections(dbg_img, boxes)
                    stem     = Path(frame.frame_path).stem
                    out_jpg  = debug_dir / f"{stem}.jpg"
                    cv2.imwrite(str(out_jpg), annotated)

            # Log every 10 frames, or any frame that has detections
            if idx % 10 == 0 or len(boxes) > 0:
                logger.info(
                    "Frame %d/%d | t=%.1fs | gps=(%s,%s) | "
                    "lighting=%-10s | %d det(s) %s",
                    idx + 1,
                    n_frames,
                    frame.timestamp_s,
                    f"{frame.latitude:.5f}"
                        if frame.latitude  is not None else "None",
                    f"{frame.longitude:.5f}"
                        if frame.longitude is not None else "None",
                    frame.lighting or "unknown",
                    len(boxes),
                    [f"{b.class_name}({b.confidence:.2f})" for b in boxes],
                )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        frames_with_hits = sum(1 for r in results if r.has_detections)

        logger.info("=== Detection complete ===")
        logger.info("  Frames processed   : %d", len(results))
        logger.info(
            "  Frames with hits   : %d / %d  (%.0f%%)",
            frames_with_hits, len(results),
            100 * frames_with_hits / max(len(results), 1),
        )
        logger.info("  Boxes accepted     : %d", total_boxes)
        logger.info(
            "  Boxes dropped      : %d  (below per-class threshold)",
            total_dropped,
        )

        class_counts: dict[str, int] = {}
        for r in results:
            for b in r.boxes:
                class_counts[b.class_name] = \
                    class_counts.get(b.class_name, 0) + 1

        logger.info("  Per-class counts:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            tag = (
                "[DAMAGE]"  if cls in DAMAGE_CLASSES        else
                "[INFRA]"   if cls in INFRASTRUCTURE_CLASSES else
                "[MARKING]"
            )
            logger.info(
                "    %-28s : %3d  %s  threshold=%.2f",
                cls, cnt, tag, CLASS_CONF_THRESHOLDS.get(cls, 0.35),
            )

        if output_dir:
            self.save_detections(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_detections(
        results: List[DetectionResult],
        output_dir: str,
    ) -> str:
        """Save detection results to detections.json in output_dir."""
        out      = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "detections.json"

        payload = {
            "n_frames":         len(results),
            "n_boxes_accepted": sum(r.n_detections for r in results),
            "class_names":      CLASS_NAMES,
            "class_thresholds": CLASS_CONF_THRESHOLDS,
            "frames":           [r.to_dict() for r in results],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Detections saved: %s  (%d frames, %d boxes)",
            out_path, payload["n_frames"], payload["n_boxes_accepted"],
        )
        return str(out_path)

    @staticmethod
    def load_detections(detections_path: str) -> List[DetectionResult]:
        """Load a saved detections.json back into DetectionResult objects."""
        with open(detections_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results = []
        for fr in payload["frames"]:
            boxes = [
                BoundingBox(
                    x1                = b["x1"],
                    y1                = b["y1"],
                    x2                = b["x2"],
                    y2                = b["y2"],
                    class_id          = b["class_id"],
                    class_name        = b["class_name"],
                    confidence        = b["confidence"],
                    threshold_applied = b.get("threshold_applied", 0.35),
                    severity_prior    = b.get("severity_prior", "S2"),
                    is_damage         = b.get(
                        "is_damage",
                        b["class_name"] in DAMAGE_CLASSES,
                    ),
                    is_infrastructure = b.get(
                        "is_infrastructure",
                        b["class_name"] in INFRASTRUCTURE_CLASSES,
                    ),
                    is_marking        = b.get(
                        "is_marking",
                        b["class_name"] in MARKING_CLASSES,
                    ),
                )
                for b in fr["boxes"]
            ]
            results.append(DetectionResult(
                frame_path    = fr["frame_path"],
                frame_index   = fr["frame_index"],
                timestamp_s   = fr["timestamp_s"],
                latitude      = fr.get("latitude"),
                longitude     = fr.get("longitude"),
                wall_time     = fr.get("wall_time"),
                lighting      = fr.get("lighting", "unknown"),
                sun_elevation = fr.get("sun_elevation"),
                image_width   = fr["image_width"],
                image_height  = fr["image_height"],
                boxes         = boxes,
            ))

        logger.info(
            "Loaded detections: %s  (%d frames)",
            detections_path, len(results),
        )
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
        description="Stage 2 -- RT-DETR-L road damage detector (N-RDD2024)."
    )
    parser.add_argument(
        "--manifest", required=True,
        help="manifest.json produced by Stage 1 (Preprocessor)",
    )
    parser.add_argument(
        "--weights",
        default="ml/weights/rtdetr_l_nrdd2024.pt",
        help="Path to RT-DETR-L .pt checkpoint",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for detections.json",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Reference confidence threshold (documented value; actual "
             "per-class thresholds are defined in CLASS_CONF_THRESHOLDS)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.6,
        help="IoU threshold (default 0.6)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu | cuda | cuda:0  (default: cpu)",
    )
    parser.add_argument(
        "--save_debug", action="store_true",
        help="Save annotated frames (boxes + labels) to output/debug/ "
             "for frames that have at least one detection",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from pipeline.preprocessor import Preprocessor
    frames = Preprocessor.load_manifest(args.manifest)
    logger.info(
        "Loaded %d frames from manifest: %s", len(frames), args.manifest,
    )

    cfg = DetectorConfig(
        weights = args.weights,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device,
        save_debug = args.save_debug,
    )
    detector = Detector(cfg)
    detector.run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()