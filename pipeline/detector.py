"""
pipeline/detector.py
---------------------
Stage 2 of the CRIS inference pipeline.

Responsibilities:
  - Load RT-DETR-L weights (PSO-optimised checkpoint)
  - Run inference on every frame in the preprocessor manifest
  - Apply confidence threshold (default 0.35 — recommended operational value
    per Chapter 4 evaluation: precision > 0.70 for all classes at this threshold)
  - Return one DetectionResult per frame, containing all accepted bounding boxes

TTA note:
  Test Time Augmentation (flip + rotate averaging) was evaluated during training
  and produced zero accuracy improvement for RT-DETR on this dataset.
  It is intentionally disabled here. (Ref: Section 4.5 of thesis)

Class mapping (matches RDD2022 training schema):
  0 → longitudinal_crack
  1 → transverse_crack
  2 → alligator_crack
  3 → pothole
  4 → patch_deterioration  (no training data — will never fire)

Usage (module):
    from pipeline.preprocessor import Preprocessor
    from pipeline.detector import Detector, DetectorConfig

    frames  = Preprocessor.load_manifest("data/processed/frames/run/manifest.json")
    cfg     = DetectorConfig(weights="ml/weights/rtdetr_l_rdd2022.pt", conf=0.35)
    det     = Detector(cfg)
    results = det.run(frames)

Usage (CLI):
    python pipeline/detector.py \
        --manifest  data/processed/frames/validation/manifest.json \
        --weights   ml/weights/rtdetr_l_rdd2022.pt \
        --output    data/processed/detections/validation/ \
        [--conf 0.35] [--device cpu] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class definitions — must match the order used during RT-DETR training
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "longitudinal_crack",   # 0
    "transverse_crack",     # 1
    "alligator_crack",      # 2
    "pothole",              # 3
    "patch_deterioration",  # 4  — no training data, included for schema completeness
]

# Severity hints per class — used downstream by the rule-based severity classifier
# These are not ground truth; they inform the prior before depth/area are available.
CLASS_SEVERITY_PRIOR = {
    "longitudinal_crack":  "S2",
    "transverse_crack":    "S2",
    "alligator_crack":     "S3",
    "pothole":             "S3",
    "patch_deterioration": "S2",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    """
    Configuration for Stage 2 — RT-DETR inference.

    weights:
        Path to the trained RT-DETR-L checkpoint (.pt file).
        Use ml/weights/rtdetr_l_rdd2022.pt (PSO-optimised, mAP50=0.465 val).

    conf:
        Confidence threshold. Detections below this are discarded.
        0.35 is the recommended operational value — it sits near the
        macro-averaged F1 peak (0.369) and gives precision > 0.70 for
        all four trained classes. (Ref: Section 4.5, Table 4.9)

    iou:
        IoU threshold for internal duplicate suppression. RT-DETR uses
        bipartite matching so NMS is not applied, but Ultralytics still
        accepts this parameter for the decoder's score filtering.
        0.6 is the standard value used during mAP evaluation.

    imgsz:
        Inference resolution. Must match training resolution (640).

    device:
        "cpu", "cuda", "cuda:0", "0", etc.
        Defaults to "cpu" for portability — switch to "cuda" on Kaggle/GPU.

    batch_size:
        Number of frames to infer in a single forward pass.
        On CPU, 1 is fastest due to memory transfer overhead.
        On GPU, increase to 4–8 for throughput.
    """
    weights: str = "ml/weights/rtdetr_l_rdd2022.pt"
    conf: float = 0.35
    iou: float = 0.6
    imgsz: int = 640
    device: str = "cpu"
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    """A single detection bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float
    severity_prior: str   # from CLASS_SEVERITY_PRIOR

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
    def centre(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "class_id":      self.class_id,
            "class_name":    self.class_name,
            "confidence":    round(self.confidence, 4),
            "severity_prior": self.severity_prior,
        }


@dataclass
class DetectionResult:
    """All detections for one frame."""
    frame_path: str
    frame_index: int
    timestamp_s: float
    latitude: Optional[float]
    longitude: Optional[float]
    wall_time: Optional[str]
    lighting: str
    sun_elevation: Optional[float]
    image_width: int
    image_height: int
    boxes: List[BoundingBox] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return len(self.boxes)

    @property
    def has_detections(self) -> bool:
        return len(self.boxes) > 0

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
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """
    Stage 2 — RT-DETR-L inference wrapper.

    The model is loaded once at construction time and reused across all frames.
    Ultralytics handles image resizing and normalisation internally.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self._model = None
        logger.info(
            "Detector config | weights=%s | conf=%.2f | iou=%.2f | "
            "imgsz=%d | device=%s | batch=%d",
            self.cfg.weights, self.cfg.conf, self.cfg.iou,
            self.cfg.imgsz, self.cfg.device, self.cfg.batch_size,
        )

    def _load_model(self):
        """Lazy-load the RT-DETR model on first call."""
        if self._model is not None:
            return

        weights_path = Path(self.cfg.weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.cfg.weights}\n"
                f"Download from Kaggle dataset 'cluj-road-weights' and place at "
                f"ml/weights/rtdetr_l_rdd2022.pt"
            )

        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics==8.2.18"
            )

        logger.info("Loading RT-DETR-L weights from: %s", self.cfg.weights)
        self._model = RTDETR(str(weights_path))
        logger.info("Model loaded. Classes: %s", CLASS_NAMES)

    def run(
        self,
        frames,   # List[FrameResult] from preprocessor
        output_dir: Optional[str] = None,
    ) -> List[DetectionResult]:
        """
        Run RT-DETR inference on all frames in the manifest.

        Args:
            frames:     List of FrameResult objects from preprocessor.
            output_dir: If provided, saves detections.json here.

        Returns:
            List of DetectionResult, one per frame (including frames with
            zero detections — these are kept so downstream stages can
            correlate by frame_index).
        """
        self._load_model()

        results: List[DetectionResult] = []
        n_frames = len(frames)
        total_boxes = 0

        logger.info("=== Detector.run === %d frames", n_frames)

        for idx, frame in enumerate(frames):
            if not Path(frame.frame_path).exists():
                logger.warning(
                    "Frame %d: image file not found at %s — skipping",
                    frame.frame_index, frame.frame_path,
                )
                continue

            # Run inference — Ultralytics handles resize, normalise, forward pass
            preds = self._model.predict(
                source=frame.frame_path,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                imgsz=self.cfg.imgsz,
                device=self.cfg.device,
                verbose=False,
            )

            # Ultralytics returns a list of Results, one per image
            pred = preds[0]
            img_h, img_w = pred.orig_shape

            boxes: List[BoundingBox] = []
            if pred.boxes is not None and len(pred.boxes) > 0:
                for box in pred.boxes:
                    coords = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
                    cls_id = int(box.cls[0].item())
                    conf   = float(box.conf[0].item())

                    # Guard against unexpected class IDs
                    if cls_id >= len(CLASS_NAMES):
                        logger.warning(
                            "Frame %d: unknown class_id %d — skipping box",
                            frame.frame_index, cls_id,
                        )
                        continue

                    cls_name = CLASS_NAMES[cls_id]
                    boxes.append(BoundingBox(
                        x1=coords[0], y1=coords[1],
                        x2=coords[2], y2=coords[3],
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        severity_prior=CLASS_SEVERITY_PRIOR[cls_name],
                    ))

            total_boxes += len(boxes)

            result = DetectionResult(
                frame_path=frame.frame_path,
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                latitude=frame.latitude,
                longitude=frame.longitude,
                wall_time=frame.wall_time.isoformat() if frame.wall_time else None,
                lighting=frame.lighting,
                sun_elevation=frame.sun_elevation,
                image_width=img_w,
                image_height=img_h,
                boxes=boxes,
            )
            results.append(result)

            if idx % 10 == 0 or len(boxes) > 0:
                logger.info(
                    "Frame %d/%d | t=%.1fs | %d detection(s) %s",
                    idx + 1, n_frames,
                    frame.timestamp_s,
                    len(boxes),
                    [f"{b.class_name}({b.confidence:.2f})" for b in boxes],
                )

        # Summary
        frames_with_detections = sum(1 for r in results if r.has_detections)
        logger.info("=== Detection complete ===")
        logger.info("  Frames processed     : %d", len(results))
        logger.info("  Frames with hits     : %d / %d (%.0f%%)",
                    frames_with_detections, len(results),
                    100 * frames_with_detections / max(len(results), 1))
        logger.info("  Total boxes          : %d", total_boxes)

        # Class breakdown
        class_counts: dict = {}
        for r in results:
            for b in r.boxes:
                class_counts[b.class_name] = class_counts.get(b.class_name, 0) + 1
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            logger.info("    %-25s : %d", cls, cnt)

        if output_dir:
            self.save_detections(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_detections(results: List[DetectionResult], output_dir: str) -> str:
        """Save detection results to detections.json in output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "detections.json"

        payload = {
            "n_frames":   len(results),
            "n_boxes":    sum(r.n_detections for r in results),
            "class_names": CLASS_NAMES,
            "frames":     [r.to_dict() for r in results],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Detections saved: %s (%d frames, %d boxes)",
                    out_path, payload["n_frames"], payload["n_boxes"])
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
                    x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                    class_id=b["class_id"],
                    class_name=b["class_name"],
                    confidence=b["confidence"],
                    severity_prior=b.get("severity_prior", "S2"),
                )
                for b in fr["boxes"]
            ]
            results.append(DetectionResult(
                frame_path=fr["frame_path"],
                frame_index=fr["frame_index"],
                timestamp_s=fr["timestamp_s"],
                latitude=fr.get("latitude"),
                longitude=fr.get("longitude"),
                wall_time=fr.get("wall_time"),
                lighting=fr.get("lighting", "unknown"),
                sun_elevation=fr.get("sun_elevation"),
                image_width=fr["image_width"],
                image_height=fr["image_height"],
                boxes=boxes,
            ))

        logger.info("Loaded detections: %s (%d frames)", detections_path, len(results))
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
        description="CRIS Stage 2 — RT-DETR-L road damage detector."
    )
    parser.add_argument("--manifest",  required=True,
                        help="manifest.json from preprocessor (Stage 1)")
    parser.add_argument("--weights",   default="ml/weights/rtdetr_l_rdd2022.pt",
                        help="Path to RT-DETR-L .pt weights file")
    parser.add_argument("--output",    required=True,
                        help="Output directory for detections.json")
    parser.add_argument("--conf",      type=float, default=0.35,
                        help="Confidence threshold (default 0.35)")
    parser.add_argument("--iou",       type=float, default=0.6,
                        help="IoU threshold (default 0.6)")
    parser.add_argument("--device",    default="cpu",
                        help="Device: cpu | cuda | cuda:0 (default: cpu)")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Load preprocessor manifest
    from pipeline.preprocessor import Preprocessor
    frames = Preprocessor.load_manifest(args.manifest)
    logger.info("Loaded %d frames from manifest", len(frames))

    cfg = DetectorConfig(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    detector = Detector(cfg)
    detector.run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()