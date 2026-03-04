
"""
ml/detection/evaluate.py

PURPOSE:
    Evaluates a trained RT-DETR-L checkpoint on the val and test sets.
    Reports per-class AP, overall mAP50, mAP50-95, and confusion matrix.
    Used after training to assess model quality before running PSO.

USAGE:
    # Evaluate best checkpoint (default)
    python ml/detection/evaluate.py

    # Evaluate SWA weights
    python ml/detection/evaluate.py --weights runs/detect/rtdetr_road/weights/swa.pt

    # Evaluate on test set instead of val
    python ml/detection/evaluate.py --split test

    # Evaluate with TTA (Test Time Augmentation) — slower but +0.5-1 mAP
    python ml/detection/evaluate.py --tta

    # Full report: val + test + TTA + confusion matrix
    python ml/detection/evaluate.py --full

INPUT:
    runs/detect/rtdetr_road/weights/best.pt   (or --weights path)
    data/detection/dataset.yaml

OUTPUT:
    Printed metrics table (per-class AP, mAP50, mAP50-95)
    ml/evaluation/eval_{timestamp}.json       (full metrics, saved for comparison)
    runs/detect/rtdetr_road/confusion_matrix.png  (if --full)

METRICS EXPLAINED:
    mAP50      : mean AP at IoU=0.50 (lenient, good for rough localization)
    mAP50-95   : mean AP averaged over IoU 0.50:0.95 (strict, main metric)
    AP per class: how well each damage type is detected independently

WHAT TO LOOK FOR:
    - mAP50-95 > 0.40 after 100 epochs = good baseline
    - mAP50-95 > 0.50 = strong model, ready for production
    - patch_deterioration AP = 0 is expected (no training samples)
    - If alligator_crack AP << others: class imbalance issue,
      consider increasing cls loss weight in PSO search
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from ultralytics import RTDETR

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_YAML   = ROOT / "data" / "detection" / "dataset.yaml"
WEIGHTS_DIR = ROOT / "ml" / "weights"
EVAL_DIR    = ROOT / "ml" / "evaluation"
RUN_DIR     = ROOT / "runs" / "detect" / "rtdetr_road"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]


def find_best_weights() -> Path:
    """
    Return the best available weights in priority order:
    swa.pt > best.pt > rtdetr_l_rdd2022.pt
    """
    candidates = [
        RUN_DIR / "weights" / "swa.pt",
        RUN_DIR / "weights" / "best.pt",
        WEIGHTS_DIR / "rtdetr_l_rdd2022.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_eval(
    weights:    str  = None,
    split:      str  = "val",
    tta:        bool = False,
    save_json:  bool = True,
    conf:       float = 0.001,
    iou:        float = 0.6,
) -> dict:
    """
    Run evaluation and return metrics dict.

    Args:
        weights   : path to .pt checkpoint, or None to auto-detect
        split     : "val" or "test"
        tta       : enable Test Time Augmentation
        save_json : save results to ml/evaluation/
        conf      : confidence threshold for NMS
        iou       : IoU threshold for NMS

    Returns:
        dict with mAP50, mAP50-95, per-class AP, and raw results
    """
    # ── Resolve weights ────────────────────────────────────────────────────────
    if weights is None:
        weights_path = find_best_weights()
        if weights_path is None:
            logger.error("No trained weights found.")
            logger.error("Expected one of:")
            logger.error(f"  {RUN_DIR / 'weights' / 'swa.pt'}")
            logger.error(f"  {RUN_DIR / 'weights' / 'best.pt'}")
            logger.error(f"  {WEIGHTS_DIR / 'rtdetr_l_rdd2022.pt'}")
            sys.exit(1)
    else:
        weights_path = Path(weights)
        if not weights_path.exists():
            logger.error(f"Weights not found: {weights_path}")
            sys.exit(1)

    logger.info("── RT-DETR-L Evaluation ───────────────────────────────────")
    logger.info(f"  Weights  : {weights_path}")
    logger.info(f"  Dataset  : {DATA_YAML}")
    logger.info(f"  Split    : {split}")
    logger.info(f"  TTA      : {tta}")
    logger.info(f"  conf     : {conf}   iou: {iou}")

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU — evaluation will be slow on CPU.")

    # ── Load model and evaluate ────────────────────────────────────────────────
    model = RTDETR(str(weights_path))

    results = model.val(
        data    = str(DATA_YAML),
        split   = split,
        imgsz   = 640,
        batch   = 8,
        conf    = conf,
        iou     = iou,
        device  = 0 if torch.cuda.is_available() else "cpu",
        augment = tta,           # TTA flag
        plots   = True,
        save_json = False,
        verbose = True,
    )

    # ── Extract metrics ────────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}

    map50    = float(rd.get("metrics/mAP50(B)",    0.0))
    map5095  = float(rd.get("metrics/mAP50-95(B)", 0.0))
    precision= float(rd.get("metrics/precision(B)",0.0))
    recall   = float(rd.get("metrics/recall(B)",   0.0))

    # Per-class AP (Ultralytics stores these in results.ap_class_index + results.ap)
    per_class = {}
    if hasattr(results, "ap_class_index") and hasattr(results, "ap"):
        for idx, ap_val in zip(results.ap_class_index, results.ap):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class[name] = float(ap_val)
    # Fill missing classes with 0
    for name in CLASS_NAMES:
        if name not in per_class:
            per_class[name] = 0.0

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("\n── Results ────────────────────────────────────────────────")
    logger.info(f"  {'Metric':<30}  {'Value':>8}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  {'mAP50':<30}  {map50:>8.4f}")
    logger.info(f"  {'mAP50-95':<30}  {map5095:>8.4f}")
    logger.info(f"  {'Precision':<30}  {precision:>8.4f}")
    logger.info(f"  {'Recall':<30}  {recall:>8.4f}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  Per-class AP50-95:")
    for name, ap in per_class.items():
        bar = "█" * int(ap * 30)
        flag = " ← NO TRAINING DATA" if name == "patch_deterioration" and ap == 0 else ""
        logger.info(f"    {name:<28}  {ap:.4f}  {bar}{flag}")

    # ── Grade ──────────────────────────────────────────────────────────────────
    logger.info(f"\n  {'─'*40}")
    if map5095 >= 0.55:
        grade = "★★★ Excellent — production ready"
    elif map5095 >= 0.45:
        grade = "★★  Good — run PSO to push further"
    elif map5095 >= 0.35:
        grade = "★   Acceptable — PSO tuning needed"
    elif map5095 >= 0.20:
        grade = "⚠   Weak — check training logs for issues"
    else:
        grade = "✗   Poor — model may not have converged"
    logger.info(f"  Grade: {grade}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "timestamp":     datetime.now().isoformat(),
        "weights":       str(weights_path),
        "split":         split,
        "tta":           tta,
        "mAP50":         map50,
        "mAP50-95":      map5095,
        "precision":     precision,
        "recall":        recall,
        "per_class_AP":  per_class,
    }

    if save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EVAL_DIR / f"eval_{split}{'_tta' if tta else ''}_{ts}.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.success(f"\n  Saved → {out_path}")

    return output


def compare_checkpoints():
    """
    Evaluate all saved checkpoints and rank them.
    Useful for choosing whether to use best.pt, swa.pt, or last.pt.
    """
    candidates = {
        "best.pt": RUN_DIR / "weights" / "best.pt",
        "swa.pt":  RUN_DIR / "weights" / "swa.pt",
        "last.pt": RUN_DIR / "weights" / "last.pt",
    }

    results = {}
    for label, path in candidates.items():
        if not path.exists():
            logger.info(f"  {label}: not found — skipping")
            continue
        logger.info(f"\n  Evaluating {label} ...")
        r = run_eval(weights=str(path), split="val", save_json=False)
        results[label] = r["mAP50-95"]

    if not results:
        logger.error("No checkpoints found.")
        return

    logger.info("\n── Checkpoint Comparison ──────────────────────────────────")
    for label, score in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        logger.info(f"  {label:<12}  mAP50-95={score:.4f}  {bar}")

    best = max(results, key=results.get)
    logger.success(f"\n  Winner: {best}  (mAP50-95={results[best]:.4f})")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RT-DETR-L checkpoint"
    )
    p.add_argument("--weights", type=str, default=None,
                   help="Path to .pt weights (default: auto-detect best/swa)")
    p.add_argument("--split",   type=str, default="val",
                   choices=["val", "test"],
                   help="Dataset split to evaluate (default: val)")
    p.add_argument("--tta",     action="store_true",
                   help="Enable Test Time Augmentation (+0.5-1 mAP, 3× slower)")
    p.add_argument("--full",    action="store_true",
                   help="Full report: val + test + TTA + checkpoint comparison")
    p.add_argument("--compare", action="store_true",
                   help="Compare best.pt vs swa.pt vs last.pt")
    p.add_argument("--conf",    type=float, default=0.001,
                   help="Confidence threshold (default: 0.001)")
    p.add_argument("--iou",     type=float, default=0.6,
                   help="IoU threshold (default: 0.6)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        compare_checkpoints()
    elif args.full:
        logger.info("── Full Evaluation Report ─────────────────────────────────")
        logger.info("\n[1/4] Val set (standard)")
        run_eval(weights=args.weights, split="val",  tta=False)
        logger.info("\n[2/4] Val set (TTA)")
        run_eval(weights=args.weights, split="val",  tta=True)
        logger.info("\n[3/4] Test set (standard)")
        run_eval(weights=args.weights, split="test", tta=False)
        logger.info("\n[4/4] Checkpoint comparison")
        compare_checkpoints()
    else:
        run_eval(
            weights   = args.weights,
            split     = args.split,
            tta       = args.tta,
            conf      = args.conf,
            iou       = args.iou,
        )