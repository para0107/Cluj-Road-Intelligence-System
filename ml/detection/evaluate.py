"""
ml/detection/evaluate.py

PURPOSE:
    Evaluates a trained RT-DETR-L checkpoint on the val and/or test sets.
    Reports per-class AP@50, overall mAP50, mAP50-95, Precision, Recall.
    Supports checkpoint comparison (best.pt vs last.pt) and results.csv merging.

REFERENCES:
    - RT-DETR: Zhao et al., 2024 — arxiv.org/abs/2304.08069
    - COCO mAP definition: Lin et al., 2014 — arxiv.org/abs/1405.0312
    - PASCAL VOC AP protocol: Everingham et al., 2010 — doi:10.1007/s11263-009-0275-4
    - TTA: RT-DETRv2 bag-of-freebies — arxiv.org/abs/2407.17140

USAGE:
    # Evaluate best.pt (auto-detected from your Kaggle run folder)
    python ml/detection/evaluate.py

    # Evaluate a specific checkpoint
    python ml/detection/evaluate.py --weights runs/detect/rtdetr_road/Kaggle/Phase2/Run3_42_60/best.pt

    # Evaluate on test split
    python ml/detection/evaluate.py --split test

    # Evaluate with TTA (~3x slower, typically +0.5-1 mAP50)
    python ml/detection/evaluate.py --tta

    # Full report: val + val-TTA + test + checkpoint comparison
    python ml/detection/evaluate.py --full

    # Compare best.pt vs last.pt
    python ml/detection/evaluate.py --compare

    # Merge all results.csv files across all Kaggle runs into one xlsx
    python ml/detection/evaluate.py --merge-results

INPUT:
    runs/detect/rtdetr_road/Kaggle/Phase2/Run3_42_60/best.pt   (auto-detected)
    data/detection/dataset.yaml

OUTPUT:
    ml/evaluation/eval_{split}[_tta]_{timestamp}.json     — metrics per run
    ml/evaluation/eval_{split}[_tta]_{timestamp}.log      — full log file
    ml/evaluation/all_runs_results.xlsx                   — merged CSVs (--merge-results)

METRICS:
    mAP50      — mean AP at IoU=0.50 across all classes (COCO/VOC protocol)
    mAP50-95   — mean AP averaged over IoU thresholds 0.50:0.05:0.95 (strict)
    AP50       — per-class AP at IoU=0.50 (what Ultralytics box.ap stores)
    Precision  — TP / (TP + FP) at the operating confidence threshold
    Recall     — TP / (TP + FN) at the operating confidence threshold

WHAT TO LOOK FOR:
    - patch_deterioration AP = 0.0 is EXPECTED — no training samples exist yet
    - Recall < Precision is a concern for CRIS: missed damage costs more
      than false positives in a municipal survey context
    - Operational conf threshold for CRIS is 0.35 (balances P/R for deployment)
    - Evaluation conf=0.001 is standard for mAP computation (scores all detections)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_YAML   = ROOT / "data" / "detection" / "dataset.yaml"
WEIGHTS_DIR = ROOT / "ml" / "weights"
EVAL_DIR    = ROOT / "ml" / "evaluation"

# Actual Kaggle run structure from your project:
# runs/detect/rtdetr_road/Kaggle/Phase2/Run3_42_60/best.pt
KAGGLE_PHASE2_DIR = ROOT / "runs" / "detect" / "rtdetr_road" / "Kaggle" / "Phase2"
BEST_RUN_DIR      = KAGGLE_PHASE2_DIR / "Run3_42_60"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]


# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logger(log_path: Path) -> logging.Logger:
    """
    Configure standard logging to both stdout and a file.
    Using stdlib logging (not loguru) for Kaggle compatibility.
    File is written to ml/evaluation/ so it survives after the run.
    """
    logger = logging.getLogger("cris.evaluate")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler — always write, so you can retrieve logs later
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Weight resolution ──────────────────────────────────────────────────────────
def find_best_weights() -> Path | None:
    """
    Auto-detect weights in priority order:
        best.pt  (from Run3_42_60 — highest mAP50 checkpoint saved by Ultralytics)
        last.pt  (from Run3_42_60 — final epoch checkpoint)
        rtdetr_l_rdd2022.pt  (in ml/weights/ — fallback)

    No swa.pt exists in this project (SWA was not applied in the final run).

    Returns:
        Path to the first existing candidate, or None if none found.
    """
    candidates = [
        BEST_RUN_DIR / "best.pt",
        BEST_RUN_DIR / "last.pt",
        WEIGHTS_DIR / "rtdetr_l_rdd2022.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ── Core evaluation ────────────────────────────────────────────────────────────
def run_eval(
    weights:   str   = None,
    split:     str   = "val",
    tta:       bool  = False,
    save_json: bool  = True,
    conf:      float = 0.001,
    iou:       float = 0.6,
    logger:    logging.Logger = None,
) -> dict:
    """
    Run Ultralytics model.val() on the specified split and extract metrics.

    Confidence is set to 0.001 by default — this is the standard value for
    mAP computation because it scores all detections before the PR curve
    is built. For operational deployment use conf=0.35.

    Per-class AP is AP@IoU=0.50 as stored in results.box.ap by Ultralytics 8.2.x.
    This follows the PASCAL VOC AP protocol (Everingham et al., 2010).

    Args:
        weights   : path to .pt file, or None to auto-detect
        split     : "val" or "test"
        tta       : enable Test Time Augmentation (flip + rotate, averaged)
        save_json : whether to write eval_{timestamp}.json to ml/evaluation/
        conf      : confidence threshold for NMS during evaluation
        iou       : IoU threshold for NMS during evaluation
        logger    : logging.Logger instance (created internally if None)

    Returns:
        dict with all metrics and metadata
    """
    # ── Setup logger if not provided ───────────────────────────────────────────
    if logger is None:
        ts_log = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = EVAL_DIR / f"eval_{split}{'_tta' if tta else ''}_{ts_log}.log"
        logger = setup_logger(log_path)

    # ── Resolve weights ────────────────────────────────────────────────────────
    if weights is None:
        weights_path = find_best_weights()
        if weights_path is None:
            logger.error("No trained weights found. Searched:")
            logger.error(f"  {BEST_RUN_DIR / 'best.pt'}")
            logger.error(f"  {BEST_RUN_DIR / 'last.pt'}")
            logger.error(f"  {WEIGHTS_DIR / 'rtdetr_l_rdd2022.pt'}")
            logger.error("Pass --weights explicitly or copy your checkpoint to ml/weights/.")
            sys.exit(1)
    else:
        weights_path = Path(weights)
        if not weights_path.exists():
            logger.error(f"Weights not found: {weights_path}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("RT-DETR-L Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Weights  : {weights_path}")
    logger.info(f"  Dataset  : {DATA_YAML}")
    logger.info(f"  Split    : {split}")
    logger.info(f"  TTA      : {tta}")
    logger.info(f"  conf     : {conf}  (use 0.001 for mAP, 0.35 for deployment)")
    logger.info(f"  iou      : {iou}")

    if not DATA_YAML.exists():
        logger.error(f"dataset.yaml not found: {DATA_YAML}")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU found — evaluation will be slow on CPU.")
    else:
        logger.info(f"  Device   : GPU {torch.cuda.get_device_name(0)}")

    # ── Load model ─────────────────────────────────────────────────────────────
    # Import here so the module can be imported without ultralytics installed
    # (e.g. for --merge-results only)
    try:
        from ultralytics import RTDETR
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics==8.2.18")
        sys.exit(1)

    logger.info("Loading model...")
    model = RTDETR(str(weights_path))

    # ── Run validation ─────────────────────────────────────────────────────────
    logger.info(f"Running model.val() on '{split}' split...")
    if tta:
        logger.info("TTA enabled: flip + rotate augmentations, predictions averaged.")
        logger.info("Expect ~3x longer runtime.")

    results = model.val(
        data      = str(DATA_YAML),
        split     = split,
        imgsz     = 640,
        batch     = 8,
        conf      = conf,
        iou       = iou,
        device    = 0 if torch.cuda.is_available() else "cpu",
        augment   = tta,
        plots     = True,
        save_json = False,
        verbose   = True,
    )

    # ── Extract overall metrics ────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}

    map50     = float(rd.get("metrics/mAP50(B)",     0.0))
    map5095   = float(rd.get("metrics/mAP50-95(B)",  0.0))
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall    = float(rd.get("metrics/recall(B)",    0.0))
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Extract per-class AP@50 ────────────────────────────────────────────────
    # Ultralytics 8.2.x stores AP@0.50 in results.box.ap (one value per class).
    # This follows PASCAL VOC AP protocol (Everingham et al., 2010).
    # Note: this is AP@50, NOT AP@50-95, despite the field name 'ap'.
    per_class_ap50 = {}
    try:
        box = results.box
        for idx, ap_val in zip(box.ap_class_index, box.ap):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class_ap50[name] = round(float(ap_val), 6)
    except Exception as e:
        logger.warning(f"Could not extract per-class AP@50: {e}")
        logger.warning("The results.box attribute may differ across Ultralytics versions.")

    # Fill missing classes with 0.0
    for name in CLASS_NAMES:
        if name not in per_class_ap50:
            per_class_ap50[name] = 0.0

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  {'Metric':<30}  {'Value':>8}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  {'mAP50':<30}  {map50:>8.4f}")
    logger.info(f"  {'mAP50-95':<30}  {map5095:>8.4f}")
    logger.info(f"  {'Precision':<30}  {precision:>8.4f}")
    logger.info(f"  {'Recall':<30}  {recall:>8.4f}")
    logger.info(f"  {'F1':<30}  {f1:>8.4f}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  Per-class AP@50:")
    for name, ap in per_class_ap50.items():
        bar  = "█" * int(ap * 30)
        note = ""
        if name == "patch_deterioration" and ap == 0.0:
            note = "  ← EXPECTED: no training samples"
        elif ap < 0.15:
            note = "  ← low: domain gap / class imbalance"
        logger.info(f"    {name:<28}  {ap:.4f}  {bar}{note}")

    # ── Recall vs Precision note ───────────────────────────────────────────────
    logger.info("")
    if recall < precision:
        logger.warning(
            f"Recall ({recall:.4f}) < Precision ({precision:.4f}). "
            "For CRIS (municipal survey), false negatives cost more than false positives. "
            "Consider lowering conf threshold at deployment (current recommended: 0.35)."
        )
    else:
        logger.info(
            f"Recall ({recall:.4f}) >= Precision ({precision:.4f}). "
            "Good balance for municipal survey use case."
        )

    # ── Grade ──────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("-" * 40)
    if map5095 >= 0.55:
        grade = "EXCELLENT — production ready"
    elif map5095 >= 0.40:
        grade = "GOOD — acceptable for Cluj fine-tuning phase"
    elif map5095 >= 0.25:
        grade = "ACCEPTABLE — PSO tuning or more data needed"
    elif map5095 >= 0.15:
        grade = "WEAK — check training logs for convergence issues"
    else:
        grade = "POOR — model may not have converged"
    logger.info(f"  Grade (mAP50-95={map5095:.4f}): {grade}")
    logger.info("-" * 40)

    # ── Build output dict ──────────────────────────────────────────────────────
    output = {
        "timestamp":       datetime.now().isoformat(),
        "weights":         str(weights_path),
        "split":           split,
        "tta":             tta,
        "conf":            conf,
        "iou":             iou,
        "mAP50":           round(map50, 6),
        "mAP50-95":        round(map5095, 6),
        "precision":       round(precision, 6),
        "recall":          round(recall, 6),
        "F1":              round(f1, 6),
        "per_class_AP50":  per_class_ap50,
        "note": (
            "per_class_AP50 values are AP@IoU=0.50 as stored in "
            "results.box.ap by Ultralytics 8.2.x (PASCAL VOC protocol). "
            "mAP50-95 is the strict COCO metric averaged over IoU 0.50:0.05:0.95."
        ),
    }

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EVAL_DIR / f"eval_{split}{'_tta' if tta else ''}_{ts}.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"  Saved JSON → {out_path}")

    return output


# ── Checkpoint comparison ──────────────────────────────────────────────────────
def compare_checkpoints(logger: logging.Logger) -> None:
    """
    Evaluate best.pt and last.pt from Run3_42_60 and rank by mAP50-95.
    swa.pt is not included — it was not produced in this training run.

    Saves a comparison JSON to ml/evaluation/checkpoint_comparison_{timestamp}.json.
    """
    candidates = {
        "best.pt": BEST_RUN_DIR / "best.pt",
        "last.pt": BEST_RUN_DIR / "last.pt",
    }

    scores = {}
    full_results = {}

    for label, path in candidates.items():
        if not path.exists():
            logger.info(f"  {label}: not found at {path} — skipping")
            continue
        logger.info(f"\nEvaluating {label} on val set...")
        r = run_eval(weights=str(path), split="val", save_json=False, logger=logger)
        scores[label] = r["mAP50-95"]
        full_results[label] = r

    if not scores:
        logger.error("No checkpoints found for comparison.")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECKPOINT COMPARISON (val set, mAP50-95)")
    logger.info("=" * 60)
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        logger.info(f"  {label:<12}  mAP50-95={score:.4f}  {bar}")

    winner = max(scores, key=scores.get)
    logger.info("")
    logger.info(f"  Winner: {winner}  (mAP50-95={scores[winner]:.4f})")
    logger.info(f"  Recommended checkpoint for pipeline: {BEST_RUN_DIR / winner}")

    # Save comparison
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"checkpoint_comparison_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "winner":    winner,
        "scores":    scores,
        "details":   full_results,
    }, indent=2))
    logger.info(f"  Saved comparison → {out_path}")


# ── Results CSV merge ──────────────────────────────────────────────────────────
def merge_results_csvs(logger: logging.Logger) -> None:
    """
    Finds all results.csv files across all Kaggle run subfolders:
        runs/detect/rtdetr_road/Kaggle/Phase2/Run*/results.csv
        runs/detect/rtdetr_road/Kaggle/Phase2/*/results.csv  (PSO runs)

    Merges them into a single ml/evaluation/all_runs_results.xlsx with:
        - One sheet per run subfolder
        - A 'Summary' sheet with final-epoch metrics from each run side by side

    This gives you the complete training history across all sessions
    for plotting and comparison.

    NOTE: Works with actual files on disk. Will warn if no CSVs are found.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Run: pip install pandas openpyxl")
        return

    # Find all results.csv files under the Kaggle Phase2 directory
    csv_files = sorted(KAGGLE_PHASE2_DIR.rglob("results.csv"))

    # Also check the top-level rtdetr_road folder (baseline run CSV)
    top_csv = ROOT / "runs" / "detect" / "rtdetr_road" / "results.csv"
    if top_csv.exists():
        csv_files = [top_csv] + list(csv_files)

    if not csv_files:
        logger.warning(f"No results.csv files found under {KAGGLE_PHASE2_DIR}")
        logger.warning("Make sure your Kaggle run outputs are downloaded locally.")
        return

    logger.info(f"Found {len(csv_files)} results.csv file(s):")
    for f in csv_files:
        logger.info(f"  {f.relative_to(ROOT)}")

    out_path = EVAL_DIR / "all_runs_results.xlsx"

    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl not installed. Run: pip install openpyxl")
        return

    summary_rows = []

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for csv_path in csv_files:
            # Build a short sheet name from the folder structure
            # e.g. "Run3_42_60" or "baseline"
            rel = csv_path.relative_to(ROOT / "runs" / "detect" / "rtdetr_road")
            parts = rel.parts  # e.g. ('Kaggle', 'Phase2', 'Run3_42_60', 'results.csv')
            sheet_name = "_".join(parts[:-1]) if len(parts) > 1 else "baseline"
            sheet_name = sheet_name[:31]  # Excel sheet name limit

            try:
                df = pd.read_csv(csv_path)
                # Strip whitespace from column names (Ultralytics CSVs often have leading spaces)
                df.columns = df.columns.str.strip()
                df.insert(0, "run", sheet_name)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"  Written sheet: {sheet_name}  ({len(df)} epochs)")

                # Collect last row for summary
                if len(df) > 0:
                    last = df.iloc[-1].to_dict()
                    last["run"] = sheet_name
                    last["csv_path"] = str(csv_path.relative_to(ROOT))
                    summary_rows.append(last)

            except Exception as e:
                logger.warning(f"  Could not read {csv_path}: {e}")

        # Write summary sheet
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Move 'run' and 'csv_path' to front
            cols = ["run", "csv_path"] + [
                c for c in summary_df.columns if c not in ("run", "csv_path")
            ]
            summary_df = summary_df[cols]
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            logger.info(f"  Written Summary sheet ({len(summary_rows)} runs)")

    logger.info(f"\nMerged results saved → {out_path}")
    logger.info("Open in Excel/LibreOffice to compare metrics across all runs.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RT-DETR-L checkpoint — CRIS project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/detection/evaluate.py
  python ml/detection/evaluate.py --split test
  python ml/detection/evaluate.py --tta
  python ml/detection/evaluate.py --full
  python ml/detection/evaluate.py --compare
  python ml/detection/evaluate.py --merge-results
  python ml/detection/evaluate.py --weights runs/detect/rtdetr_road/Kaggle/Phase2/Run3_42_60/best.pt
        """,
    )
    p.add_argument("--weights",        type=str,   default=None,
                   help="Path to .pt weights (default: auto-detect from Run3_42_60)")
    p.add_argument("--split",          type=str,   default="val",
                   choices=["val", "test"],
                   help="Dataset split to evaluate (default: val)")
    p.add_argument("--tta",            action="store_true",
                   help="Enable Test Time Augmentation (~3x slower, +0.5-1 mAP50)")
    p.add_argument("--full",           action="store_true",
                   help="Full report: val + val-TTA + test + checkpoint comparison")
    p.add_argument("--compare",        action="store_true",
                   help="Compare best.pt vs last.pt from Run3_42_60")
    p.add_argument("--merge-results",  action="store_true",
                   help="Merge all results.csv files into ml/evaluation/all_runs_results.xlsx")
    p.add_argument("--conf",           type=float, default=0.001,
                   help="Confidence threshold for eval NMS (default: 0.001 for mAP)")
    p.add_argument("--iou",            type=float, default=0.6,
                   help="IoU threshold for eval NMS (default: 0.6)")
    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Set up a shared logger for this session
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = (
        "full"    if args.full           else
        "compare" if args.compare        else
        "merge"   if args.merge_results  else
        f"{args.split}{'_tta' if args.tta else ''}"
    )
    log_path = EVAL_DIR / f"eval_{mode_tag}_{ts}.log"
    logger   = setup_logger(log_path)
    logger.info(f"Log file: {log_path}")

    if args.merge_results:
        merge_results_csvs(logger)

    elif args.compare:
        compare_checkpoints(logger)

    elif args.full:
        logger.info("Full Evaluation Report")
        logger.info("=" * 60)

        logger.info("\n[1/4] Val set — standard (conf=0.001)")
        run_eval(weights=args.weights, split="val",  tta=False,
                 conf=args.conf, iou=args.iou, logger=logger)

        logger.info("\n[2/4] Val set — TTA (flip+rotate)")
        run_eval(weights=args.weights, split="val",  tta=True,
                 conf=args.conf, iou=args.iou, logger=logger)

        logger.info("\n[3/4] Test set — standard")
        run_eval(weights=args.weights, split="test", tta=False,
                 conf=args.conf, iou=args.iou, logger=logger)

        logger.info("\n[4/4] Checkpoint comparison (best.pt vs last.pt)")
        compare_checkpoints(logger)

    else:
        run_eval(
            weights   = args.weights,
            split     = args.split,
            tta       = args.tta,
            conf      = args.conf,
            iou       = args.iou,
            logger    = logger,
        )