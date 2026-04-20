"""
ml/detection/evaluate.py

PURPOSE:
    Evaluates a trained RT-DETR-L checkpoint on the val and/or test sets.
    Reports per-class AP@50, overall mAP50, mAP50-95, Precision, Recall, F1.
    Supports checkpoint comparison (best.pt vs last.pt) and results merging.

REFERENCES:
    - RT-DETR: Zhao et al., 2024 — arxiv.org/abs/2304.08069
    - COCO mAP definition: Lin et al., 2014 — arxiv.org/abs/1405.0312
    - PASCAL VOC AP protocol: Everingham et al., 2010 — doi:10.1007/s11263-009-0275-4
    - TTA bag-of-freebies: RT-DETRv2, Lv et al., 2024 — arxiv.org/abs/2407.17140

USAGE:
    python ml/detection/evaluate.py                          # val, auto-detect best.pt
    python ml/detection/evaluate.py --split test             # test split
    python ml/detection/evaluate.py --tta                    # with TTA (~3x slower)
    python ml/detection/evaluate.py --full                   # val + val-TTA + test + compare
    python ml/detection/evaluate.py --compare                # best.pt vs last.pt
    python ml/detection/evaluate.py --merge-results          # merge all results.csv -> xlsx
    python ml/detection/evaluate.py --weights <path>         # explicit checkpoint

INPUT:
    runs/detect/rtdetr_road/final_fine_tune/best.pt   (auto-detected)
    data/detection/dataset.yaml

OUTPUT:
    ml/evaluation/eval_{split}[_tta]_{timestamp}.json   -- metrics per run
    ml/evaluation/eval_{mode}_{timestamp}.log           -- full log (survives Kaggle death)
    ml/evaluation/all_runs_results.xlsx                 -- merged CSVs (--merge-results)

METRICS:
    mAP50     -- mean AP at IoU=0.50 (COCO/VOC protocol, Lin et al. 2014)
    mAP50-95  -- mean AP over IoU 0.50:0.05:0.95 (strict COCO metric)
    AP@50     -- per-class AP at IoU=0.50 (Ultralytics box.ap, VOC protocol)
    Precision -- TP / (TP + FP) at evaluation conf threshold
    Recall    -- TP / (TP + FN) at evaluation conf threshold
    F1        -- harmonic mean of Precision and Recall

NOTES:
    - conf=0.001 is standard for mAP computation (scores all detections for PR curve)
    - conf=0.35 is the recommended operational threshold for CRIS deployment
    - patch_deterioration AP=0.0 is expected: no training samples in RDD2022+Pothole600
    - swa.pt was NOT produced in the final training run -- not included in comparison
    - Recall < Precision is a concern for CRIS: missed road damage costs more
      than false positives in a municipal survey context
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
DATA_YAML    = ROOT / "data" / "detection" / "dataset.yaml"
WEIGHTS_DIR  = ROOT / "ml" / "weights"
EVAL_DIR     = ROOT / "ml" / "evaluation"

# Final consolidated run folder -- best.pt and last.pt live here
BEST_RUN_DIR = ROOT / "runs" / "detect" / "rtdetr_road" / "final_fine_tune"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logger(log_path: Path) -> logging.Logger:
    """
    Configure stdlib logging to both stdout and a .log file.
    Using stdlib (not loguru) for Kaggle compatibility.
    Log file is written to ml/evaluation/ and survives Kaggle session death.
    """
    logger = logging.getLogger("cris.evaluate")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Weight resolution ──────────────────────────────────────────────────────────
def find_best_weights() -> Path | None:
    """
    Auto-detect weights in priority order:
        best.pt             -- highest val mAP50 checkpoint (Ultralytics early stopping)
        last.pt             -- final epoch checkpoint
        rtdetr_l_rdd2022.pt -- fallback in ml/weights/

    swa.pt is NOT included -- it was not produced in this training run.
    Returns the first existing candidate, or None.
    """
    candidates = [
        BEST_RUN_DIR / "best.pt",
        BEST_RUN_DIR / "last.pt",
        WEIGHTS_DIR  / "rtdetr_l_rdd2022.pt",
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
    Run Ultralytics model.val() and extract all metrics.

    conf=0.001 is the standard value for mAP computation -- it scores all
    detections before building the PR curve (PASCAL VOC / COCO protocol).
    For operational CRIS deployment use conf=0.35.

    Per-class values are AP@IoU=0.50 as stored in results.box.ap by
    Ultralytics 8.2.x, following the PASCAL VOC AP protocol
    (Everingham et al., 2010, doi:10.1007/s11263-009-0275-4).

    Args:
        weights   : path to .pt file, or None to auto-detect
        split     : "val" or "test"
        tta       : enable TTA (flip + rotate, predictions averaged)
        save_json : write eval_{timestamp}.json to ml/evaluation/
        conf      : NMS confidence threshold for evaluation
        iou       : NMS IoU threshold for evaluation
        logger    : logging.Logger -- created internally if None

    Returns:
        dict with all metrics and run metadata
    """
    if logger is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = EVAL_DIR / f"eval_{split}{'_tta' if tta else ''}_{ts}.log"
        logger = setup_logger(log_path)

    # ── Resolve weights ────────────────────────────────────────────────────────
    if weights is None:
        weights_path = find_best_weights()
        if weights_path is None:
            logger.error("No trained weights found. Searched:")
            logger.error(f"  {BEST_RUN_DIR / 'best.pt'}")
            logger.error(f"  {BEST_RUN_DIR / 'last.pt'}")
            logger.error(f"  {WEIGHTS_DIR  / 'rtdetr_l_rdd2022.pt'}")
            logger.error("Pass --weights explicitly or copy checkpoint to ml/weights/.")
            sys.exit(1)
    else:
        weights_path = Path(weights)
        if not weights_path.exists():
            logger.error(f"Weights not found: {weights_path}")
            sys.exit(1)

    logger.info("=" * 62)
    logger.info("RT-DETR-L Evaluation  |  CRIS -- Cluj Road Intelligence System")
    logger.info("=" * 62)
    logger.info(f"  Weights  : {weights_path}")
    logger.info(f"  Dataset  : {DATA_YAML}")
    logger.info(f"  Split    : {split}")
    logger.info(f"  TTA      : {tta}")
    logger.info(f"  conf     : {conf}  (0.001 for mAP, 0.35 for deployment)")
    logger.info(f"  iou      : {iou}")

    if not DATA_YAML.exists():
        logger.error(f"dataset.yaml not found: {DATA_YAML}")
        logger.error("On Kaggle: run the dataset.yaml generation cell first.")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU -- evaluation will be slow on CPU.")
    else:
        logger.info(f"  Device   : {torch.cuda.get_device_name(0)}")

    try:
        from ultralytics import RTDETR
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics==8.2.18")
        sys.exit(1)

    logger.info("Loading model...")
    model = RTDETR(str(weights_path))

    logger.info(f"Running model.val() on '{split}' split...")
    if tta:
        logger.info("TTA enabled: flip + rotate, predictions averaged (~3x slower).")

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

    # ── Overall metrics ────────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}

    map50     = float(rd.get("metrics/mAP50(B)",     0.0))
    map5095   = float(rd.get("metrics/mAP50-95(B)",  0.0))
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall    = float(rd.get("metrics/recall(B)",    0.0))
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # ── Per-class AP@50 ────────────────────────────────────────────────────────
    # Ultralytics 8.2.x stores AP@IoU=0.50 in results.box.ap.
    # Follows PASCAL VOC AP protocol (Everingham et al., 2010).
    per_class_ap50 = {}
    try:
        box = results.box
        for idx, ap_val in zip(box.ap_class_index, box.ap):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class_ap50[name] = round(float(ap_val), 6)
    except Exception as e:
        logger.warning(f"Could not extract per-class AP@50: {e}")

    for name in CLASS_NAMES:
        if name not in per_class_ap50:
            per_class_ap50[name] = 0.0

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 62)
    logger.info("RESULTS")
    logger.info("=" * 62)
    logger.info(f"  {'Metric':<30}  {'Value':>8}")
    logger.info(f"  {'-'*42}")
    logger.info(f"  {'mAP50':<30}  {map50:>8.4f}")
    logger.info(f"  {'mAP50-95':<30}  {map5095:>8.4f}")
    logger.info(f"  {'Precision':<30}  {precision:>8.4f}")
    logger.info(f"  {'Recall':<30}  {recall:>8.4f}")
    logger.info(f"  {'F1':<30}  {f1:>8.4f}")
    logger.info(f"  {'-'*42}")
    logger.info(f"  Per-class AP@50  (PASCAL VOC protocol):")
    for name, ap in per_class_ap50.items():
        bar  = "█" * int(ap * 30)
        note = ""
        if name == "patch_deterioration" and ap == 0.0:
            note = "  <- EXPECTED: no training samples in RDD2022+Pothole600"
        elif ap < 0.15:
            note = "  <- low: domain gap likely cause"
        logger.info(f"    {name:<28}  {ap:.4f}  {bar}{note}")

    # ── Recall vs Precision warning ────────────────────────────────────────────
    logger.info("")
    if recall < precision:
        logger.warning(
            f"Recall ({recall:.4f}) < Precision ({precision:.4f}). "
            "For CRIS, false negatives (missed damage) cost more than false positives. "
            "Lower conf at deployment -- recommended threshold: conf=0.35."
        )
    else:
        logger.info(
            f"Recall ({recall:.4f}) >= Precision ({precision:.4f}). "
            "Good balance for municipal road survey use case."
        )

    # ── Grade ──────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("-" * 42)
    if map5095 >= 0.55:
        grade = "EXCELLENT -- production ready"
    elif map5095 >= 0.40:
        grade = "GOOD -- ready for Cluj fine-tuning phase"
    elif map5095 >= 0.25:
        grade = "ACCEPTABLE -- more data or PSO tuning needed"
    elif map5095 >= 0.15:
        grade = "WEAK -- check training convergence"
    else:
        grade = "POOR -- model likely did not converge"
    logger.info(f"  Grade (mAP50-95={map5095:.4f}): {grade}")
    logger.info("-" * 42)

    # ── Build output dict ──────────────────────────────────────────────────────
    output = {
        "timestamp":      datetime.now().isoformat(),
        "weights":        str(weights_path),
        "split":          split,
        "tta":            tta,
        "conf":           conf,
        "iou":            iou,
        "mAP50":          round(map50,     6),
        "mAP50-95":       round(map5095,   6),
        "precision":      round(precision, 6),
        "recall":         round(recall,    6),
        "F1":             round(f1,        6),
        "per_class_AP50": per_class_ap50,
        "note": (
            "per_class_AP50 = AP@IoU=0.50 per Ultralytics 8.2.x results.box.ap "
            "(PASCAL VOC protocol, Everingham et al. 2010). "
            "mAP50-95 is the strict COCO metric (Lin et al. 2014)."
        ),
    }

    if save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EVAL_DIR / f"eval_{split}{'_tta' if tta else ''}_{ts}.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"  Saved JSON  -> {out_path}")

    return output


# ── Checkpoint comparison ──────────────────────────────────────────────────────
def compare_checkpoints(logger: logging.Logger) -> None:
    """
    Evaluate best.pt and last.pt from final_fine_tune/ on the val set
    and rank by mAP50-95.

    swa.pt is excluded -- not produced in this training run.
    Saves a comparison JSON to ml/evaluation/.
    """
    candidates = {
        "best.pt": BEST_RUN_DIR / "best.pt",
        "last.pt": BEST_RUN_DIR / "last.pt",
    }

    scores       = {}
    full_results = {}

    for label, path in candidates.items():
        if not path.exists():
            logger.info(f"  {label}: not found at {path} -- skipping")
            continue
        logger.info(f"\nEvaluating {label} on val set...")
        r = run_eval(weights=str(path), split="val", save_json=False, logger=logger)
        scores[label]       = r["mAP50-95"]
        full_results[label] = r

    if not scores:
        logger.error("No checkpoints found for comparison.")
        return

    logger.info("")
    logger.info("=" * 62)
    logger.info("CHECKPOINT COMPARISON  (val set, ranked by mAP50-95)")
    logger.info("=" * 62)
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        logger.info(f"  {label:<12}  mAP50-95={score:.4f}  {bar}")

    winner = max(scores, key=scores.get)
    logger.info("")
    logger.info(f"  Winner : {winner}  (mAP50-95={scores[winner]:.4f})")
    logger.info(f"  Use this checkpoint for the CRIS inference pipeline:")
    logger.info(f"    {BEST_RUN_DIR / winner}")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"checkpoint_comparison_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp":   datetime.now().isoformat(),
        "winner":      winner,
        "winner_path": str(BEST_RUN_DIR / winner),
        "scores":      scores,
        "details":     full_results,
    }, indent=2))
    logger.info(f"  Saved comparison -> {out_path}")


# ── Results CSV merge ──────────────────────────────────────────────────────────
def merge_results_csvs(logger: logging.Logger) -> None:
    """
    Find all results.csv files under runs/detect/rtdetr_road/ and merge
    them into ml/evaluation/all_runs_results.xlsx.

    Sheet layout:
        One sheet per run subfolder (named after the folder)
        Summary sheet -- final epoch metrics from every run side by side

    Reads actual files from disk. Warns if no CSVs are found.
    """
    try:
        import pandas as pd
        import openpyxl  # noqa: F401
    except ImportError:
        logger.error("Missing deps: pip install pandas openpyxl")
        return

    rtdetr_root = ROOT / "runs" / "detect" / "rtdetr_road"
    csv_files   = sorted(rtdetr_root.rglob("results.csv"))

    if not csv_files:
        logger.warning(f"No results.csv files found under {rtdetr_root}")
        logger.warning("Download your Kaggle run outputs locally first.")
        return

    logger.info(f"Found {len(csv_files)} results.csv file(s):")
    for f in csv_files:
        logger.info(f"  {f.relative_to(ROOT)}")

    out_path     = EVAL_DIR / "all_runs_results.xlsx"
    summary_rows = []

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for csv_path in csv_files:
            sheet_name = csv_path.parent.name[:31]  # Excel 31-char sheet name limit
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                df.insert(0, "run", sheet_name)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"  Written sheet '{sheet_name}'  ({len(df)} epochs)")

                if len(df) > 0:
                    last = df.iloc[-1].to_dict()
                    last["run"]      = sheet_name
                    last["csv_path"] = str(csv_path.relative_to(ROOT))
                    summary_rows.append(last)

            except Exception as e:
                logger.warning(f"  Could not read {csv_path}: {e}")

        if summary_rows:
            import pandas as pd
            summary_df = pd.DataFrame(summary_rows)
            cols = ["run", "csv_path"] + [
                c for c in summary_df.columns if c not in ("run", "csv_path")
            ]
            summary_df[cols].to_excel(writer, sheet_name="Summary", index=False)
            logger.info(f"  Written 'Summary' sheet  ({len(summary_rows)} runs)")

    logger.info(f"\nMerged results -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RT-DETR-L -- CRIS project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python ml/detection/evaluate.py
  python ml/detection/evaluate.py --split test
  python ml/detection/evaluate.py --tta
  python ml/detection/evaluate.py --full
  python ml/detection/evaluate.py --compare
  python ml/detection/evaluate.py --merge-results
  python ml/detection/evaluate.py --weights runs/detect/rtdetr_road/final_fine_tune/best.pt
        """,
    )
    p.add_argument("--weights",       type=str,   default=None,
                   help="Path to .pt weights (default: auto-detect from final_fine_tune/)")
    p.add_argument("--split",         type=str,   default="val",
                   choices=["val", "test"],
                   help="Dataset split (default: val)")
    p.add_argument("--tta",           action="store_true",
                   help="Enable TTA -- flip+rotate, averaged (~3x slower)")
    p.add_argument("--full",          action="store_true",
                   help="Full report: val + val-TTA + test + checkpoint comparison")
    p.add_argument("--compare",       action="store_true",
                   help="Compare best.pt vs last.pt from final_fine_tune/")
    p.add_argument("--merge-results", action="store_true",
                   help="Merge all results.csv -> ml/evaluation/all_runs_results.xlsx")
    p.add_argument("--conf",          type=float, default=0.001,
                   help="Confidence threshold (default: 0.001 for mAP computation)")
    p.add_argument("--iou",           type=float, default=0.6,
                   help="IoU threshold (default: 0.6)")
    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = (
        "full"    if args.full           else
        "compare" if args.compare        else
        "merge"   if args.merge_results  else
        f"{args.split}{'_tta' if args.tta else ''}"
    )
    log_path = EVAL_DIR / f"eval_{mode_tag}_{ts}.log"
    logger   = setup_logger(log_path)
    logger.info(f"Log -> {log_path}")

    if args.merge_results:
        merge_results_csvs(logger)

    elif args.compare:
        compare_checkpoints(logger)

    elif args.full:
        logger.info("Full Evaluation Report")
        logger.info("=" * 62)
        logger.info("\n[1/4] Val set -- standard (conf=0.001)")
        run_eval(weights=args.weights, split="val",  tta=False,
                 conf=args.conf, iou=args.iou, logger=logger)
        logger.info("\n[2/4] Val set -- TTA")
        run_eval(weights=args.weights, split="val",  tta=True,
                 conf=args.conf, iou=args.iou, logger=logger)
        logger.info("\n[3/4] Test set -- standard")
        run_eval(weights=args.weights, split="test", tta=False,
                 conf=args.conf, iou=args.iou, logger=logger)
        logger.info("\n[4/4] Checkpoint comparison")
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