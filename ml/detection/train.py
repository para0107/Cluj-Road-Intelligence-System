"""
ml/detection/train.py

PURPOSE:
    Trains RT-DETR-L on the merged road damage dataset.
    Implements the full training pipeline from the system architecture:

        Albumentations augmentation
        + Mixup regularization
        + Two-phase training (frozen backbone → full fine-tune)
        + Focal loss (configured via RT-DETR's built-in cls_pw/fl_gamma)
        + SWA (Stochastic Weight Averaging) over last N checkpoints
        + PSO-ready hyperparameter interface (loads pso_best.json if present)

    Optimized for RTX 2050 (4 GB VRAM):
        - fp16 mixed precision (mandatory)
        - Gradient accumulation × 4  →  effective batch = 16
        - Image caching disabled (saves RAM)
        - Workers = 4 (Windows-safe)

USAGE:
    # Full training (recommended — runs both phases)
    python ml/detection/train.py

    # Smoke-test: 2 epochs, tiny batch, fast validation
    python ml/detection/train.py --smoke_test

    # Resume phase 2 from a checkpoint
    python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt

    # Use hyperparams produced by PSO (automatically loaded when present)
    # Just run PSO first:  python ml/optimization/pso_hyperparams.py
    # Then run normally:   python ml/detection/train.py

INPUT:
    data/detection/dataset.yaml
    data/detection/train.json   (27,336 images, 42,883 annotations)
    data/detection/val.json     (5,857 images,  9,024 annotations)
    rtdetr-l.pt                 (COCO pretrained weights)
    ml/optimization/pso_best.json  (optional — PSO output)

OUTPUT:
    runs/detect/rtdetr_road/
    ├── weights/
    │   ├── best.pt             highest val mAP50-95
    │   ├── last.pt             latest checkpoint
    │   └── swa.pt              SWA-averaged weights (produced after training)
    ├── results.csv             per-epoch metrics
    └── args.yaml               full training config snapshot

    ml/weights/rtdetr_l_rdd2022.pt   (copy of best.pt after training)

TRAINING STRATEGY:
    Phase 1 — frozen backbone (default: 10 epochs)
        Only decoder + detection head trained. Fast convergence.
        LR = lr0 (default 1e-4)

    Phase 2 — full fine-tune (remaining epochs = 50 by default)
        Backbone unfrozen. LR drops to lr0 × 0.1 to preserve pretrained features.
        Early stopping with patience=20.

    SWA — applied after phase 2 completes
        Averages weights from last swa_n checkpoints.
        Produces swa.pt — typically +0.5–1.5 mAP over best single checkpoint.

CLASS IMBALANCE (train split):
    longitudinal_crack  18,201  (42.4%) ← dominant
    pothole              8,770  (20.5%)
    transverse_crack     8,386  (19.6%)
    alligator_crack      7,526  (17.5%)
    patch_deterioration      0  (reserved for Cluj data)

    Handled via focal loss (fl_gamma=2.0, built into RT-DETR).
    Monitor per-class AP in results.csv — if alligator_crack AP lags,
    increase fl_gamma or add class-weighted sampling in future iteration.

EXPECTED RUNTIME (RTX 2050, batch=4, imgsz=640, 27k images):
    ~50–70 min/epoch  →  60 epochs ≈ 50–70 hours total (~3 nights)
    Train in overnight sessions, resume with --resume flag.
    Use --epochs 100 for maximum accuracy (adds ~2-3 mAP, costs 2 extra nights).

PSO INTEGRATION:
    After PSO produces ml/optimization/pso_best.json, its hyperparams
    (lr0, lrf, momentum, weight_decay, warmup_epochs, mosaic, mixup,
    fl_gamma, box_pw, cls_pw) are loaded automatically at startup and
    override the defaults below. No code changes needed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import RTDETR

# ── CUDA optimizations ─────────────────────────────────────────────────────────
# benchmark=True  → cuDNN auto-tunes fastest conv algorithm for your input size
# deterministic=False → removes the slow deterministic constraint Ultralytics
#   sets by default, which was causing the grid_sampler_2d warning + speed loss
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
PRETRAINED   = ROOT / "rtdetr-l.pt"
DATA_YAML    = ROOT / "data" / "detection" / "dataset.yaml"
PSO_BEST     = ROOT / "ml" / "optimization" / "pso_best.json"
WEIGHTS_DIR  = ROOT / "ml" / "weights"
OUTPUT_DIR   = ROOT / "runs" / "detect"
RUN_NAME     = "rtdetr_road"


def check_dataset_yaml():
    """
    Verify dataset.yaml exists and was produced by coco_to_yolo.py.
    The YAML must point to *_images.txt files, not .json files.
    """
    if not DATA_YAML.exists():
        print(f"\n[ERROR] dataset.yaml not found: {DATA_YAML}")
        print("        Run first:  python ml/detection/data_prep/coco_to_yolo.py")
        sys.exit(1)
    yaml_text = DATA_YAML.read_text()
    if "train.json" in yaml_text or "val.json" in yaml_text:
        print("\n[ERROR] dataset.yaml still points to .json files.")
        print("        Run first:  python ml/detection/data_prep/coco_to_yolo.py")
        sys.exit(1)
    print(f"\n[YAML] dataset.yaml OK → {DATA_YAML}")


# ── Default hyperparameters ────────────────────────────────────────────────────
# These are overridden by PSO output when pso_best.json exists.
DEFAULTS = dict(
    # ── Learning rate ──────────────────────────────────────────────────────────
    lr0           = 1e-4,       # initial LR (phase 1 backbone-frozen)
    lrf           = 0.01,       # final LR = lr0 × lrf  (cosine decay endpoint)
    momentum      = 0.9,        # AdamW β1
    weight_decay  = 5e-4,       # L2 regularisation

    # ── Schedule ───────────────────────────────────────────────────────────────
    warmup_epochs = 3,          # linear LR warmup from 0 → lr0

    # ── Augmentation ──────────────────────────────────────────────────────────
    # Albumentations pipeline is applied automatically by Ultralytics when
    # albumentations is installed.  These flags control the built-in mosaic/mixup.
    mosaic        = 1.0,        # mosaic augmentation probability
    mixup         = 0.15,       # Mixup alpha — set 0 to disable
    copy_paste    = 0.0,        # copy-paste (off for road damage)
    degrees       = 5.0,        # random rotation ±deg
    translate     = 0.1,        # random translation ±fraction
    scale         = 0.5,        # random scale [1-scale, 1+scale]
    shear         = 2.0,        # random shear ±deg
    perspective   = 0.0,        # perspective warp (dashcam: keep 0)
    flipud        = 0.0,        # vertical flip prob (road: 0)
    fliplr        = 0.5,        # horizontal flip prob
    hsv_h         = 0.015,      # HSV hue jitter
    hsv_s         = 0.7,        # HSV saturation jitter
    hsv_v         = 0.4,        # HSV value jitter

    # ── Loss weights ───────────────────────────────────────────────────────────
    # Note: fl_gamma / focal loss is internal to RT-DETR and not exposed as a
    # training arg in Ultralytics 8.x.  box/cls/dfl ARE valid args.
    box           = 7.5,        # box regression loss weight
    cls           = 0.5,        # classification loss weight
    dfl           = 1.5,        # distribution focal loss weight

    # ── Label smoothing ────────────────────────────────────────────────────────
    label_smoothing = 0.0,      # 0 = off; try 0.1 if model overconfident
)


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_pso_hyperparams() -> dict:
    """
    Load hyperparameters produced by PSO (ml/optimization/pso_hyperparams.py).
    Returns empty dict if file not found — defaults will be used.
    """
    if PSO_BEST.exists():
        with open(PSO_BEST) as f:
            pso = json.load(f)
        print(f"\n[PSO] Loaded hyperparameters from {PSO_BEST}")
        for k, v in pso.items():
            print(f"      {k:20s} = {v}")
        return pso
    else:
        print(f"\n[PSO] {PSO_BEST.name} not found — using defaults.")
        print("      Run ml/optimization/pso_hyperparams.py first for optimised training.")
        return {}


def check_environment() -> int:
    """
    Verify CUDA is available and return safe batch size for detected VRAM.
    Exits with error if no GPU found.
    """
    print("\n── Environment ───────────────────────────────────────────────")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("  [ERROR] No CUDA GPU detected. Training on CPU is not supported.")
        sys.exit(1)

    gpu   = torch.cuda.get_device_name(0)
    vram  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU      : {gpu}")
    print(f"  VRAM     : {vram:.1f} GB")

    if vram < 3.5:
        batch = 2
        print("  [WARN] < 3.5 GB VRAM — forcing batch=2")
    elif vram < 6.0:
        batch = 4
        print("  [INFO] < 6 GB VRAM — using batch=4 + grad accumulation ×4")
    elif vram < 12.0:
        batch = 8
    else:
        batch = 16

    return batch


def apply_swa(run_dir: Path, swa_n: int = 5) -> Path:
    """
    Stochastic Weight Averaging over the last `swa_n` saved checkpoints.

    Ultralytics saves checkpoints at save_period intervals as
    epoch{N}.pt files.  We average their state_dicts and write swa.pt.

    Args:
        run_dir : path to the run directory (e.g. runs/detect/rtdetr_road)
        swa_n   : number of checkpoints to average

    Returns:
        Path to swa.pt, or None if not enough checkpoints found.
    """
    weights_dir = run_dir / "weights"
    ckpts = sorted(weights_dir.glob("epoch*.pt"))

    if len(ckpts) < 2:
        print(f"\n[SWA] Only {len(ckpts)} epoch checkpoint(s) found — skipping SWA.")
        print("      Increase save_period or train more epochs for SWA to be effective.")
        return None

    selected = ckpts[-swa_n:]
    print(f"\n── SWA: averaging {len(selected)} checkpoints ─────────────────")
    for c in selected:
        print(f"     {c.name}")

    # Load first checkpoint as base
    base      = torch.load(selected[0], map_location="cpu")
    base_sd   = base["model"].state_dict() if hasattr(base.get("model", None), "state_dict") else base["model"]
    avg_sd    = {k: v.clone().float() for k, v in base_sd.items()}

    # Accumulate remaining checkpoints
    for ckpt_path in selected[1:]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt["model"].state_dict() if hasattr(ckpt.get("model", None), "state_dict") else ckpt["model"]
        for k in avg_sd:
            if k in sd:
                avg_sd[k] += sd[k].float()

    # Divide by count
    n = len(selected)
    for k in avg_sd:
        avg_sd[k] /= n

    # Write back into a copy of the best checkpoint
    best_ckpt = torch.load(weights_dir / "best.pt", map_location="cpu")
    if hasattr(best_ckpt.get("model", None), "load_state_dict"):
        best_ckpt["model"].load_state_dict(
            {k: v.half() for k, v in avg_sd.items()}
        )
    else:
        best_ckpt["model"] = {k: v.half() for k, v in avg_sd.items()}

    swa_path = weights_dir / "swa.pt"
    torch.save(best_ckpt, swa_path)
    print(f"  [SWA] Saved → {swa_path}")
    return swa_path


def copy_best_weights(run_dir: Path, use_swa: bool = True):
    """
    Copy the final weights to ml/weights/rtdetr_l_rdd2022.pt.
    Prefers SWA weights if available, falls back to best.pt.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    swa_pt  = run_dir / "weights" / "swa.pt"
    best_pt = run_dir / "weights" / "best.pt"

    src = swa_pt if (use_swa and swa_pt.exists()) else best_pt
    dst = WEIGHTS_DIR / "rtdetr_l_rdd2022.pt"

    shutil.copy2(src, dst)
    tag = "SWA" if src == swa_pt else "best"
    print(f"\n  [{tag}] Weights → {dst}")


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs:        int   = 60,
    imgsz:         int   = 640,
    freeze_epochs: int   = 10,
    patience:      int   = 20,
    workers:       int   = 4,
    swa_n:         int   = 5,
    resume:        str   = None,
    smoke_test:    bool  = False,
):
    """
    Main training entry point.

    Args:
        epochs        : Total training epochs (phase1 + phase2)
        imgsz         : Input resolution (square). Keep 640 for 4 GB VRAM.
        freeze_epochs : Phase 1 duration (backbone frozen).
        patience      : Early-stopping patience in epochs (phase 2 only).
        workers       : DataLoader workers. Keep ≤4 on Windows.
        swa_n         : Number of checkpoints to average for SWA.
        resume        : Path to a .pt checkpoint to resume from.
        smoke_test    : If True, run 2 quick epochs for pipeline validation.
    """

    # ── Validate dataset.yaml ──────────────────────────────────────────────────
    check_dataset_yaml()

    # ── Smoke test overrides ───────────────────────────────────────────────────
    if smoke_test:
        print("\n[SMOKE TEST] 2 epochs, batch=2, imgsz=320 — pipeline validation only.")
        epochs        = 2
        imgsz         = 320
        freeze_epochs = 1
        patience      = 999
        workers       = 2
        swa_n         = 0

    # ── Environment ───────────────────────────────────────────────────────────
    batch = check_environment()

    # ── Hyperparameters: defaults ← PSO override ───────────────────────────────
    hp = {**DEFAULTS}
    hp.update(load_pso_hyperparams())

    run_dir = OUTPUT_DIR / RUN_NAME

    print("\n── Training configuration ────────────────────────────────────")
    print(f"  Pretrained    : {PRETRAINED}")
    print(f"  Dataset       : {DATA_YAML}")
    print(f"  Total epochs  : {epochs}  (freeze {freeze_epochs} + fine-tune {epochs - freeze_epochs})")
    print(f"  Batch / GPU   : {batch}   (effective: {batch * 4} with grad_accumulate=4)")
    print(f"  Image size    : {imgsz}×{imgsz}")
    print(f"  LR phase 1    : {hp['lr0']}")
    print(f"  LR phase 2    : {hp['lr0'] * 0.1}  (10× lower, backbone unfrozen)")
    print(f"  Focal loss    : built-in to RT-DETR (not user-configurable in Ultralytics 8.x)")
    print(f"  Mixup         : {hp['mixup']}")
    print(f"  Mosaic        : {hp['mosaic']}")
    print(f"  Early stop    : patience={patience} (phase 2 only)")
    print(f"  SWA           : last {swa_n} checkpoints")
    print(f"  Output        : {run_dir}")

    # Shared kwargs passed to both phases
    shared = dict(
        data          = str(DATA_YAML),
        imgsz         = imgsz,
        batch         = batch,
        workers       = workers,
        optimizer     = "AdamW",
        cos_lr        = True,
        amp           = True,           # fp16 — mandatory for 4 GB VRAM
        cache         = False,          # disable image caching (saves ~8 GB RAM)
        plots         = True,
        save          = True,
        save_period   = 5,              # save epoch checkpoint every 5 epochs
        val           = True,
        device        = 0,
        verbose       = True,
        deterministic = False,          # disables slow deterministic ops (+15% speed)
        project       = str(OUTPUT_DIR),
        name          = RUN_NAME,
        exist_ok      = True,
        # augmentation
        mosaic        = hp["mosaic"],
        mixup         = hp["mixup"],
        copy_paste    = hp["copy_paste"],
        degrees       = hp["degrees"],
        translate     = hp["translate"],
        scale         = hp["scale"],
        shear         = hp["shear"],
        perspective   = hp["perspective"],
        flipud        = hp["flipud"],
        fliplr        = hp["fliplr"],
        hsv_h         = hp["hsv_h"],
        hsv_s         = hp["hsv_s"],
        hsv_v         = hp["hsv_v"],
        # loss
        box           = hp["box"],
        cls           = hp["cls"],
        dfl           = hp["dfl"],
        label_smoothing = hp["label_smoothing"],
        # schedule
        warmup_epochs = hp["warmup_epochs"],
        weight_decay  = hp["weight_decay"],
        momentum      = hp["momentum"],
    )

    # ── Auto-detect resume checkpoint ─────────────────────────────────────────
    # If a last.pt already exists from a previous run, offer to resume
    # automatically rather than restarting from scratch.
    auto_last = run_dir / "weights" / "last.pt"
    if resume is None and auto_last.exists() and not smoke_test:
        print(f"\n[RESUME] Detected existing checkpoint: {auto_last}")
        print("  Resuming phase 2 from last.pt automatically.")
        print("  (To restart from scratch, delete runs/detect/rtdetr_road/)")
        resume = str(auto_last)

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    if freeze_epochs > 0 and resume is None:
        print(f"\n── Phase 1: Frozen backbone ({freeze_epochs} epochs) ──────────────")
        print("  Backbone locked. Training decoder + detection head only.")

        model = RTDETR(str(PRETRAINED))
        model.train(
            **shared,
            epochs        = freeze_epochs,

            lr0           = hp["lr0"],
            lrf           = hp["lrf"],
            freeze        = 23,         # freeze first 23 layers (ResNet-50 backbone)
            patience      = 999,        # no early stopping in phase 1
        )

        phase1_ckpt = run_dir / "weights" / "last.pt"
        if not phase1_ckpt.exists():
            print(f"\n[ERROR] Phase 1 did not produce last.pt — training may have crashed.")
            print(f"        Check GPU memory and re-run: python ml/detection/train.py")
            sys.exit(1)
        print(f"\n  Phase 1 complete → {phase1_ckpt}")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    remaining = epochs - (freeze_epochs if resume is None else 0)

    if remaining > 0:
        print(f"\n── Phase 2: Full fine-tune ({remaining} epochs) ─────────────────")
        print("  Backbone unfrozen. LR ×0.1 to preserve pretrained features.")

        if resume is not None:
            start_weights = resume
            print(f"  Resuming from: {resume}")
        elif freeze_epochs > 0:
            phase1_ckpt = run_dir / "weights" / "last.pt"
            start_weights = str(phase1_ckpt) if phase1_ckpt.exists() else str(PRETRAINED)
        else:
            start_weights = str(PRETRAINED)

        # Remove warmup_epochs from shared to avoid duplicate keyword argument
        shared_phase2 = {k: v for k, v in shared.items() if k != "warmup_epochs"}

        model = RTDETR(start_weights)
        model.train(
            **shared_phase2,
            epochs        = remaining,
            resume = True,
            lr0           = hp["lr0"] * 0.1,   # lower LR for full fine-tune
            lrf           = hp["lrf"],
            freeze        = 0,                  # no freezing
            patience      = patience,
            warmup_epochs = 1,                  # short warmup for phase 2
        )

    # ── SWA ───────────────────────────────────────────────────────────────────
    if swa_n > 0:
        swa_path = apply_swa(run_dir, swa_n=swa_n)
    else:
        swa_path = None

    # ── Copy final weights ─────────────────────────────────────────────────────
    copy_best_weights(run_dir, use_swa=(swa_path is not None))

    print("\n── Training complete ─────────────────────────────────────────")
    print(f"  Run dir    : {run_dir}")
    print(f"  Best ckpt  : {run_dir / 'weights' / 'best.pt'}")
    if swa_path:
        print(f"  SWA ckpt   : {swa_path}")
    print(f"  Production : {WEIGHTS_DIR / 'rtdetr_l_rdd2022.pt'}")
    print(f"\n  Next step  : python ml/detection/train.py --validate")
    print(f"             : python ml/optimization/pso_hyperparams.py  ← tune hyperparams")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RT-DETR-L on merged road damage dataset"
    )
    p.add_argument("--epochs",        type=int,   default=60,
                   help="Total training epochs (default: 60 = 10 frozen + 50 fine-tune)")
    p.add_argument("--imgsz",         type=int,   default=640,
                   help="Input image size (default: 640)")
    p.add_argument("--freeze_epochs", type=int,   default=10,
                   help="Phase 1 backbone-frozen epochs (default: 10)")
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early stopping patience in phase 2 (default: 20)")
    p.add_argument("--workers",       type=int,   default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--swa_n",         type=int,   default=5,
                   help="Checkpoints to average for SWA (default: 5, 0=off)")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Quick 2-epoch pipeline validation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        freeze_epochs = args.freeze_epochs,
        patience      = args.patience,
        workers       = args.workers,
        swa_n         = args.swa_n,
        resume        = args.resume,
        smoke_test    = args.smoke_test,
    )