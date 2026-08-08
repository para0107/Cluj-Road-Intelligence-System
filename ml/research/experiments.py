"""
ml/research/experiments.py
--------------------------
The experiment registry for the RDDS detector research program.

This module is the single source of truth for E0-E7 as defined in
ml/research/RESEARCH_PROGRAM.md. The trainer (ml/detection/train_experiment.py) and
the SageMaker launcher (ml/aws/launch.py) both read from here, so a hyperparameter
cannot drift between "what the plan says" and "what the job ran".

Every spec carries its hypothesis, its falsifier and its gate as data, not as a
comment. They are written into the run's config.json, so six weeks later the run
directory still says what the run was trying to find out and what would have counted
as a negative result. That is the difference between an experiment and a training job.

Usage:
    from ml.research.experiments import REGISTRY, get, variants_of

    spec = get("E3-1024sq")
    print(spec.hypothesis)
    print(spec.train_kwargs())          # ready to splat into model.train()

CLI:
    python ml/research/experiments.py --list
    python ml/research/experiments.py --show E4b
    python ml/research/experiments.py --stage E3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Allow `python ml/research/experiments.py` as well as `import ml.research.experiments`.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__all__ = ["ExperimentSpec", "REGISTRY", "get", "variants_of", "stage_of", "BASELINE_HP"]


# ---------------------------------------------------------------------------
# The baseline hyperparameters.
#
# These are the PSO-optimised values from ml/optimization/pso_best.json, merged with
# the non-searched defaults from ml/detection/train.py. They reproduce the current
# production checkpoint (val mAP50 0.5637 / mAP50-95 0.2945 at epoch 57).
#
# CAUTION: several of these exist only because the original training ran on a 4 GB
# RTX 2050. batch and imgsz in particular were VRAM-constrained, not tuned. On a
# 24 GB A10G they should be treated as free parameters again, which is what E3 does.
# ---------------------------------------------------------------------------
BASELINE_HP: dict[str, Any] = {
    # --- searched by PSO (ml/optimization/pso_best.json) ---
    "lr0": 0.0004465118975867086,
    "weight_decay": 0.000526695253368712,
    "warmup_epochs": 1,
    "mosaic": 0.8603609096800973,
    "mixup": 0.20451311070797243,
    "box": 7.684851652043976,
    "cls": 0.4867776329667799,
    # --- not searched, inherited from train.py DEFAULTS ---
    "lrf": 0.01,
    "momentum": 0.9,
    "dfl": 1.5,
    "label_smoothing": 0.0,
    "copy_paste": 0.0,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0,  # dashcam: keep 0
    "flipud": 0.0,       # road scenes are not vertically symmetric
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    # --- optimiser / schedule ---
    "optimizer": "AdamW",
    "cos_lr": True,
    "amp": True,
}

# NOTE on fliplr for this dataset:
#   fliplr=0.5 horizontally mirrors the frame. That maps a longitudinal crack to a
#   longitudinal crack (fine) and a transverse crack to a transverse crack (fine),
#   so it is label-preserving here. It is listed explicitly because if E4 introduces
#   an orientation-aware prior, the interaction between that prior and this
#   augmentation has to be reasoned about rather than inherited silently.


# ---------------------------------------------------------------------------
# Runtime estimation, derived from THIS project's own measurements
#
# runs/detect/nrdd_2024/results.csv records 57 epochs in 42,292 s, a median of
# 740 s/epoch. That was an RTX 2050 (4 GB) at 640px with batch 4 - the batch size was
# forced by VRAM, not chosen.
#
# The research GPUs (A10G 24 GB / L4 24 GB) are roughly 3-4x faster on fp16 AND allow
# batch 16, which improves utilisation further. GPU_FACTOR is set to the conservative
# end of that range: over-estimating runtime wastes planning, under-estimating it
# wastes a 72-hour window that cannot be extended.
#
# Compute time scales with pixel count, so resolution scales quadratically.
#
# These are estimates from one measurement, not a benchmark. Re-derive GPU_FACTOR
# from your first real run's epoch timing before planning a whole weekend around it.
# ---------------------------------------------------------------------------
MEASURED_S_PER_EPOCH_640 = 740.0   # median, runs/detect/nrdd_2024/results.csv
MEASURED_ON = "RTX 2050 4GB, batch 4, imgsz 640"
GPU_FACTOR = 3.5                   # A10G / L4 vs the measured baseline

# ---------------------------------------------------------------------------
# Weekend-1 correction: measure the target GPU, do not extrapolate to it.
#
# The constants above predict 740/3.5 = 211 s/epoch on the L4. The three completed
# E0 runs actually recorded 357 s (frozen backbone) and 547 s (full fine-tune) per
# epoch at 640px / batch 16, so the extrapolation was optimistic by a factor of ~2.6
# on the phase that dominates a run. EXPERIMENTS.md section 10 logs the cost of that
# error; these two constants are what stop it recurring.
#
# Source: runs/research/2026080*_E0-baseline_s133*/metrics.csv, elapsed_s deltas.
# Spread across the three seeds was 1.6 s and 2.6 s respectively - this is a
# measurement, not an estimate.
#
# The two phases differ because a frozen backbone skips most of the backward pass.
# Any spec with freeze_epochs > 0 pays both rates, so the estimator models them
# separately rather than averaging.
# ---------------------------------------------------------------------------
L4_S_PER_EPOCH_640_FROZEN = 357.5
L4_S_PER_EPOCH_640_FULL = 546.7
L4_MEASURED_ON = "NVIDIA L4 22GB, batch 16, imgsz 640, 13.3k train images"
L4_FIXED_OVERHEAD_S = 430.0        # model load, final val, test eval, artefact export


def estimate_hours(epochs: int, imgsz: int | tuple[int, int], n_seeds: int = 1,
                   gpu_factor: float = GPU_FACTOR, freeze_epochs: int = 0,
                   throughput_gain: float = 1.0) -> float:
    """
    Estimated GPU-hours for a spec, from the measured L4 epoch times.

    Args:
        epochs: total epochs, frozen phase included.
        imgsz: square side, or (w, h). Cost is modelled as linear in pixel count.
        n_seeds: runs at this configuration.
        gpu_factor: retained for callers that still pass it; ignored, because the
            timings are now measured on the target GPU rather than extrapolated to it.
        freeze_epochs: epochs spent with the backbone frozen, which are cheaper.
        throughput_gain: speedup from a larger batch. Leave at 1.0 until measured -
            assuming a batch win and not getting one is how a queue overruns.

    Pixel scaling is the weakest part of this model: attention cost is not exactly
    linear in pixels. Treat non-640 estimates as +-20% and re-measure after the first
    epoch at a new resolution.
    """
    if epochs <= 0:
        return 0.25   # evaluation-only experiments still need an instance briefly
    px = imgsz * imgsz if isinstance(imgsz, int) else imgsz[0] * imgsz[1]
    scale = px / (640 * 640)
    n_frozen = min(max(freeze_epochs, 0), epochs)
    n_full = epochs - n_frozen
    per_seed = (
        n_frozen * L4_S_PER_EPOCH_640_FROZEN + n_full * L4_S_PER_EPOCH_640_FULL
    ) * scale / max(throughput_gain, 1e-6) + L4_FIXED_OVERHEAD_S
    return round(per_seed * n_seeds / 3600.0, 2)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """One experiment. Immutable description; the trainer resolves it into a run."""

    id: str
    stage: str                    # E0 .. E7
    title: str
    hypothesis: str
    falsifier: str                # what observation would prove the hypothesis wrong
    gate: str                     # what decision this experiment's result drives
    primary_metric: str = "test/mAP50-95"

    # model / data
    model: str = "rtdetr-l.pt"    # ultralytics weights id or path
    imgsz: int | tuple[int, int] = 640
    epochs: int = 60
    freeze_epochs: int = 10
    batch: int = 0                # 0 = auto from detected VRAM
    seeds: tuple[int, ...] = (1337,)

    # which dataset (ml/research/datasets.py) and which classes of it
    # (ml/research/class_sets.py). "all10" reproduces the current production setup.
    dataset: str = "nrdd2024"
    class_set: str = "all10"

    # HOW the data must be staged (see DATA_VARIANTS). Two specs with the same model
    # and class set but different data variants are NOT comparable by default; the
    # variant is what the experiment is manipulating.
    data_variant: str = "standard"

    # hyperparameter deltas applied on top of BASELINE_HP
    overrides: dict[str, Any] = field(default_factory=dict)

    # free-form notes that belong with the result, not in a commit message
    notes: str = ""

    # compute hint for the launcher.
    #   instance      - for SageMaker training jobs (personal account)
    #   est_gpu_hours - 0 means "derive it"; set a number only to override
    instance: str = "ml.g5.2xlarge"
    est_gpu_hours: float = 0.0

    def hours(self, gpu_factor: float = GPU_FACTOR,
              throughput_gain: float = 1.0) -> float:
        """Estimated GPU-hours for ALL seeds of this spec."""
        if self.est_gpu_hours:
            return self.est_gpu_hours
        return estimate_hours(self.epochs, self.imgsz, len(self.seeds), gpu_factor,
                              self.freeze_epochs, throughput_gain)

    def hours_per_seed(self, gpu_factor: float = GPU_FACTOR,
                       throughput_gain: float = 1.0) -> float:
        return estimate_hours(self.epochs, self.imgsz, 1, gpu_factor,
                              self.freeze_epochs, throughput_gain)

    def stage_command(self, source: str = "$SRC", out: str = "$STAGED") -> str:
        """
        The exact `stage_dataset.py` invocation this spec's data requires.

        Experiments that differ only in how the data was staged (E9's oversampling
        ratios, E10's country holdouts) are otherwise indistinguishable in their
        config.json, which is how two runs end up looking comparable when they are
        not. Emitting the command from the spec keeps the staging decision inside the
        registry rather than in shell history.
        """
        variant = DATA_VARIANTS[self.data_variant]
        args = " ".join(variant["stage_args"])
        return (f"python ml/aws/stage_dataset.py --source {source} "
                f"--out {out} {args}".rstrip())

    def hyperparams(self) -> dict[str, Any]:
        """BASELINE_HP with this experiment's overrides applied."""
        hp = dict(BASELINE_HP)
        hp.update(self.overrides)
        return hp

    def train_kwargs(self) -> dict[str, Any]:
        """
        Hyperparameters filtered to the keys Ultralytics `model.train()` accepts.

        Anything the registry carries for our own bookkeeping (and anything a future
        override adds that Ultralytics does not know) is dropped here rather than
        being passed through and raising deep inside the trainer.
        """
        hp = self.hyperparams()
        return {k: v for k, v in hp.items() if k in _ULTRALYTICS_TRAIN_ARGS}

    def unknown_keys(self) -> list[str]:
        """Override keys Ultralytics will not accept. Used by the validity check."""
        return sorted(k for k in self.hyperparams() if k not in _ULTRALYTICS_TRAIN_ARGS)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resolved_hyperparams"] = self.hyperparams()
        return d


# Ultralytics 8.x train() arguments we deliberately allow through. Kept explicit so a
# typo in an override is caught by --check rather than silently ignored at runtime.
_ULTRALYTICS_TRAIN_ARGS: set[str] = {
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
    "warmup_bias_lr", "box", "cls", "dfl", "label_smoothing", "nbs",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "copy_paste",
    "erasing", "crop_fraction", "auto_augment",
    "optimizer", "cos_lr", "amp", "dropout", "close_mosaic", "multi_scale",
}


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ExperimentSpec] = {}


# ---------------------------------------------------------------------------
# Data variants — how the split was built, recorded as data
#
# E9 and E10 do not change the model at all. They change how `stage_dataset.py`
# constructed the splits, which is invisible in a training config and therefore the
# easiest thing in this programme to get silently wrong. Naming each staging recipe
# and attaching it to the spec means a run directory records not just what was
# trained but what it was trained on.
#
# Near-duplicate detection is ON at `--hash dhash --hash-threshold 2`, and that
# threshold is MEASURED rather than assumed. `--calibrate-hash 300` over the pooled
# 18,995-image archive (8 Aug 2026) reported genuine re-encodes at p95 = 2 bits and
# different images starting at 3 bits: a clean gap, 0.00% false positives at 2.
#
# Two things this changes from weekend 1, both of which belong in the write-up:
#   - Weekend 1 ran with near-duplicate detection OFF (aHash had collapsed at 100%
#     false positives), so its splits do not account for near-duplicates at all.
#   - The archive contains ZERO byte-identical duplicates, so every near-duplicate
#     found here comes from dHash and would otherwise have been free to straddle a
#     split boundary.
# Consequence: these splits are NOT the weekend-1 splits, and test mAP50-95 0.1991 is
# not directly comparable to anything staged this way. The clean 3-seed E0 re-run is
# the new reference baseline.
# ---------------------------------------------------------------------------
DATA_VARIANTS: dict[str, dict] = {
    "standard": {
        "description": "The weekend-1 split: 70/15/15 stratified, train oversampled "
                       "to 0.30 of the most common class. E0's baseline.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30"],
    },
    "no_oversample": {
        "description": "Identical, but the train split is left at its natural class "
                       "distribution.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--no-oversample"],
    },
    "oversample_60": {
        "description": "Rare classes pushed to 0.60 of the most common class.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.60"],
    },
    "loco_japan": {
        "description": "Leave-one-country-out, Japan held out (37.9% of the data).",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-country", "japan"],
    },
    "loco_norway": {
        "description": "LOCO, Norway held out (14.8%). The closest proxy for "
                       "northern-European roads, which is the RDDS deployment.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-country", "norway"],
    },
    "loco_india": {
        "description": "LOCO, India held out (6.4%). The largest expected domain gap.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-country", "india"],
    },
    "loco_czech": {
        "description": "LOCO, Czech Republic held out (5.2%). European, and the "
                       "smallest archive, so the train-size confound is smallest.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-country", "czech"],
    },
    "control_2803": {
        "description": "The control for loco_norway: 2,803 images held out at random "
                       "across all six countries, matching Norway's size.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-control", "2803"],
    },
    "control_1221": {
        "description": "The control for loco_india: 1,221 images held out at random.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-control", "1221"],
    },
    "control_992": {
        "description": "The control for loco_czech: 992 images held out at random.",
        "stage_args": ["--hash", "dhash", "--hash-threshold", "2", "--oversample", "0.30",
                       "--holdout-control", "992"],
    },
}


def _reg(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.id in REGISTRY:
        raise ValueError(f"duplicate experiment id: {spec.id}")
    if spec.data_variant not in DATA_VARIANTS:
        raise ValueError(
            f"{spec.id}: unknown data_variant {spec.data_variant!r}; "
            f"expected one of {sorted(DATA_VARIANTS)}"
        )
    REGISTRY[spec.id] = spec
    return spec


# --- E0: reproduce and instrument -----------------------------------------
_reg(ExperimentSpec(
    id="E0-baseline",
    stage="E0",
    title="Reproduce the production checkpoint under seeding",
    hypothesis=(
        "The current production result (val mAP50-95 0.2945) is reproducible under "
        "explicit seeding, and per-seed variance is small enough that single-run "
        "comparisons in later experiments are meaningful."
    ),
    falsifier=(
        "Across-seed standard deviation of test mAP50-95 exceeds ~0.01, which would "
        "mean any single-run difference below that is noise."
    ),
    gate=(
        "If seed variance is large, every later experiment needs 3 seeds and the "
        "compute budget triples. Decide this before spending it, not after."
    ),
    model="rtdetr-l.pt",
    imgsz=640,
    epochs=60,
    freeze_epochs=10,
    seeds=(1337, 1338, 1339),
    notes=(
        "First run in the program to produce per-class AP for the 10-class model, and "
        "the first evaluated on a held-out test split. Expect the test number to be "
        "BELOW 0.2945: the published figure was selected on val over 57 epochs."
    ),
))

# --- E2: architecture bake-off --------------------------------------------
# (E1 is the dataset audit and runs on CPU — see ml/research/audit.py, no GPU spec.)
_E2_MODELS = {
    "rtdetrl": ("rtdetr-l.pt", "RT-DETR-L, the incumbent baseline (Zhao et al. 2023)"),
    "rtdetrx": ("rtdetr-x.pt", "RT-DETR-X, larger variant: is the baseline capacity-bound?"),
    "yolo11l": ("yolo11l.pt", "YOLO11-L, the base StripRFNet builds on"),
    "yolo12l": ("yolo12l.pt", "YOLOv12-L, current-generation CNN/attention hybrid"),
}
for _key, (_w, _desc) in _E2_MODELS.items():
    _reg(ExperimentSpec(
        id=f"E2-{_key}",
        stage="E2",
        title=f"Architecture bake-off: {_key}",
        hypothesis=(
            "Part of the gap to the field is architectural age, not method. A "
            "current-generation detector trained under the identical protocol "
            "outperforms RT-DETR-L on N-RDD2024."
        ),
        falsifier=(
            "No architecture beats RT-DETR-L by more than the E0 seed variance under "
            "matched data, schedule, augmentation and seed."
        ),
        gate=(
            "If a newer model gives >2 points mAP50-95 for free, adopt it as the "
            "baseline BEFORE E4. Demonstrating a novel module on an outdated backbone "
            "invites the obvious reviewer question."
        ),
        model=_w,
        imgsz=640,
        epochs=60,
        freeze_epochs=10,
        seeds=(1337,),
        notes=_desc + " | Only the architecture varies; everything else is E0-identical.",
    ))

# --- E3: resolution and input aspect ratio --------------------------------
_E3_SIZES: list[tuple[str, int | tuple[int, int], str]] = [
    ("640sq",   640,          "Baseline. Inherited from a 4 GB VRAM constraint, not tuned."),
    ("800sq",   800,          "Modest upscale; first test of the sub-pixel-crack hypothesis."),
    ("1024sq",  1024,         "Large upscale. Watch latency against the E7 edge target."),
    ("1024x576", (1024, 576), "Native 16:9. Removes ~44% padding waste from letterboxing."),
]
for _name, _sz, _desc in _E3_SIZES:
    _reg(ExperimentSpec(
        id=f"E3-{_name}",
        stage="E3",
        title=f"Input resolution / aspect: {_name}",
        hypothesis=(
            "Thin cracks are destroyed by the input pipeline before the model sees "
            "them. At 640x640 a hairline crack approaches sub-pixel width, and "
            "letterboxing 16:9 dashcam frames into a square spends a large fraction of "
            "the input budget on padding."
        ),
        falsifier=(
            "AP on the thin classes (longitudinal, transverse, rutting) does not "
            "improve with resolution, or improves no more than the compact classes do."
        ),
        gate=(
            "Highest expected gain per GPU-hour in the program. Run it BEFORE E4, so "
            "any E4 claim is measured on top of the resolution win rather than "
            "duplicating it."
        ),
        model="rtdetr-l.pt",
        imgsz=_sz,
        epochs=60,
        freeze_epochs=10,
        seeds=(1337,),
        notes=_desc,
    ))

# --- E4: the contribution -------------------------------------------------
_reg(ExperimentSpec(
    id="E4a-striprf",
    stage="E4",
    title="Strip receptive fields in the RT-DETR encoder",
    hypothesis=(
        "Asymmetric 1xk and kx1 convolutions in the hybrid encoder capture slender "
        "crack structure that square kernels miss, reproducing inside a detection "
        "transformer the gain StripRFNet reports inside a CNN."
    ),
    falsifier="No improvement in AP on longitudinal, transverse or rutting classes.",
    gate=(
        "This is the CONTROL, not the contribution. It establishes how much of any "
        "E4b gain is explained by strip receptive fields alone. Without it, E4b's "
        "novelty claim is not separable."
    ),
    model="rtdetr-l.pt",
    imgsz=1024,
    epochs=60,
    seeds=(1337,),
    notes=(
        "Requires a custom model YAML. Port of StripRFNet's SRFM (arXiv:2510.16115) "
        "into the RT-DETR encoder. Gated on the E1 correlation result."
    ),
))

_reg(ExperimentSpec(
    id="E4b-oriented-queries",
    stage="E4",
    title="Orientation-aware decoder query initialisation",
    hypothesis=(
        "RT-DETR selects encoder features to initialise decoder queries. Biasing that "
        "selection and the query positional prior by an explicit anisotropy signal "
        "predisposes a subset of queries to elongated, oriented boxes, improving the "
        "thin classes beyond what strip convolutions alone achieve."
    ),
    falsifier=(
        "AP on longitudinal/transverse/rutting does not exceed E4a. If overall mAP "
        "rises while those classes do not, the stated mechanism is wrong and must be "
        "reported as such."
    ),
    gate=(
        "THE contribution. Strip-awareness exists in CNN detectors; no published "
        "DETR-family work applies a shape prior to the object queries. If E1's "
        "correlation is weak, do not run this - the motivation collapses."
    ),
    model="rtdetr-l.pt",
    imgsz=1024,
    epochs=60,
    seeds=(1337, 1338, 1339),
    notes=(
        "Highest-risk, highest-reward item in the program. Three seeds because the "
        "headline claim depends on it and a single run will not survive review."
    ),
))

_reg(ExperimentSpec(
    id="E4c-ar-loss",
    stage="E4",
    title="Aspect-ratio-aware regression loss",
    hypothesis=(
        "Augmenting the GIoU term with an explicit aspect-ratio penalty makes a shape "
        "error on a thin crack cost more than the same IoU error on a compact pothole, "
        "correcting a bias in the standard box loss."
    ),
    falsifier="Thin-class AP unchanged, or precision degrades without a recall gain.",
    gate="Cheapest of the three E4 variants. Run it even if E4b is deferred.",
    model="rtdetr-l.pt",
    imgsz=1024,
    epochs=60,
    seeds=(1337,),
    overrides={"box": 9.0},
    notes=(
        "The box=9.0 override is a placeholder for the re-weighting that accompanies "
        "the custom loss term; the loss itself is a code change, not a hyperparameter. "
        "Do not report this spec as if the override alone were the experiment."
    ),
))

# --- E5: recall-oriented operating point ----------------------------------
_reg(ExperimentSpec(
    id="E5-recall",
    stage="E5",
    title="Cost-sensitive, recall-oriented training and thresholds",
    hypothesis=(
        "The detector is better than its F1 suggests and is simply thresholded for the "
        "wrong cost function. In a municipal survey a missed defect costs more than a "
        "false alarm an operator dismisses in one click."
    ),
    falsifier=(
        "F2 at the tuned operating point is no better than F2 at the default 0.35 "
        "threshold, i.e. there was no headroom in the operating point."
    ),
    gate=(
        "State and justify the cost ratio explicitly. Choosing beta=2 because it "
        "flatters the number is not a result."
    ),
    model="rtdetr-l.pt",
    imgsz=1024,
    epochs=60,
    seeds=(1337,),
    overrides={"cls": 0.8, "label_smoothing": 0.05},
    notes=(
        "Most of this experiment is post-hoc threshold selection on the val split "
        "(no GPU), evaluated once on test. The training-side deltas here address the "
        "class imbalance E1 will quantify. Report false positives per kilometre "
        "alongside F2 - that is the number a municipality can act on."
    ),
))

# --- E6: cross-dataset generalisation -------------------------------------
_reg(ExperimentSpec(
    id="E6-crossdata",
    stage="E6",
    title="N-RDD2024 -> RDD2022 cross-dataset transfer",
    hypothesis=(
        "A model trained on N-RDD2024 degrades substantially on road imagery from a "
        "different country, which is the situation the deployed Cluj system is in."
    ),
    falsifier="Per-class AP drop under domain shift is within the E0 seed variance.",
    gate=(
        "A measurement, not an optimisation. It is the question a reviewer asks, and "
        "it is a live engineering risk for deployment."
    ),
    model="rtdetr-l.pt",
    imgsz=1024,
    epochs=0,   # evaluation only: reuses the best checkpoint from earlier stages
    seeds=(1337,),
    notes=(
        "Map the overlapping classes explicitly and document what does NOT map - "
        "RDD2022 has no equivalent for several N-RDD2024 classes. Verify N-RDD2024's "
        "actual geographic composition from the Mendeley record; do not assume it."
    ),
))

# --- E7: deployment Pareto ------------------------------------------------
_reg(ExperimentSpec(
    id="E7-pareto",
    stage="E7",
    title="Accuracy vs latency Pareto front at the edge target",
    hypothesis="The best research model is not the best deployable model.",
    falsifier="The most accurate model is also the fastest (it will not be).",
    gate=(
        "A measurement. Its value is that this project has a genuinely deployed "
        "system behind it, which almost no RDD paper can show."
    ),
    model="rtdetr-l.pt",
    imgsz=640,
    epochs=0,   # benchmarking only
    seeds=(1337,),
    notes=(
        "Benchmark every checkpoint in the program through the same ONNX/fp16 paths "
        "pipeline/live_pipeline.py already uses, so the latency numbers describe the "
        "real deployment and not a synthetic one."
    ),
))


# --- E8: class-set ablation -----------------------------------------------
# The question: is the detector spending capacity on classes the product discards?
# pipeline/detector.py already caps both marking classes at severity S1 regardless
# of confidence, so the system treats them as non-damage while the model still pays
# full price to learn them.
#
# Every E8 variant is identical to E0 apart from the class set, so the difference IS
# the class set. Compare per-class AP on the SHARED classes only - comparing an
# aggregate mAP over 7 classes against one over 10 is meaningless, since removing
# hard classes raises the mean without improving anything.
_E8_SETS = [
    ("all10", "Control. Identical to E0; included so E8 is self-contained."),
    ("structural7", "Drop the two marking classes and manhole_cover."),
    ("rdd2022compat", "The four classes RDD2022 shares. Also the E6 training config."),
    ("cracks_merged", "Collapse the four crack subtypes into one."),
    ("core4", "The defects that generate work orders."),
]
for _cs, _desc in _E8_SETS:
    _reg(ExperimentSpec(
        id=f"E8-{_cs}",
        stage="E8",
        title=f"Class-set ablation: {_cs}",
        hypothesis=(
            "Classes the downstream product discards are consuming model capacity "
            "and classification gradient. Removing them improves AP on the classes "
            "a municipality acts on."
        ),
        falsifier=(
            "Per-class AP on the SHARED classes does not improve, or degrades. A "
            "degradation would be informative: it would mean the removed classes "
            "were acting as hard negatives (a lane line resembles a crack), which "
            "argues for keeping them as an auxiliary task rather than a target."
        ),
        gate=(
            "Compare on shared classes ONLY. Aggregate mAP over a smaller class set "
            "is not comparable to mAP over a larger one - dropping the hardest "
            "classes raises the mean without improving any prediction. "
            "ml/research/compare.py's paired per-class test does the right thing here."
        ),
        model="rtdetr-l.pt",
        imgsz=640,
        epochs=60,
        freeze_epochs=10,
        seeds=(1337,),
        dataset="nrdd2024",
        class_set=_cs,
        notes=_desc + " | Everything except the class set is E0-identical.",
    ))

# --- E6 companions: the cross-dataset pair --------------------------------
# E6 (declared above) is evaluation-only. These two TRAIN the matched-schema models
# that make the transfer measurable in the first place.
for _src, _other in (("nrdd2024", "rdd2022"), ("rdd2022", "nrdd2024")):
    _reg(ExperimentSpec(
        id=f"E6-train-{_src}",
        stage="E6",
        title=f"Train on {_src} with the shared 4-class schema",
        hypothesis=(
            f"A model trained on {_src} transfers to {_other} only to the extent the "
            f"two datasets share a domain, not merely a label schema."
        ),
        falsifier=f"Per-class AP on {_other} is within seed noise of AP on {_src}.",
        gate=(
            "Both sides MUST use the 'rdd2022compat' class set. The four shared "
            "classes have different ids in the two schemas (pothole is index 4 in "
            "N-RDD2024 and 3 in RDD2022); ml/research/datasets.py::cross_map handles "
            "it, but only if both models were trained on the matched schema."
        ),
        model="rtdetr-l.pt",
        imgsz=640,
        epochs=60,
        freeze_epochs=10,
        seeds=(1337,),
        dataset=_src,
        class_set="rdd2022compat",
        notes=(
            f"Evaluate against {_other}'s test split afterwards. Only the four "
            f"shared classes are measurable; report it that way."
        ),
    ))


# --- E9: does the long tail respond to resampling? ------------------------
# Weekend 1 measured rutting at AP 0.000 on 12/3/3 instances, and the seven damage
# classes span 0.34 to 0.58 while manhole_cover - the one class the product discards -
# sits at 0.81. Before concluding anything about architecture, find out whether the
# spread is simply the class frequency distribution showing through.
#
# The current staging oversamples rare classes to 0.30 of the most common class. That
# constant was never ablated; it is a default nobody measured. These three runs bracket
# it. Everything except the staging recipe is E0-identical.
_E9_VARIANTS = [
    ("none", "no_oversample",
     "Natural class distribution. If AP on the rare classes barely moves, the current "
     "oversampling is doing nothing and the tail needs a different treatment."),
    ("30", "standard",
     "The incumbent, included so E9 is self-contained and does not lean on E0's runs "
     "having used an identical staging build."),
    ("60", "oversample_60",
     "Twice the current target. If AP rises roughly linearly from 'none' through 0.30 "
     "to 0.60, resampling is the cheap lever and the ceiling has not been found."),
]
for _tag, _variant, _desc in _E9_VARIANTS:
    _reg(ExperimentSpec(
        id=f"E9-oversample{_tag}",
        stage="E9",
        title=f"Rare-class oversampling ablation: target={_tag}",
        hypothesis=(
            "The per-class AP spread is driven by class frequency rather than by "
            "anything about the classes themselves. Rebalancing the training "
            "distribution raises AP on the rare classes."
        ),
        falsifier=(
            "AP on the rare classes (rutting, repaired_crack, patchy_road) is flat "
            "across all three resampling levels, i.e. duplicating images does not "
            "supply the information the model is missing."
        ),
        gate=(
            "Judge on per-class AP for the RARE classes, not aggregate mAP - "
            "oversampling trades common-class performance for rare-class "
            "performance, and the aggregate hides that trade. rutting has 12 train "
            "instances and is unmeasurable either way; report it as such rather than "
            "quoting a number. If resampling is flat, the tail is a labelling-volume "
            "problem and the honest recommendation is to drop the class, not to "
            "train harder on it."
        ),
        model="rtdetr-l.pt",
        imgsz=640,
        epochs=20,
        freeze_epochs=10,
        seeds=(1337,),
        data_variant=_variant,
        notes=_desc,
    ))


# --- E10: leave-one-country-out generalisation ----------------------------
# The replacement research leg. E1 refuted the shape hypothesis and cancelled E4, so
# the programme needs a second contribution beyond "here is a clean baseline".
#
# N-RDD2024 is six country archives concatenated (japan 37.9%, usa 25.3%, norway 14.8%,
# china 10.4%, india 6.4%, czech 5.2%), and the deployed RDDS system runs in Cluj,
# which is in none of them. "Train on five, test on the sixth" is simultaneously a
# benchmark contribution the dataset has never had and the exact engineering question
# the deployment poses.
#
# EVERY fold is paired with a matched-size random-holdout control. Holding out a
# country removes both a domain AND a slice of the training set; without the control
# the two are inseparable and the number does not measure what it claims to. Report
#
#     domain_shift_effect = AP(control) - AP(LOCO)
#
# and never the LOCO figure on its own. Japan is deliberately excluded from the
# weekend-2 queue: at 37.9% of the data its train-size confound is larger than the
# effect being measured.
_E10_FOLDS = [
    ("norway", "loco_norway", "control_2803", 2803,
     "Northern European roads: the closest available proxy for the Cluj deployment, "
     "and the fold whose result the product actually depends on."),
    ("india", "loco_india", "control_1221", 1221,
     "The largest expected domain gap - different road construction, surface "
     "materials, traffic and camera mounting. The upper bound on the drop."),
    ("czech", "loco_czech", "control_992", 992,
     "European and the smallest archive, so the train-size confound is smallest and "
     "the domain effect is cleanest to attribute."),
]
for _country, _loco_variant, _ctrl_variant, _n, _why in _E10_FOLDS:
    _reg(ExperimentSpec(
        id=f"E10-loco-{_country}",
        stage="E10",
        title=f"Leave-one-country-out: {_country} held out",
        hypothesis=(
            "A detector trained on N-RDD2024 degrades substantially on roads from a "
            "country absent from training. This is the situation the deployed Cluj "
            "system is in, since Romania is in no public road-damage dataset."
        ),
        falsifier=(
            "Per-class AP on the held-out country is within the E0 seed-noise floor "
            "(0.0039 mAP50-95) of the matched random-holdout control, i.e. the model "
            "does not care which country its test images came from."
        ),
        gate=(
            f"MUST be reported against E10-control-{_country}, never alone. Holding "
            f"out {_country} removes {_n} training images as well as a domain; the "
            "control removes the same COUNT at random, so the difference between the "
            "two isolates domain shift. A LOCO number quoted without its control "
            "measures train-set size and calls it generalisation."
        ),
        model="rtdetr-l.pt",
        imgsz=640,
        epochs=20,
        freeze_epochs=10,
        seeds=(1337,),
        data_variant=_loco_variant,
        notes=_why + " | Test-split class distribution differs per country; report "
                     "per-class AP and mark classes with under ~30 test instances as "
                     "unmeasurable rather than quoting their AP.",
    ))
    _reg(ExperimentSpec(
        id=f"E10-control-{_country}",
        stage="E10",
        title=f"LOCO control for {_country}: {_n} random images held out",
        hypothesis=(
            "Part of any leave-one-country-out drop is explained purely by the "
            "smaller training set, with no domain shift involved."
        ),
        falsifier=(
            "The control scores the same as the full-data E0 baseline, which would "
            "mean removing this many images costs nothing and the whole LOCO drop is "
            "attributable to domain."
        ),
        gate=(
            "This run exists only to be subtracted from its LOCO partner. It is not "
            "a result on its own and should not appear in a leaderboard as one."
        ),
        model="rtdetr-l.pt",
        imgsz=640,
        epochs=20,
        freeze_epochs=10,
        seeds=(1337,),
        data_variant=_ctrl_variant,
        notes=f"Matched to E10-loco-{_country} at {_n} held-out images. Same seed, "
              f"same class set, same schedule - the ONLY difference from its partner "
              f"is which images were removed.",
    ))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def get(experiment_id: str) -> ExperimentSpec:
    """Fetch one spec by id, with a helpful error listing the valid ids."""
    try:
        return REGISTRY[experiment_id]
    except KeyError:
        raise KeyError(
            f"unknown experiment '{experiment_id}'. Known: {', '.join(sorted(REGISTRY))}"
        ) from None


def stage_of(experiment_id: str) -> str:
    return get(experiment_id).stage


def variants_of(stage: str) -> list[ExperimentSpec]:
    """All specs belonging to one stage, e.g. variants_of('E3')."""
    return [s for s in REGISTRY.values() if s.stage == stage.upper()]


def check_registry() -> list[str]:
    """
    Validate every spec. Returns a list of problems; empty means clean.
    Run this in CI, or at least before submitting a batch of jobs.
    """
    from ml.research.class_sets import CLASS_SETS, NRDD2024_CLASSES
    from ml.research.datasets import DATASETS

    problems: list[str] = []
    for spec in REGISTRY.values():
        unknown = spec.unknown_keys()
        if unknown:
            problems.append(f"{spec.id}: overrides Ultralytics will ignore: {unknown}")
        if not spec.seeds:
            problems.append(f"{spec.id}: no seeds")
        if spec.epochs > 0 and spec.freeze_epochs >= spec.epochs:
            problems.append(
                f"{spec.id}: freeze_epochs ({spec.freeze_epochs}) >= epochs ({spec.epochs})"
            )
        for req in ("hypothesis", "falsifier", "gate"):
            if not getattr(spec, req).strip():
                problems.append(f"{spec.id}: empty {req}")
        if spec.dataset not in DATASETS:
            problems.append(f"{spec.id}: unknown dataset '{spec.dataset}'")
        if spec.class_set not in CLASS_SETS:
            problems.append(f"{spec.id}: unknown class set '{spec.class_set}'")
        else:
            # Catch a class set that names a class the chosen dataset does not have -
            # otherwise the error surfaces only after the job has started on a GPU.
            try:
                CLASS_SETS[spec.class_set].source_to_output(
                    DATASETS[spec.dataset].classes
                    if spec.dataset in DATASETS else NRDD2024_CLASSES
                )
            except KeyError as exc:
                problems.append(f"{spec.id}: class set incompatible with dataset: {exc}")
        if spec.data_variant not in DATA_VARIANTS:
            problems.append(f"{spec.id}: unknown data_variant '{spec.data_variant}'")

    # Every leave-one-country-out fold must have a matched control registered. This is
    # the one methodological invariant of E10: without the control, the fold measures
    # training-set size and reports it as domain shift. Enforced here rather than left
    # to the write-up, because by the time it reaches the write-up the GPU hours are
    # already spent.
    for spec in REGISTRY.values():
        if spec.stage == "E10" and spec.id.startswith("E10-loco-"):
            country = spec.id[len("E10-loco-"):]
            if f"E10-control-{country}" not in REGISTRY:
                problems.append(
                    f"{spec.id}: no matched control. Register E10-control-{country} "
                    f"holding out the same number of images at random, or the fold "
                    f"cannot be attributed to domain shift."
                )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> int:
    ap = argparse.ArgumentParser(description="RDDS detector experiment registry")
    ap.add_argument("--list", action="store_true", help="list every experiment")
    ap.add_argument("--show", metavar="ID", help="print one experiment as JSON")
    ap.add_argument("--stage", metavar="E3", help="list one stage")
    ap.add_argument("--check", action="store_true", help="validate the registry")
    ap.add_argument("--budget", action="store_true", help="total estimated GPU hours")
    args = ap.parse_args()

    if args.show:
        print(json.dumps(get(args.show).to_dict(), indent=2, default=str))
        return 0

    if args.check:
        problems = check_registry()
        if problems:
            print("Registry problems:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"Registry clean: {len(REGISTRY)} experiments, no problems.")
        return 0

    if args.budget:
        total = sum(s.hours() for s in REGISTRY.values())
        by_stage: dict[str, float] = {}
        for s in REGISTRY.values():
            by_stage[s.stage] = by_stage.get(s.stage, 0.0) + s.hours()
        print("Estimated GPU hours (rough, verify against real epoch timings):\n")
        for stage in sorted(by_stage):
            print(f"  {stage:4s}  {by_stage[stage]:7.1f} h")
        print(f"  {'ALL':4s}  {total:7.1f} h")
        print(
            "\nAt ~$1.52/h on-demand ml.g5.2xlarge that is roughly "
            f"${total * 1.52:,.0f}; managed spot typically cuts this substantially. "
            "Verify current pricing before committing."
        )
        return 0

    specs = variants_of(args.stage) if args.stage else list(REGISTRY.values())
    if not specs:
        print(f"no experiments in stage {args.stage}")
        return 1

    for s in sorted(specs, key=lambda x: (x.stage, x.id)):
        size = f"{s.imgsz}" if isinstance(s.imgsz, int) else f"{s.imgsz[0]}x{s.imgsz[1]}"
        print(f"{s.id:24s} [{s.stage}] {s.title}")
        print(f"{'':24s} model={s.model} imgsz={size} epochs={s.epochs} "
              f"seeds={len(s.seeds)} ~{s.hours():.1f} GPU-h")
    print(f"\n{len(specs)} experiment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
