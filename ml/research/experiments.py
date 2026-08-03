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

    # hyperparameter deltas applied on top of BASELINE_HP
    overrides: dict[str, Any] = field(default_factory=dict)

    # free-form notes that belong with the result, not in a commit message
    notes: str = ""

    # compute hint for the launcher
    instance: str = "ml.g5.2xlarge"
    est_gpu_hours: float = 0.0

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


def _reg(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.id in REGISTRY:
        raise ValueError(f"duplicate experiment id: {spec.id}")
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
    est_gpu_hours=3 * 12,
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
        est_gpu_hours=12,
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
        est_gpu_hours=12 if _sz == 640 else 20,
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
    est_gpu_hours=22,
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
    est_gpu_hours=3 * 22,
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
    est_gpu_hours=22,
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
    est_gpu_hours=22,
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
    est_gpu_hours=1,
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
    est_gpu_hours=2,
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
        est_gpu_hours=12,
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
        est_gpu_hours=12,
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
        total = sum(s.est_gpu_hours for s in REGISTRY.values())
        by_stage: dict[str, float] = {}
        for s in REGISTRY.values():
            by_stage[s.stage] = by_stage.get(s.stage, 0.0) + s.est_gpu_hours
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
              f"seeds={len(s.seeds)} ~{s.est_gpu_hours:.0f} GPU-h")
    print(f"\n{len(specs)} experiment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
