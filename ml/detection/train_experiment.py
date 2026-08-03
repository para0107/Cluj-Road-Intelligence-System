"""
ml/detection/train_experiment.py
--------------------------------
Runs ONE experiment from ml/research/experiments.py at ONE seed, reproducibly.

This is both the local trainer and the SageMaker training-job entry point. The same
file runs unchanged in either context; ml.repro.sagemaker_paths() resolves the
difference.

Relationship to ml/detection/train.py:
    train.py produced the current production checkpoint and is kept as-is for
    provenance - do not edit it, the published numbers came from it. This module
    supersedes it for all research work, and fixes what makes train.py unusable for
    a research program:
      - no seed argument, so runs are not reproducible
      - no run manifest, so a metric cannot be traced to a commit or a config
      - no dataset fingerprint, so "which data produced this" is unanswerable
      - metrics only in Ultralytics' own results.csv, with no config alongside
      - no test-split evaluation, so every number was selection-biased

What one invocation produces, under runs/research/<ts>_<exp>_s<seed>/:
    run.json           git SHA, versions, GPU, seed, dataset hash, timings
    config.json        the fully resolved spec + hyperparameters
    metrics.csv        per-epoch train/val losses and mAP
    per_class_ap.json  per-class AP@50 on the TEST split (feeds ml/research/anisotropy.py)
    test_metrics.json  the headline numbers, computed once, at the end
    ultralytics/       Ultralytics' own run dir, including weights/best.pt

Usage (local):
    python ml/detection/train_experiment.py --experiment E0-baseline --seed 1337 \\
        --data /path/to/staged/dataset_nrdd2024_research.yaml

Usage (SageMaker): the launcher passes these as hyperparameters; the data channel is
mounted at SM_CHANNEL_TRAINING and the yaml is found inside it.

References:
    RT-DETR      Zhao et al., 2024. arXiv:2304.08069
    Ultralytics  https://docs.ultralytics.com/modes/train/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.repro import (  # noqa: E402
    RunDir,
    get_run_context,
    hash_dataset,
    sagemaker_paths,
    seed_everything,
)
from ml.research.class_sets import CLASS_SETS  # noqa: E402
from ml.research.class_sets import materialise as materialise_class_set  # noqa: E402
from ml.research.datasets import get_dataset  # noqa: E402
from ml.research.experiments import get as get_spec  # noqa: E402
from ml.tracking import Tracker  # noqa: E402


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def resolve_batch(requested: int, imgsz: int) -> int:
    """
    Pick a batch size from detected VRAM, scaled down for larger inputs.

    The original train.py hardcoded batch=4 for the 4 GB RTX 2050 it ran on. Carrying
    that constant onto a 24 GB A10G would waste most of the card and change the
    effective learning-rate/batch relationship the PSO search was tuned against, so
    it is recomputed here rather than inherited.

    Returns `requested` unchanged when it is non-zero: an explicit choice wins.
    """
    if requested and requested > 0:
        return requested
    try:
        import torch
    except ImportError:
        return 4
    if not torch.cuda.is_available():
        return 2
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if vram < 3.5:
        base = 2
    elif vram < 6.0:
        base = 4
    elif vram < 12.0:
        base = 8
    elif vram < 26.0:      # A10G / L4 24 GB
        base = 16
    elif vram < 48.0:
        base = 24
    else:
        base = 32

    # Activation memory scales roughly with pixel count.
    scale = (640 / max(imgsz, 1)) ** 2
    return max(2, int(base * min(1.0, scale)))


def find_data_yaml(explicit: Optional[str], data_dir: Path) -> Path:
    """Locate the dataset yaml, preferring an explicit path over a channel search."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        cand = data_dir / explicit
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"dataset yaml not found: {explicit}")

    candidates = sorted(data_dir.glob("*.yaml")) + sorted(data_dir.glob("**/*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no *.yaml under the data channel {data_dir}")
    preferred = [c for c in candidates if "research" in c.name]
    return (preferred or candidates)[0]


def rewrite_yaml_path(yaml_path: Path, new_root: Path, out_dir: Path) -> Path:
    """
    Rewrite the yaml's `path:` to where the data actually is.

    Necessary because the staged yaml records the path it was BUILT at, and in a
    SageMaker job the same data appears under /opt/ml/input/data/training. Writing a
    corrected copy is safer than mutating the input channel, which is read-only in
    some configurations.
    """
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("path:"):
            out.append(f"path: {new_root}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, f"path: {new_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / yaml_path.name
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def extract_metrics(results: Any) -> dict[str, float]:
    """
    Pull the headline numbers out of an Ultralytics results object.

    Written defensively: the attribute layout of `results.box` has moved between
    Ultralytics minor versions, and a KeyError here after a 12-hour training run is
    an expensive way to find that out.
    """
    m: dict[str, float] = {}
    box = getattr(results, "box", None)
    if box is None:
        return m
    for key, attr in (("mAP50", "map50"), ("mAP50-95", "map"),
                      ("precision", "mp"), ("recall", "mr")):
        try:
            v = getattr(box, attr, None)
            if v is not None:
                m[key] = float(v)
        except (TypeError, ValueError):
            continue
    p, r = m.get("precision"), m.get("recall")
    if p is not None and r is not None and (p + r) > 0:
        m["F1"] = 2 * p * r / (p + r)
    return m


def extract_per_class_ap(results: Any, names: dict | list) -> dict[str, float]:
    """
    Per-class AP@50, keyed by class NAME.

    Ultralytics returns `box.ap50` as an array ordered by the classes that were
    actually present in the evaluated split, with `box.ap_class_index` giving the
    original class ids. Zipping against the full name list without that indirection
    silently mislabels every class whenever one is absent - which is exactly the
    situation with a long-tailed dataset.
    """
    out: dict[str, float] = {}
    box = getattr(results, "box", None)
    if box is None:
        return out

    name_list = (
        [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    )
    ap50 = getattr(box, "ap50", None)
    idx = getattr(box, "ap_class_index", None)
    if ap50 is None:
        return out

    try:
        ap50 = [float(v) for v in ap50]
    except TypeError:
        return out

    if idx is not None:
        try:
            idx = [int(v) for v in idx]
        except TypeError:
            idx = None

    if idx is not None and len(idx) == len(ap50):
        for cid, ap in zip(idx, ap50):
            out[name_list[cid] if 0 <= cid < len(name_list) else f"class_{cid}"] = ap
    else:
        for cid, ap in enumerate(ap50):
            out[name_list[cid] if cid < len(name_list) else f"class_{cid}"] = ap
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_experiment(
    experiment: str,
    seed: int,
    data_yaml: Optional[str],
    epochs_override: Optional[int],
    batch_override: int,
    workers: int,
    device: str,
    runs_root: Optional[str],
    deterministic: bool,
    skip_test: bool,
) -> int:
    spec = get_spec(experiment)
    paths = sagemaker_paths()

    # -- imgsz. Ultralytics train() takes a single int; a non-square target is
    #    expressed as the long side plus rect=True, which batches images by aspect
    #    ratio instead of letterboxing them to a square. Handled explicitly rather
    #    than passing a tuple that fails deep inside the dataloader.
    if isinstance(spec.imgsz, tuple):
        imgsz, rect = max(spec.imgsz), True
        imgsz_note = f"non-square {spec.imgsz[0]}x{spec.imgsz[1]} via imgsz={imgsz}, rect=True"
    else:
        imgsz, rect = int(spec.imgsz), False
        imgsz_note = f"square {imgsz}"

    epochs = epochs_override if epochs_override is not None else spec.epochs
    batch = resolve_batch(batch_override or spec.batch, imgsz)

    seed_everything(seed, deterministic=deterministic)

    run = RunDir.create(
        root=runs_root or (paths["output"] / "runs"),
        name=f"{spec.id}_s{seed}",
    )
    print(f"[run] {run.path}")

    # -- data ---------------------------------------------------------------
    # The S3 channel holds ONE canonical copy of the dataset with its full class
    # schema. A class-set variant is derived here, at job start, into local scratch:
    # labels are rewritten and images are symlinked, which takes seconds. Uploading
    # one dataset copy per ablation would take hours per variant and is the reason
    # class-subset experiments usually do not get run.
    class_set = CLASS_SETS[spec.class_set]
    dataset_spec = get_dataset(spec.dataset)

    if spec.class_set == "all10" and spec.dataset == "nrdd2024":
        # Fast path: the canonical schema needs no rewriting.
        yaml_src = find_data_yaml(data_yaml, paths["data"])
        yaml_path = rewrite_yaml_path(yaml_src, paths["data"], run.path / "data")
        effective_root = paths["data"]
        print(f"[data] {yaml_src} -> {yaml_path} (root {effective_root})")
    else:
        view_root = Path(os.environ.get("SM_SCRATCH", "/tmp")) / f"view_{spec.class_set}"
        print(f"[data] deriving class set '{spec.class_set}' "
              f"({len(class_set.keep)} classes) from {paths['data']}")
        print(f"[data] dropping: {', '.join(class_set.dropped(dataset_spec.classes)) or '(none)'}")
        yaml_path = materialise_class_set(
            src_root=paths["data"],
            dst_root=view_root,
            class_set=class_set,
            source_names=dataset_spec.classes,
        )
        effective_root = view_root
        yaml_src = yaml_path

    try:
        # Fingerprint the CANONICAL data, not the derived view: the view is a
        # deterministic function of (canonical data, class set), both of which are
        # already recorded, so hashing it would add nothing and cost a full rescan.
        ds_hash = hash_dataset(paths["data"])
    except (FileNotFoundError, OSError) as exc:
        ds_hash = {"error": str(exc)}
        print(f"[data] WARNING: could not fingerprint dataset: {exc}", file=sys.stderr)

    # -- provenance, written BEFORE training so a crashed run is still traceable --
    ctx = get_run_context(seed=seed, extra={
        "experiment": spec.id,
        "stage": spec.stage,
        "dataset": spec.dataset,
        "dataset_citation": dataset_spec.citation,
        "class_set": spec.class_set,
        "n_classes": len(class_set.keep),
        "classes": class_set.output_names(),
        "classes_dropped": class_set.dropped(dataset_spec.classes),
        "dataset_hash": ds_hash,
        "dataset_yaml": str(yaml_src),
        "imgsz_note": imgsz_note,
        "resolved_batch": batch,
    })
    run.save_json("run.json", ctx)
    config = {
        **spec.to_dict(),
        "seed": seed,
        "resolved_epochs": epochs,
        "resolved_batch": batch,
        "resolved_imgsz": imgsz,
        "rect": rect,
        "device": device,
        "workers": workers,
        "deterministic": deterministic,
        "class_set_detail": class_set.to_dict(),
    }
    run.save_json("config.json", config)

    if ctx.get("git_dirty"):
        print("[warn] working tree is DIRTY. This run is not reportable - the code "
              "that produced it is not the code at the recorded SHA.", file=sys.stderr)

    # -- MLflow. Additive only: the local CSV and run.json above stay authoritative,
    #    and a tracking failure is never allowed to fail the job.
    tracker = Tracker.start(
        experiment=os.environ.get("MLFLOW_EXPERIMENT", "RDDS-detector"),
        run_name=f"{spec.id}_s{seed}",
        params={
            "experiment": spec.id, "stage": spec.stage, "seed": seed,
            "model": spec.model, "imgsz": imgsz, "rect": rect, "batch": batch,
            "epochs": epochs, "freeze_epochs": spec.freeze_epochs,
            "dataset": spec.dataset, "class_set": spec.class_set,
            "n_classes": len(class_set.keep),
            **spec.hyperparams(),
        },
        tags={
            "stage": spec.stage,
            "git_sha": ctx.get("git_sha") or "unknown",
            "git_dirty": ctx.get("git_dirty"),
            "gpu": ctx.get("gpu") or "none",
            "dataset_sha256": (ds_hash or {}).get("sha256", "unknown"),
            "hypothesis": spec.hypothesis[:480],
        },
    )

    # -- model --------------------------------------------------------------
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        print("[error] ultralytics not installed: pip install ultralytics", file=sys.stderr)
        return 1

    loader = RTDETR if "rtdetr" in spec.model.lower() else YOLO
    print(f"[model] {spec.model} via {loader.__name__}")

    shared: dict[str, Any] = dict(
        data=str(yaml_path),
        imgsz=imgsz,
        rect=rect,
        batch=batch,
        workers=workers,
        device=device,
        seed=seed,                 # Ultralytics' own RNG, separate from ours
        deterministic=deterministic,
        cache=False,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
        project=str(run.path.resolve() / "ultralytics"),
        name=spec.id,
        exist_ok=True,
        **spec.train_kwargs(),
    )

    # Continuous checkpointing so a managed-spot interruption costs one interval,
    # not the whole run.
    ckpt_dir = paths["checkpoints"]
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        ckpt_dir = None

    epoch_row = {"n": 0}

    def on_epoch_end(trainer) -> None:
        """Mirror Ultralytics' per-epoch metrics into our own CSV."""
        try:
            m = dict(getattr(trainer, "metrics", {}) or {})
            lr = getattr(trainer, "lr", {}) or {}
            row = {
                "epoch": int(getattr(trainer, "epoch", epoch_row["n"])),
                "phase": epoch_row.get("phase", "?"),
                "train_loss": float(getattr(trainer, "tloss", float("nan")).sum())
                if hasattr(getattr(trainer, "tloss", None), "sum")
                else "",
                "val_mAP50": m.get("metrics/mAP50(B)", ""),
                "val_mAP50-95": m.get("metrics/mAP50-95(B)", ""),
                "val_precision": m.get("metrics/precision(B)", ""),
                "val_recall": m.get("metrics/recall(B)", ""),
                "lr": next(iter(lr.values()), ""),
                "elapsed_s": round(time.time() - epoch_row["t0"], 1),
            }
            run.log_metrics(**row)
            tracker.log_metrics(
                {k: v for k, v in row.items() if k not in ("phase",)},
                step=row["epoch"],
            )
            v = m.get("metrics/mAP50-95(B)")
            if v is not None:
                run.note_best(float(v))
            epoch_row["n"] += 1
        except Exception as exc:  # never let logging kill a training run
            print(f"[log] epoch logging failed: {exc}", file=sys.stderr)

    epoch_row["t0"] = time.time()

    try:
        # -- phase 1: frozen backbone --------------------------------------
        if spec.freeze_epochs > 0 and epochs > spec.freeze_epochs:
            epoch_row["phase"] = "frozen"
            print(f"\n[phase 1] frozen backbone, {spec.freeze_epochs} epochs")
            model = loader(spec.model)
            model.add_callback("on_fit_epoch_end", on_epoch_end)
            model.train(**shared, epochs=spec.freeze_epochs, freeze=23, patience=999)

            last = run.path.resolve() / "ultralytics" / spec.id / "weights" / "last.pt"
            if not last.exists():
                print(f"[error] phase 1 produced no checkpoint at {last}", file=sys.stderr)
                return 1
            start_from, remaining = str(last), epochs - spec.freeze_epochs
        else:
            start_from, remaining = spec.model, epochs

        # -- phase 2: full fine-tune ---------------------------------------
        if remaining > 0:
            epoch_row["phase"] = "full"
            print(f"\n[phase 2] full fine-tune, {remaining} epochs, lr0 x0.1")
            hp2 = dict(shared)
            hp2["lr0"] = shared.get("lr0", 1e-4) * 0.1
            hp2["warmup_epochs"] = 1
            model = loader(start_from)
            model.add_callback("on_fit_epoch_end", on_epoch_end)
            model.train(**hp2, epochs=remaining, freeze=0, patience=20)

        best = run.path.resolve() / "ultralytics" / spec.id / "weights" / "best.pt"
        if not best.exists():
            print(f"[error] no best.pt at {best}", file=sys.stderr)
            return 1

        if ckpt_dir:
            try:
                shutil.copy2(best, ckpt_dir / f"{spec.id}_s{seed}_best.pt")
            except OSError as exc:
                print(f"[ckpt] copy failed: {exc}", file=sys.stderr)

        # -- final evaluation ------------------------------------------------
        results_out: dict[str, Any] = {}

        final = loader(str(best))
        val_res = final.val(data=str(yaml_path), split="val", imgsz=imgsz,
                            conf=0.001, iou=0.6, device=device, verbose=False)
        results_out["val"] = extract_metrics(val_res)
        results_out["val_per_class_AP50"] = extract_per_class_ap(
            val_res, getattr(final, "names", {}) or {}
        )

        if not skip_test:
            # The test split is touched exactly once, here, after all selection is
            # complete. Any earlier peek would recreate the bias this program exists
            # to remove.
            print("\n[test] evaluating on the held-out test split (once)")
            try:
                test_res = final.val(data=str(yaml_path), split="test", imgsz=imgsz,
                                     conf=0.001, iou=0.6, device=device, verbose=False)
                results_out["test"] = extract_metrics(test_res)
                results_out["test_per_class_AP50"] = extract_per_class_ap(
                    test_res, getattr(final, "names", {}) or {}
                )
            except Exception as exc:
                results_out["test_error"] = str(exc)
                print(f"[test] failed (is `test:` defined in the yaml?): {exc}",
                      file=sys.stderr)

        run.save_json("test_metrics.json", results_out)
        per_class = results_out.get("test_per_class_AP50") or results_out.get(
            "val_per_class_AP50", {}
        )
        run.save_json("per_class_ap.json", {
            "split": "test" if "test_per_class_AP50" in results_out else "val",
            "experiment": spec.id,
            "seed": seed,
            "per_class_AP50": per_class,
            "note": (
                "AP@IoU=0.50 per class, PASCAL VOC protocol (Everingham et al. 2010) "
                "as implemented by Ultralytics box.ap50. Feeds ml/research/anisotropy.py."
            ),
        })

        # Ship the checkpoint to the SageMaker model channel.
        try:
            paths["model"].mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, paths["model"] / f"{spec.id}_s{seed}.pt")
        except OSError as exc:
            print(f"[model] copy failed: {exc}", file=sys.stderr)

        run.finalize(extra={"results": results_out, "status": "complete"})

        # -- MLflow: final metrics, per-class AP, and the artefacts ---------
        for split in ("val", "test"):
            if split in results_out:
                tracker.log_metrics(
                    {f"final_{split}_{k}": v for k, v in results_out[split].items()}
                )
        for split in ("val", "test"):
            key = f"{split}_per_class_AP50"
            if key in results_out:
                tracker.log_metrics(
                    {f"AP50_{split}_{c}": v for c, v in results_out[key].items()}
                )
        tracker.log_artifact(run.path / "metrics.csv")
        tracker.log_artifact(run.path / "per_class_ap.json")
        tracker.log_artifact(run.path / "test_metrics.json")
        tracker.log_artifact(run.path / "config.json")
        tracker.log_artifact(run.path / "run.json")
        tracker.end(status="FINISHED")

        print(f"\n[done] {run.path}")
        print(f"  dataset={spec.dataset}  class_set={spec.class_set} "
              f"({len(class_set.keep)} classes)")
        for split in ("val", "test"):
            if split in results_out:
                m = results_out[split]
                print(f"  {split:5s} mAP50={m.get('mAP50', float('nan')):.4f}  "
                      f"mAP50-95={m.get('mAP50-95', float('nan')):.4f}  "
                      f"P={m.get('precision', float('nan')):.4f}  "
                      f"R={m.get('recall', float('nan')):.4f}")
        return 0

    except KeyboardInterrupt:
        run.finalize(extra={"status": "interrupted"})
        tracker.end(status="KILLED")
        return 130
    except Exception as exc:
        traceback.print_exc()
        run.finalize(extra={"status": "failed", "error": str(exc)})
        tracker.end(status="FAILED")
        # Non-zero exit marks the SageMaker job failed, which is what we want.
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Run one RDDS detector experiment")
    ap.add_argument("--experiment", required=True, help="id from ml/research/experiments.py")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data", help="dataset yaml (default: found in the data channel)")
    ap.add_argument("--epochs", type=int, help="override the spec's epoch count")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from VRAM")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs-root", help="override the run directory root")
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise reproducibility; slower")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip the held-out test evaluation")
    args = ap.parse_args()

    # SageMaker passes hyperparameters as --key value, but also exports them; accept
    # either so the same entry point works from the launcher and by hand.
    for env_key, attr in (("SM_HP_EXPERIMENT", "experiment"), ("SM_HP_SEED", "seed")):
        if os.environ.get(env_key) and not getattr(args, attr, None):
            setattr(args, attr, os.environ[env_key])

    return run_experiment(
        experiment=args.experiment,
        seed=int(args.seed),
        data_yaml=args.data,
        epochs_override=args.epochs,
        batch_override=args.batch,
        workers=args.workers,
        device=args.device,
        runs_root=args.runs_root,
        deterministic=args.deterministic,
        skip_test=args.skip_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
