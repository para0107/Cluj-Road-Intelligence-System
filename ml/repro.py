"""
ml/repro.py
-----------
Reproducibility and run-tracking primitives for the RDDS detector research program.

Every experiment in ml/research/RESEARCH_PROGRAM.md runs through this module, so that
each run carries its own provenance and no result has to be taken on trust.

What this guarantees for a run:
  - all four RNGs seeded (random, numpy, torch CPU, torch CUDA) from one call
  - git SHA + dirty flag + argv + library versions captured before training starts
  - per-epoch metrics in a CSV that can be diffed and plotted against another run
  - one run = one directory, so runs never clobber each other
  - a dataset manifest hash, so "which data produced this number" is answerable

Zero hard dependencies: torch and numpy are imported lazily, so the audit and analysis
tools can import this on a CPU-only box with no ML stack installed.

Adapted from the ml-experiment skill's repro.py, extended with dataset hashing and
SageMaker path resolution.

References:
    Determinism in PyTorch — https://pytorch.org/docs/stable/notes/randomness.html
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = [
    "seed_everything",
    "get_run_context",
    "RunDir",
    "hash_dataset",
    "sagemaker_paths",
    "is_sagemaker",
]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 1337, deterministic: bool = False) -> int:
    """
    Seed random / numpy / torch / CUDA from one call. Returns the seed so it can be
    logged by the caller.

    ``deterministic=True`` additionally forces deterministic algorithms and cuDNN.
    That is slower and a few ops have no deterministic kernel (they raise), so it
    defaults off. Turn it on only when bitwise-identical runs are the point.

    Note: Ultralytics sets its own seed internally from the ``seed`` train argument.
    Call this *and* pass ``seed=`` to ``model.train()`` — they cover different RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:  # some ops have no deterministic implementation
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    return seed


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git(*args: str) -> Optional[str]:
    """Run a git command, returning stripped stdout or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def get_run_context(seed: Optional[int] = None, extra: Optional[dict] = None) -> dict:
    """
    Snapshot everything needed to reproduce a run: git state, argv, library versions,
    hardware. Call this *before* training so the context reflects the starting state.

    ``git_dirty=True`` in a run.json means the code that produced the metric is not
    the code at that SHA. Treat such runs as non-reportable.
    """
    ctx: dict[str, Any] = {
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": seed,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "sagemaker": is_sagemaker(),
    }

    try:
        import torch

        ctx["torch"] = torch.__version__
        ctx["cuda"] = torch.version.cuda
        ctx["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            ctx["gpu"] = props.name
            ctx["gpu_count"] = torch.cuda.device_count()
            ctx["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
        else:
            ctx["gpu"] = None
    except ImportError:
        ctx["torch"] = None

    try:
        import ultralytics

        ctx["ultralytics"] = ultralytics.__version__
    except ImportError:
        ctx["ultralytics"] = None

    if extra:
        ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
def hash_dataset(
    root: Path | str,
    patterns: Iterable[str] = ("**/*.txt", "**/*.yaml"),
    include_image_names: bool = True,
) -> dict:
    """
    Produce a stable fingerprint of a dataset split so a run can prove which data it
    saw. Hashes label file *contents* (cheap, and the thing that actually changes)
    plus image file *names* (not contents, which would be gigabytes).

    Returns a dict with the digest and the counts that produced it, so a mismatch can
    be diagnosed rather than just detected.

    This is real work over real files — it does not fabricate a hash if the directory
    is missing; it raises.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    digest = hashlib.sha256()
    n_label_files = 0
    n_images = 0

    label_paths: list[Path] = []
    for pat in patterns:
        label_paths.extend(sorted(root.glob(pat)))

    for p in sorted(set(label_paths)):
        if not p.is_file():
            continue
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(p.read_bytes())
        n_label_files += 1

    if include_image_names:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_names = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.suffix.lower() in exts
        )
        n_images = len(img_names)
        for name in img_names:
            digest.update(name.encode())

    return {
        "root": str(root),
        "sha256": digest.hexdigest(),
        "n_label_files": n_label_files,
        "n_images": n_images,
    }


# ---------------------------------------------------------------------------
# SageMaker environment
# ---------------------------------------------------------------------------
def is_sagemaker() -> bool:
    """True when running inside a SageMaker training container."""
    return Path("/opt/ml").is_dir() and "SM_MODEL_DIR" in os.environ


def sagemaker_paths() -> dict[str, Path]:
    """
    Resolve the SageMaker container paths, falling back to local equivalents so the
    same entry point runs unchanged on a laptop and in a training job.

    SageMaker contract:
      SM_CHANNEL_<NAME>  input channels, downloaded from S3 before the job starts
      SM_MODEL_DIR       /opt/ml/model      -> tarred to S3 on success
      SM_OUTPUT_DATA_DIR /opt/ml/output/data -> tarred to S3 on success or failure
      /opt/ml/checkpoints                    -> synced continuously; survives spot
                                                interruption. This is the one that
                                                makes managed spot safe.
    """
    if is_sagemaker():
        return {
            "data": Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")),
            "model": Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")),
            "output": Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")),
            "checkpoints": Path("/opt/ml/checkpoints"),
        }

    root = Path(__file__).resolve().parents[1]
    return {
        "data": root / "data" / "detection",
        "model": root / "ml" / "weights",
        "output": root / "runs" / "research",
        "checkpoints": root / "runs" / "research" / "_checkpoints",
    }


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
class RunDir:
    """
    One run = one directory under ``<root>/<timestamp>_<name>/``.

    Layout produced:
        run.json      provenance (git, versions, host, seed, dataset hash, timings)
        config.json   the full resolved hyperparameter config
        metrics.csv   one row per epoch
        best.pt       best checkpoint by the monitored metric (when save_checkpoint used)
        last.pt       last checkpoint, for resume
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.path / "metrics.csv"
        self._metric_fields: Optional[list[str]] = None
        self._best: Optional[float] = None

        # Resume-safe: if metrics.csv already exists, adopt its header rather than
        # overwriting a partially completed run (matters for spot interruption).
        if self._metrics_path.exists():
            try:
                with self._metrics_path.open(newline="", encoding="utf-8") as f:
                    header = next(csv.reader(f), None)
                if header:
                    self._metric_fields = header
            except (OSError, StopIteration):
                pass

    @classmethod
    def create(cls, root: str | Path = "runs/research", name: str = "run") -> "RunDir":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(Path(root) / f"{ts}_{name}")

    # -- writing -----------------------------------------------------------
    def save_json(self, filename: str, obj: dict) -> Path:
        p = self.path / filename
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return p

    def log_metrics(self, **fields: Any) -> None:
        """Append one row to metrics.csv. The first call fixes the column order."""
        if self._metric_fields is None:
            self._metric_fields = list(fields.keys())
            with self._metrics_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._metric_fields)
        with self._metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([fields.get(k, "") for k in self._metric_fields])

    def save_checkpoint(
        self,
        model,
        optimizer=None,
        epoch: int = 0,
        metric: Optional[float] = None,
        monitor: str = "metric",
        mode: str = "max",
    ) -> None:
        """Always writes last.pt; writes best.pt only when ``metric`` improves."""
        try:
            import torch
        except ImportError:
            return

        state = {
            "epoch": epoch,
            "model": model.state_dict() if hasattr(model, "state_dict") else model,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            monitor: metric,
        }
        torch.save(state, self.path / "last.pt")

        if metric is not None:
            improved = self._best is None or (
                metric > self._best if mode == "max" else metric < self._best
            )
            if improved:
                self._best = metric
                torch.save(state, self.path / "best.pt")

    def note_best(self, metric: float, mode: str = "max") -> bool:
        """
        Record a metric without writing a checkpoint. For Ultralytics, which manages
        its own best.pt — we still want the best value stamped into run.json.
        """
        improved = self._best is None or (
            metric > self._best if mode == "max" else metric < self._best
        )
        if improved:
            self._best = metric
        return improved

    def finalize(self, run_json: str = "run.json", extra: Optional[dict] = None) -> None:
        """Stamp end time and best metric into run.json, merging with what's there."""
        p = self.path / run_json
        data: dict = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        data["best_metric"] = self._best
        if extra:
            data.update(extra)
        self.save_json(run_json, data)


if __name__ == "__main__":
    # Self-test: no torch, no GPU, no dataset required.
    import tempfile

    seed_everything(1337)
    with tempfile.TemporaryDirectory() as tmp:
        run = RunDir.create(root=tmp, name="selftest")
        run.save_json("config.json", {"lr": 1e-4, "epochs": 2})
        run.save_json("run.json", get_run_context(seed=1337))
        for e in range(3):
            run.log_metrics(
                epoch=e,
                train_loss=1.0 / (e + 1),
                val_loss=1.2 / (e + 1),
                map50=0.50 + e * 0.02,
            )
            run.note_best(0.50 + e * 0.02)
        run.finalize()
        print(f"wrote {run.path}")
        print((run.path / "metrics.csv").read_text())
        print(json.dumps(json.loads((run.path / "run.json").read_text()), indent=2))
