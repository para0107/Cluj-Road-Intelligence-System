"""
ml/tracking.py
--------------
Optional MLflow logging, on top of (never instead of) the CSV and run.json that
ml/repro.py writes.

WHY BOTH

    You have a managed MLflow app in SageMaker Studio, and its run-comparison UI and
    model registry are better than anything worth rebuilding here. But MLflow becomes
    the single point of failure the moment it is the only record: a dropped network
    call, an expired session, or a deleted tracking server and a 12-hour training run
    has no result.

    So the local artefacts stay authoritative. ml/research/compare.py reads those,
    which means comparisons still work on a plane with no AWS session. MLflow is the
    nice front end, not the source of truth.

EVERY CALL HERE IS FAIL-SAFE
    A tracking failure must never fail a training job. Each method swallows its own
    exceptions and disables the tracker after the first failure, so a broken server
    costs one warning line rather than a run.

SETUP
    pip install mlflow sagemaker-mlflow

    Then point at your tracking server, by ARN:
        export MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-west-2:<acct>:mlflow-tracking-server/<name>

    Find the ARN in Studio under the MLflow app, or:
        aws sagemaker list-mlflow-tracking-servers --region us-west-2

    Inside a SageMaker training job the same variable is passed through by
    ml/aws/launch.py, so nothing extra is needed there.

Usage:
    from ml.tracking import Tracker

    tr = Tracker.start(experiment="RDDS-detector", run_name="E0-baseline_s1337",
                       params={...}, tags={...})
    tr.log_metrics({"val_mAP50": 0.56}, step=epoch)
    tr.log_artifact(run.path / "metrics.csv")
    tr.end(status="FINISHED")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

__all__ = ["Tracker", "mlflow_available"]

# MLflow rejects params over 500 chars and metric names with certain characters.
_MAX_PARAM_LEN = 480


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def _sanitise_key(k: str) -> str:
    """MLflow allows alphanumerics, underscore, dash, period, space, slash."""
    return "".join(c if (c.isalnum() or c in "_-. /") else "_" for c in str(k))[:250]


def _flatten(d: dict, prefix: str = "", out: Optional[dict] = None) -> dict:
    """
    Flatten nested config into MLflow's flat param space, truncating long values.

    Nested dicts are common here (a spec carries `resolved_hyperparams`), and MLflow
    params are strings only, so this is done once rather than at every call site.
    """
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, f"{key}.", out)
        elif isinstance(v, (list, tuple)):
            out[key] = str(list(v))[:_MAX_PARAM_LEN]
        else:
            out[key] = str(v)[:_MAX_PARAM_LEN]
    return out


class Tracker:
    """
    A no-op-safe MLflow wrapper.

    When MLflow is missing, unconfigured, or failing, every method is a silent no-op
    and `self.active` is False. Calling code never needs to branch on availability.
    """

    def __init__(self, run: Any = None, active: bool = False) -> None:
        self._run = run
        self.active = active
        self._mlflow = None
        if active:
            import mlflow
            self._mlflow = mlflow

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def start(
        cls,
        experiment: str = "RDDS-detector",
        run_name: Optional[str] = None,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
    ) -> "Tracker":
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            print("[mlflow] MLFLOW_TRACKING_URI not set - tracking disabled "
                  "(local CSV and run.json are unaffected)", file=sys.stderr)
            return cls(active=False)

        if not mlflow_available():
            print("[mlflow] mlflow not installed - tracking disabled. "
                  "pip install mlflow sagemaker-mlflow", file=sys.stderr)
            return cls(active=False)

        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_name=run_name)
            t = cls(run=run, active=True)
            if params:
                t.log_params(params)
            if tags:
                t.set_tags(tags)
            print(f"[mlflow] logging to {experiment}/{run_name} "
                  f"(run_id={run.info.run_id})")
            return t
        except Exception as exc:
            print(f"[mlflow] could not start run, tracking disabled: {exc}",
                  file=sys.stderr)
            return cls(active=False)

    def _guard(self, what: str, fn) -> None:
        """Run fn; on any failure warn once and disable tracking for the rest of the run."""
        if not self.active:
            return
        try:
            fn()
        except Exception as exc:
            print(f"[mlflow] {what} failed, disabling tracking: {exc}", file=sys.stderr)
            self.active = False

    # -- logging -----------------------------------------------------------
    def log_params(self, params: dict) -> None:
        self._guard("log_params", lambda: self._mlflow.log_params(
            {_sanitise_key(k): v for k, v in _flatten(params).items()}
        ))

    def set_tags(self, tags: dict) -> None:
        self._guard("set_tags", lambda: self._mlflow.set_tags(
            {_sanitise_key(k): str(v)[:_MAX_PARAM_LEN] for k, v in tags.items()}
        ))

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log numeric metrics. Non-numeric and NaN values are skipped, not coerced."""
        def _do() -> None:
            clean: dict[str, float] = {}
            for k, v in metrics.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f != f:  # NaN
                    continue
                clean[_sanitise_key(k)] = f
            if clean:
                self._mlflow.log_metrics(clean, step=step)

        self._guard("log_metrics", _do)

    def log_artifact(self, path: Path | str, artifact_path: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists():
            return
        self._guard(
            "log_artifact",
            lambda: (
                self._mlflow.log_artifacts(str(p), artifact_path)
                if p.is_dir()
                else self._mlflow.log_artifact(str(p), artifact_path)
            ),
        )

    def log_dict(self, obj: dict, filename: str) -> None:
        self._guard("log_dict", lambda: self._mlflow.log_dict(obj, filename))

    def end(self, status: str = "FINISHED") -> None:
        if not self.active:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            print(f"[mlflow] end_run failed: {exc}", file=sys.stderr)
        finally:
            self.active = False

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end(status="FAILED" if exc_type else "FINISHED")
        return False   # never swallow the caller's exception


if __name__ == "__main__":
    # Self-test: proves the no-op path is safe when nothing is configured.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    t = Tracker.start(run_name="selftest", params={"a": 1, "nested": {"b": 2}})
    t.log_metrics({"x": 1.0, "bad": "not-a-number", "nan": float("nan")}, step=0)
    t.log_artifact("/nonexistent/path")
    t.end()
    print(f"no-op path OK (active={t.active})")
    print("flatten:", _flatten({"a": 1, "n": {"b": [1, 2, 3]}}))
