"""
ml/aws/launch.py
----------------
Submit experiments from ml/research/experiments.py as SageMaker training jobs.

Design decisions worth knowing before you change anything here.

MANAGED SPOT IS THE DEFAULT.
    These are multi-hour, fully restartable jobs reading immutable data - the
    archetypal spot workload. Spot typically cuts the bill substantially. The trade
    is interruption, which is handled by `checkpoint_s3_uri`: SageMaker syncs
    /opt/ml/checkpoints to S3 continuously and restores it on restart, so an
    interruption costs one checkpoint interval rather than the run. Pass --on-demand
    if you need a hard deadline instead of a cheap bill.

ONE JOB PER (EXPERIMENT, SEED).
    Not one job looping over seeds. A seed that crashes then takes its siblings with
    it, and a spot reclaim would lose all three. Separate jobs fail independently and
    the results are separately traceable.

THE REGISTRY IS THE SOURCE OF TRUTH.
    Hyperparameters are not repeated here. The launcher passes an experiment id; the
    training container resolves it from the same registry file. That is the only way
    to guarantee the job that ran matches the plan that described it.

PREREQUISITES
    pip install sagemaker boto3
    A SageMaker execution role with S3 read on the data prefix and write on the
    output prefix. Inside SageMaker Unified Studio, sagemaker.get_execution_role()
    picks this up automatically; from a laptop, pass --role explicitly.

USAGE
    # See what would be submitted, submit nothing
    python ml/aws/launch.py --stage E3 --data s3://bucket/nrdd2024/v1 --dry-run

    # Submit one experiment, all its seeds
    python ml/aws/launch.py --experiment E0-baseline \\
        --data s3://bucket/nrdd2024/v1 --output s3://bucket/rdds-research

    # Submit a whole stage
    python ml/aws/launch.py --stage E2 \\
        --data s3://bucket/nrdd2024/v1 --output s3://bucket/rdds-research

    # Check on what is running
    python ml/aws/launch.py --status
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.research.experiments import REGISTRY, ExperimentSpec  # noqa: E402
from ml.research.experiments import get as get_spec  # noqa: E402
from ml.research.experiments import variants_of  # noqa: E402

# Deep Learning Container framework versions. Pin these: an unpinned framework
# version means a rerun six months from now silently uses a different PyTorch and
# the comparison is no longer apples to apples.
FRAMEWORK_VERSION = "2.3.0"
PY_VERSION = "py311"

# Rough on-demand USD/hour, for the --dry-run estimate only. AWS pricing changes;
# these are a sanity check, not a quote. Verify at
# https://aws.amazon.com/sagemaker/ai/pricing/ before committing a budget.
APPROX_HOURLY: dict[str, float] = {
    "ml.g5.xlarge": 1.01,
    "ml.g5.2xlarge": 1.52,
    "ml.g5.4xlarge": 2.03,
    "ml.g5.12xlarge": 7.09,
    "ml.g6.2xlarge": 1.20,
    "ml.p3.2xlarge": 3.83,
}
SPOT_DISCOUNT = 0.30  # assume ~70% off; actual varies by region and capacity


def job_name(spec: ExperimentSpec, seed: int) -> str:
    """
    SageMaker job names: <=63 chars, alphanumerics and hyphens only, unique.
    Experiment ids contain dots and underscores, so sanitise rather than hope.
    """
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9-]", "-", spec.id).strip("-")
    name = f"rdds-{safe}-s{seed}-{stamp}"
    return name[:63].rstrip("-")


def build_estimator(
    spec: ExperimentSpec,
    seed: int,
    output_uri: str,
    role: Optional[str],
    instance: Optional[str],
    use_spot: bool,
    max_run_hours: int,
):
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        raise RuntimeError(
            "sagemaker SDK not installed: pip install sagemaker boto3"
        ) from None

    if role is None:
        try:
            role = sagemaker.get_execution_role()
        except Exception as exc:
            raise RuntimeError(
                "could not resolve a SageMaker execution role automatically. "
                "Pass --role arn:aws:iam::<acct>:role/<name>."
            ) from exc

    inst = instance or spec.instance
    max_run = max_run_hours * 3600
    name = job_name(spec, seed)

    kwargs = dict(
        entry_point="train_experiment.py",
        source_dir=str(ROOT),          # ships ml/ so the registry travels with the job
        role=role,
        instance_count=1,
        instance_type=inst,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        base_job_name=f"rdds-{spec.stage.lower()}",
        output_path=output_uri.rstrip("/") + "/models",
        code_location=output_uri.rstrip("/") + "/code",
        hyperparameters={
            "experiment": spec.id,
            "seed": seed,
            "workers": 8,
            "device": "0",
        },
        environment={
            # Ultralytics writes settings and datasets into the home dir by default,
            # which is not writable in every container. Point both at /tmp.
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            # Where class-set views are materialised inside the container.
            "SM_SCRATCH": "/tmp",
            # Passed through so the job logs to your managed MLflow. Absent = the
            # tracker no-ops and only the local artefacts are written.
            **({"MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"]}
               if os.environ.get("MLFLOW_TRACKING_URI") else {}),
            **({"MLFLOW_EXPERIMENT": os.environ["MLFLOW_EXPERIMENT"]}
               if os.environ.get("MLFLOW_EXPERIMENT") else {}),
        },
        max_run=max_run,
        # Metric regexes surface the numbers in the SageMaker console and in
        # CloudWatch, so a run can be watched without downloading artefacts.
        metric_definitions=[
            {"Name": "val:mAP50", "Regex": r"val\s+mAP50=([0-9.]+)"},
            {"Name": "val:mAP50-95", "Regex": r"val\s+mAP50=[0-9.]+\s+mAP50-95=([0-9.]+)"},
            {"Name": "test:mAP50", "Regex": r"test\s+mAP50=([0-9.]+)"},
            {"Name": "test:mAP50-95", "Regex": r"test\s+mAP50=[0-9.]+\s+mAP50-95=([0-9.]+)"},
        ],
    )

    if use_spot:
        kwargs.update(
            use_spot_instances=True,
            # Must exceed max_run; the gap is how long SageMaker may wait for
            # capacity plus the time lost to interruptions and restarts.
            max_wait=max_run + 4 * 3600,
            checkpoint_s3_uri=f"{output_uri.rstrip('/')}/checkpoints/{name}",
            checkpoint_local_path="/opt/ml/checkpoints",
        )

    # entry_point is relative to source_dir; the trainer lives in ml/detection/.
    kwargs["entry_point"] = "ml/detection/train_experiment.py"

    return PyTorch(**kwargs), name


def submit(
    specs: list[tuple[ExperimentSpec, int]],
    data_uri: str,
    output_uri: str,
    role: Optional[str],
    instance: Optional[str],
    use_spot: bool,
    max_run_hours: int,
    dry_run: bool,
) -> int:
    total_h = 0.0
    total_cost = 0.0
    print(f"\n{'experiment':24s} {'seed':>5s} {'dataset':10s} {'class set':14s} "
          f"{'~GPU-h':>7s} {'~USD':>7s}")
    print("-" * 78)

    for spec, seed in specs:
        inst = instance or spec.instance
        hours = spec.est_gpu_hours / max(len(spec.seeds), 1)
        rate = APPROX_HOURLY.get(inst, 1.52) * (SPOT_DISCOUNT if use_spot else 1.0)
        cost = hours * rate
        total_h += hours
        total_cost += cost
        print(f"{spec.id:24s} {seed:5d} {spec.dataset:10s} {spec.class_set:14s} "
              f"{hours:7.1f} {cost:7.2f}")

    mode = "managed spot" if use_spot else "on demand"
    print("-" * 78)
    print(f"{'TOTAL':24s} {len(specs):5d} jobs {'':25s} {total_h:7.1f} {total_cost:7.2f}  ({mode})")
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("\nNote: MLFLOW_TRACKING_URI is not set, so these jobs will write local "
              "artefacts only. Export it to also log to your managed MLflow.")
    print("\nCost figures are rough estimates from a hardcoded rate table and the "
          "registry's GPU-hour guesses. Verify against AWS pricing and your first "
          "real run's epoch timings before trusting them.\n")

    if dry_run:
        print("[dry-run] nothing submitted.")
        return 0

    if not data_uri.startswith("s3://") or not output_uri.startswith("s3://"):
        print("[error] --data and --output must be s3:// URIs", file=sys.stderr)
        return 1

    submitted: list[str] = []
    for spec, seed in specs:
        try:
            est, name = build_estimator(
                spec, seed, output_uri, role, instance, use_spot, max_run_hours
            )
            est.fit(inputs={"training": data_uri}, job_name=name, wait=False)
            submitted.append(name)
            print(f"[submitted] {name}")
        except Exception as exc:
            print(f"[failed] {spec.id} seed={seed}: {exc}", file=sys.stderr)

    print(f"\n{len(submitted)}/{len(specs)} jobs submitted.")
    if submitted:
        print("Watch with: python ml/aws/launch.py --status")
    return 0 if len(submitted) == len(specs) else 1


def show_status(limit: int = 25) -> int:
    try:
        import boto3
    except ImportError:
        print("boto3 not installed: pip install boto3", file=sys.stderr)
        return 1
    sm = boto3.client("sagemaker")
    resp = sm.list_training_jobs(MaxResults=limit, SortBy="CreationTime",
                                 SortOrder="Descending", NameContains="rdds")
    jobs = resp.get("TrainingJobSummaries", [])
    if not jobs:
        print("no rdds-* training jobs found")
        return 0
    print(f"{'job':46s} {'status':14s} {'created'}")
    print("-" * 82)
    for j in jobs:
        print(f"{j['TrainingJobName'][:46]:46s} {j['TrainingJobStatus']:14s} "
              f"{j['CreationTime']:%Y-%m-%d %H:%M}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit RDDS experiments to SageMaker")
    ap.add_argument("--experiment", help="single experiment id")
    ap.add_argument("--stage", help="submit a whole stage, e.g. E3")
    ap.add_argument("--all", action="store_true", help="submit every trainable experiment")
    ap.add_argument("--data", help="s3:// prefix of the staged dataset")
    ap.add_argument("--output", help="s3:// prefix for models, code and checkpoints")
    ap.add_argument("--role", help="SageMaker execution role ARN")
    ap.add_argument("--instance", help="override the instance type")
    ap.add_argument("--on-demand", action="store_true", help="disable managed spot")
    ap.add_argument("--max-run-hours", type=int, default=48)
    ap.add_argument("--seeds", help="comma-separated seed override")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="list recent jobs and exit")
    args = ap.parse_args()

    if args.status:
        return show_status()

    if args.experiment:
        chosen = [get_spec(args.experiment)]
    elif args.stage:
        chosen = variants_of(args.stage)
        if not chosen:
            print(f"no experiments in stage {args.stage}", file=sys.stderr)
            return 1
    elif args.all:
        chosen = [s for s in REGISTRY.values() if s.epochs > 0]
    else:
        ap.error("one of --experiment / --stage / --all is required")

    # epochs == 0 marks an evaluation-only experiment (E6, E7); those do not train.
    skipped = [s.id for s in chosen if s.epochs == 0]
    chosen = [s for s in chosen if s.epochs > 0]
    if skipped:
        print(f"[skip] evaluation-only, not training jobs: {', '.join(skipped)}")
    if not chosen:
        print("nothing to submit", file=sys.stderr)
        return 1

    seed_override = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else None
    )
    pairs = [(s, seed) for s in chosen for seed in (seed_override or s.seeds)]

    if not args.dry_run and (not args.data or not args.output):
        ap.error("--data and --output are required unless --dry-run")

    return submit(
        specs=pairs,
        data_uri=args.data or "",
        output_uri=args.output or "",
        role=args.role,
        instance=args.instance,
        use_spot=not args.on_demand,
        max_run_hours=args.max_run_hours,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
