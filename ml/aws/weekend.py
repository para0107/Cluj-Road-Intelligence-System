"""
ml/aws/weekend.py
-----------------
Run the experiment queue on a temporary GPU account with a hard 72-hour deadline,
and get every result out before the account is destroyed.

WHY THIS EXISTS AND launch.py DOES NOT COVER IT

    The research-weekend accounts are not like a normal AWS account:

      1. `ec2:RunInstances` is blocked by IAM policy, and the participant guide warns
         that SageMaker AI classic returns a ValidationException. Submitting managed
         training jobs the way ml/aws/launch.py does may simply not work there.
         GPU access is the Unified Studio compute environment your notebook or Code
         Editor is already attached to.

      2. The account is deleted after exactly 72 hours, with no recovery. Anything
         not copied out is gone - checkpoints, metrics, figures, all of it.

    So on a weekend the model is inverted: instead of submitting jobs to elsewhere,
    you run experiments back to back on the GPU you are sitting on, and you export
    continuously rather than at the end.

    This module does three things launch.py cannot:
      - schedules a queue against a real wall-clock deadline and refuses to start a
        run that cannot finish before it
      - exports results to your PERSONAL bucket after every single run, so an
        unexpected end costs one experiment rather than the weekend
      - keeps going when one experiment fails, because a crash at hour 12 must not
        idle the remaining 60

USAGE

    # 1. What fits in the time left? Runs nothing.
    python ml/aws/weekend.py --plan --hours-left 70 \\
        --queue E0-baseline,E8-all10,E8-structural7

    # 2. Calibrate the estimates against YOUR gpu, using one short run
    python ml/aws/weekend.py --calibrate --data /path/to/dataset.yaml

    # 3. Run the queue, exporting after each experiment
    python ml/aws/weekend.py --run \\
        --queue E0-baseline,E8-all10,E8-structural7 \\
        --data /path/to/staged/dataset.yaml \\
        --export s3://<personal-bucket>/rdds-research \\
        --deadline "2026-08-10 18:00"

    # 4. Panic button: export everything right now
    python ml/aws/weekend.py --export-now s3://<personal-bucket>/rdds-research

RECOMMENDED QUEUE, in priority order. The reasoning is in RESEARCH_PROGRAM.md; the
short version is that E0 and E8 answer questions that change what everything else
means, so they go first even though they are not the most interesting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.research.experiments import GPU_FACTOR, estimate_hours  # noqa: E402
from ml.research.experiments import get as get_spec  # noqa: E402

# Priority order. E0 first because until the seed-noise floor is known, no later
# comparison can be judged; E8 second because it is the user's own question and is
# cheap. E4 last because it is the only item that can be dropped without invalidating
# the rest.
DEFAULT_QUEUE = [
    "E0-baseline",        # baseline + seed noise. Everything depends on it.
    "E8-all10",           # class-set control
    "E8-structural7",     # the class-selection question
    "E3-800sq",           # resolution, cheapest of the E3 rung
    "E3-1024x576",        # native dashcam aspect
    "E2-yolo11l",         # is the baseline just old?
    "E8-cracks_merged",   # is the loss in subtype confusion?
    "E5-recall",          # operating point
    "E3-1024sq",
    "E2-yolo12l",
    "E4a-striprf",        # the contribution, only if E1 supported it
    "E4c-ar-loss",
    "E4b-oriented-queries",
]

# Stop starting new runs this long before the deadline, leaving room for the final
# export. Copying a few GB to S3 is not instant and must not be what runs out of time.
EXPORT_RESERVE_H = 1.5


def _now() -> datetime:
    return datetime.now()


def parse_deadline(s: Optional[str], hours_left: Optional[float]) -> datetime:
    if s:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise SystemExit(f"could not parse --deadline '{s}' (use 'YYYY-MM-DD HH:MM')")
    if hours_left:
        return _now() + timedelta(hours=hours_left)
    # Default: assume a fresh 72-hour account.
    return _now() + timedelta(hours=72)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(data_yaml: str, device: str = "0") -> float:
    """
    Measure this GPU's actual seconds/epoch with a 2-epoch run, and report the
    implied GPU_FACTOR.

    Worth the ~10 minutes. The registry's estimates are extrapolated from an RTX 2050
    measurement with an assumed 3.5x speedup. If the real factor is 2x, a plan built
    on 3.5x overruns the weekend and the last experiments never run.
    """
    print("[calibrate] running 2 epochs to measure this GPU's throughput")
    out_root = Path("runs/research/_calibration")
    cmd = [
        sys.executable, str(ROOT / "ml/detection/train_experiment.py"),
        "--experiment", "E0-baseline", "--seed", "1337",
        "--data", data_yaml, "--epochs", "2", "--device", device,
        "--runs-root", str(out_root), "--skip-test",
    ]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = time.time() - t0

    if rc != 0:
        print("[calibrate] the run failed - fix that before planning a weekend "
              "around timings you do not have", file=sys.stderr)
        return GPU_FACTOR

    # 2 epochs plus fixed startup (weights download, dataset scan, validation).
    # Attribute a generous 180 s to startup so the per-epoch figure is not inflated.
    per_epoch = max((elapsed - 180) / 2.0, 1.0)
    factor = 740.0 / per_epoch

    print(f"\n[calibrate] {elapsed:.0f} s wall, ~{per_epoch:.0f} s/epoch")
    print(f"[calibrate] implied GPU_FACTOR = {factor:.2f} "
          f"(registry assumes {GPU_FACTOR})")
    if factor < GPU_FACTOR * 0.75:
        print(f"[calibrate] WARNING: this GPU is slower than the registry assumes. "
              f"Re-plan with --gpu-factor {factor:.1f} or the queue will overrun.",
              file=sys.stderr)
    return factor


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def plan(queue: list[str], deadline: datetime, gpu_factor: float) -> list[dict]:
    """Walk the queue against the clock, marking what fits and what does not."""
    cursor = _now()
    cutoff = deadline - timedelta(hours=EXPORT_RESERVE_H)
    rows: list[dict] = []

    for exp_id in queue:
        try:
            spec = get_spec(exp_id)
        except KeyError as exc:
            rows.append({"experiment": exp_id, "error": str(exc), "fits": False})
            continue

        for seed in spec.seeds:
            h = estimate_hours(spec.epochs, spec.imgsz, 1, gpu_factor)
            finish = cursor + timedelta(hours=h)
            fits = finish <= cutoff
            rows.append({
                "experiment": spec.id, "seed": seed, "hours": h,
                "starts": cursor, "finishes": finish, "fits": fits,
                "class_set": spec.class_set, "dataset": spec.dataset,
            })
            if fits:
                cursor = finish
    return rows


def print_plan(rows: list[dict], deadline: datetime, gpu_factor: float) -> None:
    cutoff = deadline - timedelta(hours=EXPORT_RESERVE_H)
    print(f"\nDeadline        {deadline:%Y-%m-%d %H:%M}")
    print(f"Last start by   {cutoff:%Y-%m-%d %H:%M}  "
          f"({EXPORT_RESERVE_H} h reserved for the final export)")
    print(f"GPU factor      {gpu_factor}  "
          f"(--calibrate to measure it instead of assuming)")
    print(f"\n{'experiment':24s} {'seed':>5s} {'class set':14s} {'h':>5s} "
          f"{'finishes':16s}  fits")
    print("-" * 78)

    fit_h = 0.0
    for r in rows:
        if r.get("error"):
            print(f"{r['experiment']:24s} {'':5s} {'':14s} {'':5s} "
                  f"{'UNKNOWN EXPERIMENT':16s}  no")
            continue
        mark = "yes" if r["fits"] else "NO"
        print(f"{r['experiment']:24s} {r['seed']:5d} {r['class_set']:14s} "
              f"{r['hours']:5.1f} {r['finishes']:%Y-%m-%d %H:%M}  {mark}")
        if r["fits"]:
            fit_h += r["hours"]

    n_fit = sum(1 for r in rows if r.get("fits"))
    n_out = len(rows) - n_fit
    print("-" * 78)
    print(f"{n_fit} run(s) fit in {fit_h:.1f} GPU-h. {n_out} will NOT fit.")
    if n_out:
        print("\nThe queue is ordered by priority, so what drops off the end is what "
              "matters least. If something further down matters more to you, reorder "
              "--queue rather than hoping it fits.")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(runs_root: Path, s3_uri: str, quiet: bool = False) -> bool:
    """
    Sync results to a bucket that survives the account. Returns True on success.

    Called after EVERY experiment, not once at the end. On a 72-hour account that
    disappears without warning, exporting once at the end is how a weekend of
    compute becomes nothing.
    """
    if not s3_uri:
        return False
    if not runs_root.exists():
        print(f"[export] nothing at {runs_root} yet", file=sys.stderr)
        return False

    cmd = ["aws", "s3", "sync", str(runs_root), s3_uri.rstrip("/") + "/runs",
           "--only-show-errors"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        print("[export] the aws CLI is not installed - results are NOT backed up",
              file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("[export] sync timed out after 1 h", file=sys.stderr)
        return False

    if r.returncode != 0:
        print(f"[export] FAILED: {r.stderr.strip()[:400]}", file=sys.stderr)
        return False
    if not quiet:
        print(f"[export] synced {runs_root} -> {s3_uri}/runs")
    return True


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_queue(
    queue: list[str], data_yaml: str, deadline: datetime, gpu_factor: float,
    export_uri: Optional[str], runs_root: Path, device: str,
) -> int:
    cutoff = deadline - timedelta(hours=EXPORT_RESERVE_H)
    runs_root.mkdir(parents=True, exist_ok=True)
    log = runs_root / "_weekend_log.json"
    history: list[dict] = []
    if log.exists():
        try:
            history = json.loads(log.read_text())
        except json.JSONDecodeError:
            history = []

    completed = failed = skipped = 0

    for exp_id in queue:
        try:
            spec = get_spec(exp_id)
        except KeyError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            skipped += 1
            continue

        for seed in spec.seeds:
            h = estimate_hours(spec.epochs, spec.imgsz, 1, gpu_factor)
            remaining = (cutoff - _now()).total_seconds() / 3600

            if remaining <= 0:
                print(f"\n[stop] past the last-start cutoff. "
                      f"Skipping {spec.id} and everything after it.")
                skipped += 1
                continue
            if h > remaining:
                print(f"[skip] {spec.id} s{seed} needs ~{h:.1f} h, only "
                      f"{remaining:.1f} h left before the export reserve")
                skipped += 1
                continue

            print(f"\n{'='*72}\n[run] {spec.id} seed={seed}  "
                  f"~{h:.1f} h  ({remaining:.1f} h remaining)\n{'='*72}")

            cmd = [
                sys.executable, str(ROOT / "ml/detection/train_experiment.py"),
                "--experiment", spec.id, "--seed", str(seed),
                "--data", data_yaml, "--device", device,
                "--runs-root", str(runs_root),
            ]
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            took = (time.time() - t0) / 3600

            entry = {
                "experiment": spec.id, "seed": seed, "returncode": rc,
                "estimated_h": h, "actual_h": round(took, 2),
                "finished_at": _now().isoformat(),
            }
            history.append(entry)
            log.write_text(json.dumps(history, indent=2))

            if rc == 0:
                completed += 1
                print(f"[done] {spec.id} s{seed} in {took:.2f} h "
                      f"(estimated {h:.1f} h)")
            else:
                failed += 1
                # Deliberately continue. One failure must not idle the GPU for the
                # rest of a window that cannot be extended.
                print(f"[FAILED] {spec.id} s{seed} rc={rc} - continuing with the "
                      f"queue", file=sys.stderr)

            # Export after EVERY run, success or failure. The logs from a failed run
            # are how you diagnose it after the account is gone.
            if export_uri:
                export(runs_root, export_uri, quiet=True)

            # Drift check: if reality is much slower than the estimate, say so now
            # rather than letting the plan quietly fall apart.
            if rc == 0 and took > h * 1.35:
                new_factor = gpu_factor * h / took
                print(f"[timing] this run took {took / h:.1f}x the estimate. "
                      f"Re-plan with --gpu-factor {new_factor:.1f}", file=sys.stderr)

    print(f"\n{'='*72}")
    print(f"completed {completed}   failed {failed}   skipped {skipped}")

    if export_uri:
        print("\n[export] final sync")
        ok = export(runs_root, export_uri)
        if not ok:
            print("\n!!! EXPORT FAILED. Copy runs/ out MANUALLY before the account "
                  "expires - there is no recovery after that. !!!", file=sys.stderr)
            return 1
        print(f"\nResults are safe at {export_uri}/runs")
    else:
        print("\n!!! No --export given, so NOTHING has been backed up. These files "
              "die with the account. !!!", file=sys.stderr)

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the experiment queue against a 72-hour GPU deadline"
    )
    ap.add_argument("--plan", action="store_true", help="show the schedule, run nothing")
    ap.add_argument("--run", action="store_true", help="execute the queue")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure this GPU with a 2-epoch run")
    ap.add_argument("--export-now", metavar="S3_URI",
                    help="sync results immediately and exit")
    ap.add_argument("--queue", help="comma-separated experiment ids "
                                    "(default: the recommended priority order)")
    ap.add_argument("--data", help="dataset yaml")
    ap.add_argument("--export", metavar="S3_URI",
                    help="PERSONAL bucket - must not be the temporary account's")
    ap.add_argument("--deadline", help="'YYYY-MM-DD HH:MM' when the account dies")
    ap.add_argument("--hours-left", type=float, help="alternative to --deadline")
    ap.add_argument("--gpu-factor", type=float, default=GPU_FACTOR)
    ap.add_argument("--runs-root", default="runs/research")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    if args.export_now:
        return 0 if export(runs_root, args.export_now) else 1

    if args.calibrate:
        if not args.data:
            ap.error("--calibrate needs --data")
        calibrate(args.data, args.device)
        return 0

    queue = args.queue.split(",") if args.queue else DEFAULT_QUEUE
    deadline = parse_deadline(args.deadline, args.hours_left)

    if args.plan or not args.run:
        print_plan(plan(queue, deadline, args.gpu_factor), deadline, args.gpu_factor)
        if not args.run:
            print("\n(--plan only. Add --run --data <yaml> --export s3://... to execute.)")
        return 0

    if not args.data:
        ap.error("--run needs --data")
    if not args.export:
        print("WARNING: no --export. On a temporary account everything you produce "
              "will be destroyed. Continuing in 10 s; Ctrl-C to stop.", file=sys.stderr)
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            return 130

    return run_queue(queue, args.data, deadline, args.gpu_factor,
                     args.export, runs_root, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
