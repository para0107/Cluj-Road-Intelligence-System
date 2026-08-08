"""
ml/aws/stage_all.py
-------------------
Build every staged split a queue of experiments needs, in one command.

WHY THIS EXISTS
    E9 and E10 do not change the model. They change how `stage_dataset.py` built the
    splits, so each one needs its OWN staged dataset directory, and the directory name
    has to match the spec's `data_variant` exactly or `weekend.py --data-root` will not
    find it. Doing that by hand across ten variants is the kind of task that fails once,
    silently, and produces a number nobody can explain three weeks later.

    This script derives the list from the registry, so the staging and the experiments
    cannot drift apart.

WHAT IT PRODUCES

    <out>/staged_standard/dataset_nrdd2024_research.yaml
    <out>/staged_no_oversample/...
    <out>/staged_loco_norway/...
    <out>/staged_control_2803/...
    <out>/_staging_report.json

    Point `weekend.py --data-root <out>` at it and every experiment resolves its own
    split automatically.

USAGE

    # what would be built, and how long it will take
    python ml/aws/stage_all.py --source /tmp/src --out /tmp/data \\
        --queue E0-baseline,E9-oversamplenone,E10-loco-norway --dry-run

    # build everything the weekend-2 core queue needs
    python ml/aws/stage_all.py --source /tmp/src --out /tmp/data --queue-file queue.txt

    # build every variant in the registry
    python ml/aws/stage_all.py --source /tmp/src --out /tmp/data --all

EXIT CODES
    0  every requested variant staged and verified clean
    1  a variant failed to stage
    2  a variant staged but its leakage check was not clean
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.research.experiments import DATA_VARIANTS, REGISTRY  # noqa: E402
from ml.research.experiments import get as get_spec  # noqa: E402


def variants_for(queue: list[str]) -> dict[str, list[str]]:
    """Map each needed data variant to the experiment ids that require it."""
    needed: dict[str, list[str]] = {}
    for exp_id in queue:
        spec = get_spec(exp_id)
        needed.setdefault(spec.data_variant, []).append(spec.id)
    return needed


def stage_one(variant: str, source: Path, out_dir: Path, hash_algo: str,
              hash_threshold: int | None, dry_run: bool) -> dict:
    """Run stage_dataset.py for one variant. Returns a result record."""
    args = list(DATA_VARIANTS[variant]["stage_args"])

    # The variant recipes default to `--hash none`. An explicit --hash on this script
    # overrides them, so a calibration that came back `separated: true` can be applied
    # everywhere without editing the registry.
    if hash_algo != "none":
        if "--hash" in args:
            args[args.index("--hash") + 1] = hash_algo
        else:
            args += ["--hash", hash_algo]
        if hash_threshold is not None:
            args += ["--hash-threshold", str(hash_threshold)]

    cmd = [sys.executable, str(ROOT / "ml/aws/stage_dataset.py"),
           "--source", str(source), "--out", str(out_dir)] + args
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'=' * 72}\n[stage] {variant} -> {out_dir}\n"
          f"        {DATA_VARIANTS[variant]['description']}\n{'=' * 72}")
    print("  " + " ".join(cmd) + "\n")

    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    took = time.time() - t0

    manifest = out_dir / "manifest.json"
    clean = None
    sizes = None
    if not dry_run and manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            clean = m.get("leakage", {}).get("clean")
            sizes = m.get("split_sizes")
        except json.JSONDecodeError:
            pass

    return {"variant": variant, "out": str(out_dir), "returncode": rc,
            "seconds": round(took, 1), "leakage_clean": clean, "split_sizes": sizes,
            "experiments": []}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage every split a queue needs")
    ap.add_argument("--source", required=True, help="extracted N-RDD2024 root")
    ap.add_argument("--out", required=True,
                    help="directory to hold staged_<variant>/ trees; pass this same "
                         "path to weekend.py --data-root")
    ap.add_argument("--queue", help="comma-separated experiment ids")
    ap.add_argument("--queue-file", help="file with one experiment id per line")
    ap.add_argument("--all", action="store_true", help="stage every registry variant")
    ap.add_argument("--hash", dest="hash_algo", default="none",
                    choices=["dhash", "ahash", "none"],
                    help="override the recipes' hash choice. Leave at 'none' unless "
                         "stage_dataset.py --calibrate-hash reported separated: true")
    ap.add_argument("--hash-threshold", type=int,
                    help="use the value --calibrate-hash recommended")
    ap.add_argument("--dry-run", action="store_true", help="analyse, write nothing")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave a variant alone if its manifest already exists")
    args = ap.parse_args()

    source, out = Path(args.source), Path(args.out)
    if not source.is_dir():
        print(f"--source {source} is not a directory", file=sys.stderr)
        return 1

    if args.all:
        needed = {v: ["(all)"] for v in DATA_VARIANTS}
    else:
        queue: list[str] = []
        if args.queue:
            queue += [q.strip() for q in args.queue.split(",") if q.strip()]
        if args.queue_file:
            queue += [ln.strip() for ln in Path(args.queue_file).read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")]
        if not queue:
            ap.error("pass --queue, --queue-file, or --all")
        try:
            needed = variants_for(queue)
        except KeyError as exc:
            print(f"unknown experiment: {exc}", file=sys.stderr)
            return 1

    print(f"[plan] {len(needed)} data variant(s) needed for this queue:")
    for v, exps in sorted(needed.items()):
        print(f"  staged_{v:18s} <- {', '.join(sorted(exps))}")
    print(f"\n[plan] source={source}\n[plan] out={out}\n[plan] hash={args.hash_algo}")
    if args.hash_algo == "none":
        print("[plan] near-duplicate detection OFF: exact SHA-256 dedupe only. This is "
              "the safe default until --calibrate-hash says otherwise.")

    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for variant, exps in sorted(needed.items()):
        target = out / f"staged_{variant}"
        if args.skip_existing and (target / "manifest.json").exists():
            print(f"\n[skip] staged_{variant} already exists")
            results.append({"variant": variant, "out": str(target), "returncode": 0,
                            "skipped": True, "experiments": sorted(exps)})
            continue
        r = stage_one(variant, source, target, args.hash_algo,
                      args.hash_threshold, args.dry_run)
        r["experiments"] = sorted(exps)
        results.append(r)

    print(f"\n{'=' * 72}\n[summary]\n{'=' * 72}")
    bad_rc = [r for r in results if r["returncode"] != 0]
    unclean = [r for r in results if r.get("leakage_clean") is False]
    for r in results:
        status = "ok" if r["returncode"] == 0 else f"FAILED rc={r['returncode']}"
        leak = {True: "clean", False: "LEAKAGE", None: "-"}[r.get("leakage_clean")]
        sizes = r.get("split_sizes") or {}
        size_s = ("train/val/test "
                  f"{sizes.get('train','?')}/{sizes.get('val','?')}/{sizes.get('test','?')}"
                  if sizes else "")
        print(f"  {status:14s} {leak:8s} staged_{r['variant']:18s} {size_s}")

    if not args.dry_run:
        (out / "_staging_report.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[write] {out / '_staging_report.json'}")
        print(f"\nNext: weekend.py --data-root {out} "
              f"--data {out}/staged_standard/dataset_nrdd2024_research.yaml")

    if bad_rc:
        print(f"\n{len(bad_rc)} variant(s) failed to stage.", file=sys.stderr)
        return 1
    if unclean:
        print(f"\n{len(unclean)} variant(s) reported LEAKAGE - do not train on them.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
