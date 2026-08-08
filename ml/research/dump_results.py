"""
ml/research/dump_results.py
---------------------------
Collect every result into one compact, pasteable text digest.

Purpose: after a weekend of runs the findings are spread across a dozen JSON files
in timestamped directories. This gathers the parts that matter into a single block
small enough to paste into a conversation, an email, or a thesis appendix, and
leaves out the parts that do not (weights, per-epoch rows, Ultralytics plots).

    python ml/research/dump_results.py > /tmp/all_results.txt

Then open /tmp/all_results.txt and copy it.

Options:
    --runs DIR      run directory root (default runs/research)
    --full          include per-epoch metrics.csv tails and full configs
    --out FILE      write to a file as well as stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rule(title: str, ch: str = "=") -> str:
    return f"\n{ch * 74}\n{title}\n{ch * 74}"


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dump(runs_root: Path, full: bool = False) -> str:
    out: list[str] = []
    add = out.append

    add(_rule("RDDS DETECTOR — RESULTS DIGEST"))
    add(f"runs root: {runs_root.resolve()}")

    # -- the queue ledger --------------------------------------------------
    log = runs_root / "_weekend_log.json"
    if log.exists():
        add(_rule("QUEUE LEDGER (_weekend_log.json)", "-"))
        entries = _load(log) or []
        for e in entries:
            rc = e.get("returncode")
            tag = "OK " if rc == 0 else f"rc={rc}"
            add(f"  {tag:8s} {e.get('experiment'):22s} seed={e.get('seed')} "
                f"est={e.get('estimated_h')}h actual={e.get('actual_h')}h "
                f"{e.get('finished_at', '')[:19]}")

    # -- per-run results ---------------------------------------------------
    run_dirs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    for d in run_dirs:
        tm = _load(d / "test_metrics.json")
        rj = _load(d / "run.json")
        cf = _load(d / "config.json")
        if tm is None and rj is None:
            continue

        add(_rule(d.name, "-"))

        if rj:
            add(f"  status      : {rj.get('status')}")
            add(f"  experiment  : {rj.get('experiment')}   seed: {rj.get('seed')}")
            add(f"  class_set   : {rj.get('class_set')} ({rj.get('n_classes')} classes)")
            add(f"  dataset     : {rj.get('dataset')}")
            dh = (rj.get("dataset_hash") or {}).get("sha256")
            if dh:
                add(f"  data sha256 : {dh[:16]}…  "
                    f"({(rj.get('dataset_hash') or {}).get('n_images')} images)")
            add(f"  git         : {(rj.get('git_sha') or '?')[:8]}"
                f"{'  DIRTY' if rj.get('git_dirty') else ''}")
            add(f"  gpu         : {rj.get('gpu')}")
        if cf:
            add(f"  epochs      : {cf.get('resolved_epochs')} "
                f"(freeze {cf.get('freeze_epochs')})   "
                f"imgsz {cf.get('resolved_imgsz')}   batch {cf.get('resolved_batch')}")

        for split in ("test", "val"):
            m = (tm or {}).get(split)
            if not m:
                continue
            add(f"\n  [{split}]  mAP50={m.get('mAP50', float('nan')):.4f}  "
                f"mAP50-95={m.get('mAP50-95', float('nan')):.4f}  "
                f"P={m.get('precision', float('nan')):.4f}  "
                f"R={m.get('recall', float('nan')):.4f}  "
                f"F1={m.get('F1', float('nan')):.4f}")
            pc = (tm or {}).get(f"{split}_per_class_AP50")
            if pc:
                add(f"  [{split}] per-class AP@50:")
                for k, v in sorted(pc.items(), key=lambda kv: -kv[1]):
                    add(f"      {k:26s} {v:.4f}")

        if full:
            mcsv = d / "metrics.csv"
            if mcsv.exists():
                lines = mcsv.read_text(encoding="utf-8").splitlines()
                add("\n  metrics.csv (first + last 3 rows):")
                for ln in lines[:1] + lines[-3:]:
                    add(f"      {ln}")

    # -- comparison --------------------------------------------------------
    comp = runs_root / "_comparison" / "comparison.json"
    if comp.exists():
        add(_rule("COMPARISON (_comparison/comparison.json)", "-"))
        c = _load(comp) or {}
        nf = c.get("seed_noise_floor")
        add(f"  seed-noise floor: {nf}")
        for s in c.get("summary", []):
            add(f"  {s['experiment']:24s} n={s['n_seeds']} "
                f"mean={s['mean']:.4f} std={s['std']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f}"
                f"{'  DIRTY' if s.get('any_dirty') else ''}")
        h = c.get("head_to_head")
        if h and h.get("status") == "ok":
            add(f"\n  {h['challenger']} vs {h['baseline']}")
            if h.get("aggregate_suppressed"):
                add(f"    aggregate SUPPRESSED (class counts differ: "
                    f"{h.get('n_classes_baseline')} vs {h.get('n_classes_challenger')})")
            else:
                add(f"    aggregate delta: {h.get('aggregate_delta'):+.4f}")
            add("    per-class deltas:")
            for cls, v in sorted(h.get("per_class", {}).items(),
                                 key=lambda kv: -kv[1]["delta"]):
                add(f"      {cls:26s} {v['baseline']:.4f} -> {v['challenger']:.4f}  "
                    f"{v['delta']:+.4f}")
            t = h.get("paired_test_all_classes", {})
            if t.get("status") == "ok":
                add(f"    paired test: mean diff {t['mean_difference']:+.4f}, "
                    f"p={t['p_value']:.4f}, "
                    f"{t['n_classes_improved']}/{t['n_pairs']} improved")
            add(f"\n    VERDICT: {h.get('verdict')}")

    # -- anisotropy --------------------------------------------------------
    ani = runs_root / "E1_anisotropy" / "anisotropy.json"
    if ani.exists():
        add(_rule("E1 ANISOTROPY (E1_anisotropy/anisotropy.json)", "-"))
        a = _load(ani) or {}
        add("  class                       n_boxes  anisotropy  median_AR  area%")
        for r in sorted(a.get("per_class", []),
                        key=lambda x: -x["median_abs_log2_ar"]):
            add(f"  {r['class_name']:26s} {r['n_boxes']:7d}  "
                f"{r['median_abs_log2_ar']:10.3f}  {r['median_ar']:9.2f}  "
                f"{r['median_area_frac']*100:6.3f}")
        t = a.get("hypothesis_test", {})
        if t.get("status") == "ok":
            add(f"\n  rho(AP, anisotropy) = {t['rho_ap_vs_anisotropy']:+.3f}  "
                f"p = {t['p_ap_vs_anisotropy']:.4f}")
            add(f"  rho(AP, box area)   = {t['rho_ap_vs_area']:+.3f}  "
                f"p = {t['p_ap_vs_area']:.4f}")
            add(f"  rho(anisotropy, area) = {t['rho_anisotropy_vs_area']:+.3f}")
            add(f"\n  VERDICT: {t.get('verdict')}")

    # -- dataset provenance ------------------------------------------------
    kag = runs_root / "_kaggle_nrdd2024.json"
    if kag.exists():
        k = _load(kag) or {}
        add(_rule("DATASET PROVENANCE", "-"))
        add(f"  kaggle_id   : {k.get('kaggle_id')}")
        add(f"  format      : {k.get('format')}   schema_ok: {k.get('schema_ok')}")
        add(f"  images      : {k.get('n_images')}  yolo labels: {k.get('n_yolo_labels')}")
        add(f"  declared    : {k.get('declared_classes')}")

    add(_rule("END OF DIGEST"))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="One-file digest of all results")
    ap.add_argument("--runs", default="runs/research")
    ap.add_argument("--full", action="store_true",
                    help="include metrics.csv tails")
    ap.add_argument("--out", help="also write to this file")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    text = dump(root, full=args.full)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[written] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
