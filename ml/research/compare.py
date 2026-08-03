"""
ml/research/compare.py
----------------------
Turn a pile of run directories into a defensible comparison.

The question this answers is the one that matters at a research programme: "is model
A actually better than model B, or did it get a lucky seed?" A leaderboard sorted by
mAP does not answer it. Two things are needed and both are implemented here.

1. ACROSS-SEED VARIANCE. E0 runs three seeds specifically to measure it. Any later
   difference smaller than that spread is noise, and this tool refuses to call it a
   win. This is why E0 comes first in the programme.

2. A PAIRED TEST OVER CLASSES. With 10 classes, per-class AP for two models is
   naturally paired: class by class, did A beat B? A paired permutation test on those
   10 differences is far more informative than comparing two aggregate mAP numbers,
   because it uses the per-class structure instead of averaging it away. It also
   directly answers the E4 gate, which is about the thin classes specifically rather
   than the mean.

HONEST LIMITATION, stated because it belongs in the paper too:
    The strongest test would bootstrap over TEST IMAGES, resampling the evaluation
    set to get a confidence interval on mAP itself. That requires per-image
    predictions, which Ultralytics' `val()` does not return in a convenient form.
    What is implemented here - seed variance plus a paired-over-classes permutation
    test - is weaker but is computed from artefacts every run already produces, and
    it is considerably better than the bare number comparison that is standard in
    this literature. If a reviewer asks for image-level bootstrap, dumping
    predictions to COCO JSON and resampling is the upgrade path.

Statistical references:
    Permutation tests    Ernst, 2004. doi:10.1214/088342304000000396
    Bootstrap CIs        Efron & Tibshirani, 1993, "An Introduction to the Bootstrap"
    Detector comparison  Everingham et al., 2010. doi:10.1007/s11263-009-0275-4

Usage:
    # Summarise every run found
    python ml/research/compare.py --runs runs/research

    # Head-to-head, with the per-class paired test
    python ml/research/compare.py --runs runs/research \\
        --baseline E0-baseline --challenger E3-1024sq

    # Focus the verdict on the classes E4's mechanism claim is about
    python ml/research/compare.py --runs runs/research \\
        --baseline E3-1024sq --challenger E4b-oriented-queries \\
        --focus longitudinal_crack,transverse_crack,rutting
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# Classes the E4 mechanism claim is about. Named here so the focus argument has a
# sensible default and the claim is not quietly redefined after seeing results.
THIN_CLASSES = ["longitudinal_crack", "transverse_crack", "rutting"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Run:
    path: Path
    experiment: str
    seed: Optional[int]
    git_sha: Optional[str]
    git_dirty: bool
    status: str
    metrics: dict           # {"val": {...}, "test": {...}}
    per_class: dict         # {class_name: AP50}
    per_class_split: str

    def metric(self, name: str, split: str = "test") -> Optional[float]:
        """Fetch one metric, falling back to val when test is unavailable."""
        m = self.metrics.get(split) or {}
        v = m.get(name)
        if v is None and split == "test":
            v = (self.metrics.get("val") or {}).get(name)
        return float(v) if v is not None else None


def load_runs(root: Path) -> list[Run]:
    """
    Read every run directory under `root`. Skips directories that are not runs
    rather than failing, so a partially-populated results folder still summarises.
    """
    runs: list[Run] = []
    for run_json in sorted(Path(root).rglob("run.json")):
        d = run_json.parent
        try:
            ctx = json.loads(run_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        metrics: dict = {}
        tm = d / "test_metrics.json"
        if tm.exists():
            try:
                raw = json.loads(tm.read_text(encoding="utf-8"))
                metrics = {k: v for k, v in raw.items() if k in ("val", "test")}
            except (json.JSONDecodeError, OSError):
                pass
        if not metrics and isinstance(ctx.get("results"), dict):
            metrics = {k: v for k, v in ctx["results"].items() if k in ("val", "test")}

        per_class, split = {}, "unknown"
        pc = d / "per_class_ap.json"
        if pc.exists():
            try:
                raw = json.loads(pc.read_text(encoding="utf-8"))
                per_class = {k: float(v) for k, v in raw.get("per_class_AP50", {}).items()}
                split = raw.get("split", "unknown")
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        runs.append(Run(
            path=d,
            experiment=ctx.get("experiment") or d.name,
            seed=ctx.get("seed"),
            git_sha=(ctx.get("git_sha") or "")[:8] or None,
            git_dirty=bool(ctx.get("git_dirty")),
            status=ctx.get("status", "unknown"),
            metrics=metrics,
            per_class=per_class,
            per_class_split=split,
        ))
    return runs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 1337
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean. Honest at n=3 seeds only in the sense that
    it reports how little we know - a 3-seed CI is wide, and it should look wide.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        means.append(statistics.fmean(rng.choices(values, k=n)))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def paired_permutation(
    a: Sequence[float], b: Sequence[float], n_perm: int = 100_000, seed: int = 1337
) -> dict:
    """
    Two-sided paired permutation test on the differences b - a.

    Under the null (no systematic difference) the sign of each paired difference is
    exchangeable, so the null distribution is generated by randomly flipping signs.
    This needs no normality assumption, which matters at n=10 classes where normality
    is untestable anyway.

    POWER FLOOR - important, and the reason `min_achievable_p` is returned.
        With n pairs there are only 2^n sign assignments, so the smallest two-sided
        p-value obtainable is 2/2^n = 2^(1-n):

            n=3  ->  0.250      n=6  ->  0.031
            n=4  ->  0.125      n=8  ->  0.008
            n=5  ->  0.063      n=10 ->  0.002

        At n=3 the test CANNOT return p<0.05 no matter how large the effect. Quoting
        "p=0.25, not significant" for three classes that all improved by 0.06 would
        be a misreading of the test, not a finding. Callers must check
        `underpowered` and describe the effect instead of testing it.
    """
    if len(a) != len(b) or len(a) < 3:
        return {"status": "insufficient_pairs", "n": len(a)}

    diffs = [y - x for x, y in zip(a, b)]
    observed = statistics.fmean(diffs)
    rng = random.Random(seed)
    n = len(diffs)

    hits = 0
    for _ in range(n_perm):
        m = statistics.fmean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(m) >= abs(observed):
            hits += 1

    min_p = 2.0 ** (1 - n)
    return {
        "status": "ok",
        "n_pairs": n,
        "mean_difference": observed,
        "median_difference": statistics.median(diffs),
        "p_value": (hits + 1) / (n_perm + 1),
        "min_achievable_p": min_p,
        "underpowered": min_p > 0.05,
        "n_classes_improved": sum(d > 0 for d in diffs),
        "n_classes_degraded": sum(d < 0 for d in diffs),
        "n_permutations": n_perm,
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised effect size for the paired difference."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [y - x for x, y in zip(a, b)]
    sd = statistics.stdev(diffs)
    return statistics.fmean(diffs) / sd if sd > 0 else float("inf")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarise(runs: list[Run], metric: str, split: str) -> list[dict]:
    """Group runs by experiment and report mean / spread / CI across seeds."""
    by_exp: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        if r.status in ("failed", "interrupted"):
            continue
        by_exp[r.experiment].append(r)

    out = []
    for exp, rs in by_exp.items():
        vals = [v for v in (r.metric(metric, split) for r in rs) if v is not None]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (float("nan"), float("nan"))
        out.append({
            "experiment": exp,
            "n_seeds": len(vals),
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "ci_low": lo,
            "ci_high": hi,
            "any_dirty": any(r.git_dirty for r in rs),
            "values": vals,
        })
    return sorted(out, key=lambda d: -d["mean"])


def mean_per_class(runs: list[Run], experiment: str) -> dict[str, float]:
    """Per-class AP averaged over an experiment's seeds."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.experiment != experiment:
            continue
        for k, v in r.per_class.items():
            acc[k].append(v)
    return {k: statistics.fmean(v) for k, v in acc.items() if v}


def head_to_head(
    runs: list[Run], baseline: str, challenger: str, focus: list[str],
    seed_noise: Optional[float], metric: str, split: str,
) -> dict:
    """Compare two experiments: aggregate, per-class paired test, and a verdict."""
    a_pc, b_pc = mean_per_class(runs, baseline), mean_per_class(runs, challenger)
    shared = sorted(set(a_pc) & set(b_pc))
    if not shared:
        return {"status": "no_shared_classes",
                "message": "no per_class_ap.json overlap between the two experiments"}

    a_vals = [a_pc[c] for c in shared]
    b_vals = [b_pc[c] for c in shared]
    test = paired_permutation(a_vals, b_vals)

    focus_shared = [c for c in focus if c in a_pc and c in b_pc]
    focus_test = None
    if len(focus_shared) >= 3:
        focus_test = paired_permutation(
            [a_pc[c] for c in focus_shared], [b_pc[c] for c in focus_shared]
        )

    summ = {s["experiment"]: s for s in summarise(runs, metric, split)}
    a_agg, b_agg = summ.get(baseline), summ.get(challenger)
    agg_delta = (b_agg["mean"] - a_agg["mean"]) if (a_agg and b_agg) else None

    # -- class-count guard -------------------------------------------------
    # Two models trained on different class SETS cannot be compared on aggregate
    # mAP: the mean is taken over different denominators, so dropping the hardest
    # classes raises it without improving a single prediction. When the counts
    # differ, the aggregate is suppressed entirely and the per-class paired test on
    # the shared classes becomes the only valid comparison. This enforces the E8 gate
    # rather than leaving it as a note somebody has to remember.
    n_a, n_b = len(a_pc), len(b_pc)
    class_count_mismatch = n_a != n_b
    aggregate_valid = not class_count_mismatch

    if class_count_mismatch:
        return {
            "status": "ok",
            "baseline": baseline,
            "challenger": challenger,
            "metric": metric,
            "split": split,
            "aggregate_delta": None,
            "aggregate_suppressed": True,
            "n_classes_baseline": n_a,
            "n_classes_challenger": n_b,
            "n_shared_classes": len(shared),
            "baseline_aggregate": a_agg,
            "challenger_aggregate": b_agg,
            "seed_noise_floor": seed_noise,
            "per_class": {c: {"baseline": a_pc[c], "challenger": b_pc[c],
                              "delta": b_pc[c] - a_pc[c]} for c in shared},
            "paired_test_all_classes": paired_permutation(
                [a_pc[c] for c in shared], [b_pc[c] for c in shared]
            ),
            "paired_test_focus_classes": focus_test,
            "focus_classes": focus_shared,
            "effect_size_cohens_d": cohens_d(
                [a_pc[c] for c in shared], [b_pc[c] for c in shared]
            ),
            "verdict": _class_set_verdict(
                baseline, challenger, n_a, n_b, shared, a_pc, b_pc, seed_noise
            ),
        }

    # Verdict, applying the noise floor E0 measured.
    if agg_delta is None:
        verdict = "Cannot compare: one of the experiments has no aggregate metric."
    elif seed_noise is not None and abs(agg_delta) < seed_noise:
        verdict = (
            f"NO DIFFERENCE. The aggregate gap ({agg_delta:+.4f}) is smaller than the "
            f"measured seed noise ({seed_noise:.4f}). This is not a result. Run more "
            f"seeds or accept that the change did nothing."
        )
    elif test.get("status") == "ok" and test["p_value"] < 0.05 and agg_delta > 0:
        verdict = (
            f"{challenger} BEATS {baseline}: {agg_delta:+.4f} {metric}, and the "
            f"per-class improvement is consistent "
            f"({test['n_classes_improved']}/{test['n_pairs']} classes, p="
            f"{test['p_value']:.4f})."
        )
    elif agg_delta > 0:
        verdict = (
            f"{challenger} scores higher ({agg_delta:+.4f}) but the per-class pattern "
            f"is not consistent (p={test.get('p_value', float('nan')):.4f}). The gain "
            f"may be concentrated in one or two classes - check the table before "
            f"claiming it."
        )
    else:
        verdict = f"{challenger} does NOT beat {baseline} ({agg_delta:+.4f} {metric})."

    # -- mechanism check ---------------------------------------------------
    # The E4 claim is about the thin classes specifically, so it is judged on those
    # and not on the mean. At the typical focus size (3 classes) a permutation test
    # is mathematically incapable of reaching p<0.05, so when `underpowered` is set
    # the check switches to a descriptive criterion - every focus class improved, and
    # by more than the measured seed noise - and says plainly that significance was
    # not established. Quoting the p-value there would misrepresent the test.
    if focus_test and focus_test.get("status") == "ok":
        fd = focus_test["mean_difference"]
        all_improved = focus_test["n_classes_degraded"] == 0
        margin = seed_noise if seed_noise is not None else 0.0

        if focus_test.get("underpowered"):
            if fd > 0 and all_improved and abs(fd) > margin:
                verdict += (
                    f" MECHANISM CHECK PASSES (descriptive): all "
                    f"{focus_test['n_pairs']} focus classes "
                    f"({', '.join(focus_shared)}) improved, by {fd:+.4f} on average, "
                    f"which exceeds the seed noise ({margin:.4f}). NOTE: with "
                    f"{focus_test['n_pairs']} classes the permutation test cannot go "
                    f"below p={focus_test['min_achievable_p']:.3f}, so this is a "
                    f"description of the effect, not a significance claim. To make it "
                    f"a significance claim, run more seeds and test across seeds."
                )
            else:
                verdict += (
                    f" MECHANISM CHECK FAILS: the focus classes change by {fd:+.4f} "
                    f"({focus_test['n_classes_improved']}/{focus_test['n_pairs']} "
                    f"improved). Any overall gain did not come from the stated "
                    f"mechanism. Report this rather than the headline number."
                )
        elif fd > 0 and focus_test["p_value"] < 0.10:
            verdict += (
                f" MECHANISM CHECK PASSES: the focus classes ({', '.join(focus_shared)}) "
                f"improve by {fd:+.4f} on average (p={focus_test['p_value']:.4f})."
            )
        else:
            verdict += (
                f" MECHANISM CHECK FAILS: the focus classes change by {fd:+.4f} "
                f"(p={focus_test['p_value']:.4f}). Any overall gain did not come from "
                f"the stated mechanism. Report this rather than the headline number."
            )

    return {
        "status": "ok",
        "baseline": baseline,
        "challenger": challenger,
        "metric": metric,
        "split": split,
        "aggregate_delta": agg_delta,
        "baseline_aggregate": a_agg,
        "challenger_aggregate": b_agg,
        "seed_noise_floor": seed_noise,
        "per_class": {c: {"baseline": a_pc[c], "challenger": b_pc[c],
                          "delta": b_pc[c] - a_pc[c]} for c in shared},
        "paired_test_all_classes": test,
        "paired_test_focus_classes": focus_test,
        "focus_classes": focus_shared,
        "effect_size_cohens_d": cohens_d(a_vals, b_vals),
        "verdict": verdict,
    }


def _class_set_verdict(
    baseline: str, challenger: str, n_a: int, n_b: int, shared: list[str],
    a_pc: dict[str, float], b_pc: dict[str, float], seed_noise: Optional[float],
) -> str:
    """
    Verdict for a comparison between two different class sets.

    Deliberately refuses to quote an aggregate. The only honest question is what
    happened to the classes that exist in both.
    """
    if not shared:
        return (
            f"{baseline} ({n_a} classes) and {challenger} ({n_b} classes) share no "
            f"classes. Nothing is comparable."
        )

    deltas = {c: b_pc[c] - a_pc[c] for c in shared}
    mean_d = statistics.fmean(deltas.values())
    improved = sum(v > 0 for v in deltas.values())
    margin = seed_noise if seed_noise is not None else 0.0

    head = (
        f"CLASS SETS DIFFER ({baseline}: {n_a} classes, {challenger}: {n_b}). "
        f"Aggregate mAP is NOT reported and must not be quoted - it averages over "
        f"different denominators, so removing hard classes inflates it without "
        f"improving any prediction. Judged on the {len(shared)} shared classes only: "
    )

    if abs(mean_d) < margin:
        return head + (
            f"mean change {mean_d:+.4f}, below the seed-noise floor ({margin:.4f}). "
            f"Changing the class set did nothing measurable to the classes that "
            f"remain."
        )
    if mean_d > 0:
        best = max(deltas.items(), key=lambda kv: kv[1])
        return head + (
            f"shared-class AP improves by {mean_d:+.4f} on average "
            f"({improved}/{len(shared)} classes up, largest gain {best[0]} "
            f"{best[1]:+.4f}). The removed classes were consuming capacity."
        )
    worst = min(deltas.items(), key=lambda kv: kv[1])
    return head + (
        f"shared-class AP DROPS by {mean_d:+.4f} ({improved}/{len(shared)} up, "
        f"largest loss {worst[0]} {worst[1]:+.4f}). The removed classes were "
        f"acting as useful hard negatives rather than dead weight - a lane line "
        f"resembles a crack, and labelling it may be what stopped the model calling "
        f"it one. This argues for keeping them as an auxiliary task, and it is a "
        f"more interesting finding than a gain would have been."
    )


def estimate_seed_noise(runs: list[Run], metric: str, split: str) -> Optional[float]:
    """
    The noise floor: the largest across-seed standard deviation among experiments that
    ran more than one seed. Used to refuse to call small differences wins.
    """
    stds = [s["std"] for s in summarise(runs, metric, split) if s["n_seeds"] > 1]
    return max(stds) if stds else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(summary: list[dict], h2h: Optional[dict], out: Path,
                 metric: str, split: str, noise: Optional[float]) -> Path:
    L = [
        "# RDDS detector — experiment comparison",
        "",
        f"Metric: **{metric}** on the **{split}** split.",
        "",
    ]
    if noise is not None:
        L += [
            f"Measured seed-noise floor: **{noise:.4f}**. Any difference smaller than "
            f"this is not a result.",
            "",
        ]
    else:
        L += [
            "> No multi-seed experiment has completed, so the noise floor is unknown "
            "and no difference below it can be judged. Run E0 with its three seeds "
            "before drawing conclusions from any comparison below.",
            "",
        ]

    L += ["## Leaderboard", "",
          "| Experiment | Seeds | Mean | Std | Min | Max | 95% CI |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for s in summary:
        ci = ("—" if math.isnan(s["ci_low"])
              else f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]")
        flag = " ⚠dirty" if s["any_dirty"] else ""
        L.append(
            f"| {s['experiment']}{flag} | {s['n_seeds']} | {s['mean']:.4f} | "
            f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {ci} |"
        )
    L += ["", "*⚠dirty = at least one run was produced from an uncommitted working "
              "tree and is not reportable.*", ""]

    if h2h and h2h.get("status") == "ok":
        L += [f"## {h2h['challenger']} vs {h2h['baseline']}", ""]
        if h2h.get("aggregate_suppressed"):
            L += [
                f"> **Aggregate suppressed.** The two runs use different class sets "
                f"({h2h['n_classes_baseline']} vs {h2h['n_classes_challenger']} "
                f"classes), so aggregate {metric} is not comparable and is "
                f"deliberately not reported. The {h2h['n_shared_classes']} shared "
                f"classes below are the only valid comparison.",
                "",
            ]
        else:
            L += [f"Aggregate delta: **{h2h['aggregate_delta']:+.4f}**", ""]
        L += ["| Class | Baseline | Challenger | Delta |", "|---|---:|---:|---:|"]
        for c, d in sorted(h2h["per_class"].items(), key=lambda kv: -kv[1]["delta"]):
            mark = " **(focus)**" if c in h2h.get("focus_classes", []) else ""
            L.append(f"| {c}{mark} | {d['baseline']:.4f} | {d['challenger']:.4f} | "
                     f"{d['delta']:+.4f} |")

        t = h2h["paired_test_all_classes"]
        if t.get("status") == "ok":
            L += ["", "### Paired permutation test, all classes", "",
                  f"- Mean per-class difference: **{t['mean_difference']:+.4f}**",
                  f"- Classes improved / degraded: "
                  f"**{t['n_classes_improved']} / {t['n_classes_degraded']}**",
                  f"- p-value: **{t['p_value']:.4f}** "
                  f"({t['n_permutations']:,} sign-flip permutations)",
                  f"- Cohen's d: {h2h['effect_size_cohens_d']:.3f}", ""]

        ft = h2h.get("paired_test_focus_classes")
        if ft and ft.get("status") == "ok":
            L += ["### Mechanism check, focus classes only", "",
                  f"Focus: {', '.join(h2h['focus_classes'])}", "",
                  f"- Mean difference: **{ft['mean_difference']:+.4f}**",
                  f"- Classes improved / degraded: "
                  f"**{ft['n_classes_improved']} / {ft['n_classes_degraded']}**"]
            if ft.get("underpowered"):
                L += [
                    f"- p-value: not reported. With {ft['n_pairs']} paired classes a "
                    f"sign-flip permutation test cannot fall below "
                    f"p={ft['min_achievable_p']:.3f}, so no significance claim is "
                    f"possible at this sample size. The criterion applied is "
                    f"descriptive: every focus class improved, by more than the "
                    f"measured seed noise.",
                ]
            else:
                L += [f"- p-value: **{ft['p_value']:.4f}**"]
            L += [""]

        L += ["### Verdict", "", h2h["verdict"], ""]

    L += [
        "## Method",
        "",
        "Across-seed spread is reported as the standard deviation plus a 10,000-sample "
        "percentile bootstrap CI for the mean (Efron & Tibshirani, 1993). Two "
        "experiments are compared by a two-sided paired permutation test over per-class "
        "AP@50, generating the null by random sign flips of the paired differences "
        "(Ernst, 2004) — no normality assumption, which matters at 10 classes.",
        "",
        "Limitation: the stronger test would bootstrap over test images rather than "
        "classes. That needs per-image predictions, which are not among the artefacts "
        "these runs currently emit.",
        "",
    ]

    out.mkdir(parents=True, exist_ok=True)
    p = out / "comparison.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Compare RDDS detector experiments")
    ap.add_argument("--runs", default="runs/research", help="root of run directories")
    ap.add_argument("--metric", default="mAP50-95")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--baseline")
    ap.add_argument("--challenger")
    ap.add_argument("--focus", help=f"comma-separated (default: {','.join(THIN_CLASSES)})")
    ap.add_argument("--out", default="runs/research/_comparison")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs))
    if not runs:
        print(f"no runs found under {args.runs}", file=sys.stderr)
        return 1
    print(f"[load] {len(runs)} run(s) from {args.runs}")

    dirty = [r for r in runs if r.git_dirty]
    if dirty:
        print(f"[warn] {len(dirty)} run(s) came from a dirty working tree and are "
              f"not reportable:", file=sys.stderr)
        for r in dirty[:5]:
            print(f"        {r.path.name}", file=sys.stderr)

    summary = summarise(runs, args.metric, args.split)
    if not summary:
        print(f"no runs have metric '{args.metric}' on split '{args.split}'",
              file=sys.stderr)
        return 1

    noise = estimate_seed_noise(runs, args.metric, args.split)

    print(f"\n{'experiment':26s} {'n':>3s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s}")
    print("-" * 68)
    for s in summary:
        print(f"{s['experiment']:26s} {s['n_seeds']:3d} {s['mean']:9.4f} "
              f"{s['std']:8.4f} {s['min']:9.4f} {s['max']:9.4f}")
    if noise is not None:
        print(f"\nseed-noise floor: {noise:.4f} "
              f"(differences below this are not results)")
    else:
        print("\nseed-noise floor UNKNOWN - no multi-seed experiment has completed.")

    h2h = None
    if args.baseline and args.challenger:
        focus = args.focus.split(",") if args.focus else THIN_CLASSES
        h2h = head_to_head(runs, args.baseline, args.challenger, focus,
                           noise, args.metric, args.split)
        if h2h.get("status") == "ok":
            print(f"\n{h2h['verdict']}")
        else:
            print(f"\n[h2h] {h2h.get('message', h2h.get('status'))}", file=sys.stderr)

    out = Path(args.out)
    report = write_report(summary, h2h, out, args.metric, args.split, noise)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps({"summary": summary, "head_to_head": h2h,
                    "seed_noise_floor": noise}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
