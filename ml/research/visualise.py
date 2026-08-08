"""
ml/research/visualise.py
------------------------
Turn run directories into figures you can put in a paper, and one HTML page that
shows everything at once.

DIVISION OF LABOUR WITH MLFLOW
    MLflow is the better tool for watching a run in progress - live curves, sortable
    parameter tables, no local files. Use it for that. It is the wrong tool for a
    thesis figure: you cannot control the styling, you cannot get a vector PDF, and a
    screenshot of a web UI is not a publication figure.

    So this module does what MLflow does not: static, styled, reproducible figures
    written from the same local artefacts compare.py reads, with no AWS session
    needed. Both read the same run directories, so they never disagree.

WHAT IT REFUSES TO PLOT
    Aggregate mAP across runs with different CLASS SETS. Removing the hardest
    classes raises the mean without improving any prediction, so a bar chart putting
    a 7-class run next to a 10-class run invites exactly the wrong conclusion. Where
    class counts differ, the class-ablation figure plots per-class AP on the SHARED
    classes only and says so on the figure itself. This mirrors the guard in
    compare.py; a figure that quietly contradicts the analysis is worse than no
    figure.

Usage:
    # Everything, from whatever runs exist
    python ml/research/visualise.py --runs runs/research --out runs/research/_figures

    # A specific head-to-head
    python ml/research/visualise.py --runs runs/research \\
        --compare E8-all10 E8-structural7 --out runs/research/_figures

    # Vector output for LaTeX
    python ml/research/visualise.py --runs runs/research --format pdf

Requires only matplotlib. No torch, no AWS, no network.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.research.compare import (  # noqa: E402
    Run,
    estimate_seed_noise,
    load_runs,
    mean_per_class,
    summarise,
)

# Colour-blind-safe qualitative palette (Okabe-Ito). Chosen because road-damage
# figures routinely carry 7-10 series and the default matplotlib cycle becomes
# indistinguishable in greyscale print.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
THIN_CLASSES = ["longitudinal_crack", "transverse_crack", "rutting"]


def _mpl():
    """Import and configure matplotlib, or explain why the figure cannot be made."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib is required: pip install matplotlib", file=sys.stderr)
        return None
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
    })
    return plt


# ---------------------------------------------------------------------------
# Reading per-epoch metrics
# ---------------------------------------------------------------------------
def read_metrics_csv(path: Path) -> dict[str, list[float]]:
    """
    Read a run's metrics.csv into columns of floats.

    Blank cells are common (Ultralytics does not emit every metric every epoch), so
    each column keeps only the rows where it actually has a value, paired with its
    epoch. Returning ragged columns is correct here; forward-filling would invent
    data points that were never measured.
    """
    if not path.exists():
        return {}
    cols: dict[str, list[float]] = {}
    epochs: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return {}

    for r in rows:
        try:
            e = float(r.get("epoch", "") or "nan")
        except ValueError:
            continue
        epochs.append(e)
        for k, v in r.items():
            if k == "epoch" or v in (None, ""):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            cols.setdefault(k, []).append(fv)
            cols.setdefault(f"__epoch__{k}", []).append(e)
    cols["epoch"] = epochs
    return cols


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_training_curves(runs: list[Run], out: Path, fmt: str,
                        metric: str = "val_mAP50-95") -> Optional[Path]:
    """Per-epoch validation metric, one line per run. The overfitting check."""
    plt = _mpl()
    if plt is None:
        return None

    series: list[tuple[str, list[float], list[float]]] = []
    for r in sorted(runs, key=lambda x: x.experiment):
        cols = read_metrics_csv(r.path / "metrics.csv")
        ys = cols.get(metric)
        xs = cols.get(f"__epoch__{metric}")
        if ys and xs and len(ys) == len(xs):
            label = f"{r.experiment}" + (f" (s{r.seed})" if r.seed else "")
            series.append((label, xs, ys))

    if not series:
        print(f"[figures] no run has '{metric}' in metrics.csv - skipping curves",
              file=sys.stderr)
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, xs, ys) in enumerate(series):
        ax.plot(xs, ys, label=label, lw=1.6, color=PALETTE[i % len(PALETTE)])
        ax.scatter([xs[ys.index(max(ys))]], [max(ys)], s=22, zorder=5,
                   color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("val_", "validation "))
    ax.set_title("Training progress (dots mark each run's best epoch)")
    ax.legend(fontsize=7, ncol=2)
    p = out / f"fig_training_curves.{fmt}"
    fig.savefig(p)
    plt.close(fig)
    return p


def _class_count(runs: list[Run], experiment: str) -> Optional[int]:
    """How many classes an experiment's runs reported AP for. None if unknown."""
    counts = {len(r.per_class) for r in runs if r.experiment == experiment and r.per_class}
    return counts.pop() if len(counts) == 1 else None


def fig_leaderboard(summary: list[dict], runs: list[Run], out: Path, fmt: str,
                    metric: str, noise: Optional[float]) -> Optional[Path]:
    """
    Horizontal bars with across-seed error bars, plus the noise floor drawn in.

    The noise band is the point of the figure: it makes visually obvious which
    differences are real and which are seed lottery, which a bare bar chart hides.

    CLASS-SET HONESTY. Aggregate mAP over 7 classes is not comparable to aggregate
    mAP over 10 - dropping the hardest classes raises the mean without improving a
    single prediction. So bars are COLOURED and GROUPED by class count, each is
    labelled with its count, and when more than one class set is present the figure
    carries an explicit warning. Without this the chart would silently contradict
    the guard in compare.py, which is the one thing a results figure must never do.
    """
    plt = _mpl()
    if plt is None or not summary:
        return None

    # Group by class count so like is drawn next to like.
    for s in summary:
        s["_ncls"] = _class_count(runs, s["experiment"])
    ordered = sorted(summary, key=lambda s: (-(s["_ncls"] or 0), -s["mean"]))
    rows = list(reversed(ordered))

    counts = sorted({s["_ncls"] for s in rows if s["_ncls"]}, reverse=True)
    mixed = len(counts) > 1
    colour_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(counts)}

    names = [
        f"{s['experiment']}" + (f"  [{s['_ncls']} cls]" if s["_ncls"] else "")
        for s in rows
    ]
    means = [s["mean"] for s in rows]
    errs = [s["std"] for s in rows]
    colours = [colour_of.get(s["_ncls"], "#999999") for s in rows]

    fig, ax = plt.subplots(figsize=(7.8, max(2.8, 0.46 * len(rows) + 1.8)))
    ax.barh(names, means, xerr=errs, capsize=3, color=colours, alpha=0.88,
            error_kw={"lw": 1.1, "ecolor": "#333333"})

    for i, s in enumerate(rows):
        tag = f"{s['mean']:.4f}"
        tag += (f" ±{s['std']:.4f} (n={s['n_seeds']})" if s["n_seeds"] > 1
                else "  (1 seed)")
        ax.text(s["mean"] + max(means) * 0.012, i, tag, va="center", fontsize=7)

    handles = []
    if noise is not None and means:
        # The noise band is anchored to the best run WITHIN each class set, not to
        # the global best, since a cross-class-set gap is not a real gap.
        same = [s["mean"] for s in rows if s["_ncls"] == (rows[-1]["_ncls"])]
        anchor = max(same) if same else max(means)
        ax.axvspan(anchor - noise, anchor, color="#D55E00", alpha=0.10)
        handles.append(ax.axvline(anchor - noise, color="#D55E00", ls="--", lw=1.0,
                                  label=f"seed noise ±{noise:.4f}"))
    if mixed:
        import matplotlib.patches as mpatches
        handles += [mpatches.Patch(color=colour_of[c], label=f"{c} classes")
                    for c in counts]
    if handles:
        # Above the axes, horizontal. Inside the axes the legend collides with the
        # value labels on whichever bar happens to be longest.
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  bbox_to_anchor=(1.0, 1.005), ncol=len(handles))

    ax.set_xlabel(metric)
    ax.set_xlim(0, max(means) * 1.34 if means else 1)
    # Extra pad so the title clears the legend strip sitting just above the axes.
    ax.set_title(f"{metric} by experiment", pad=24 if handles else 8)

    if mixed:
        fig.subplots_adjust(bottom=0.24)
        fig.text(
            0.01, 0.012,
            "WARNING: these runs use DIFFERENT CLASS SETS (see colours). Bars are "
            "NOT comparable across colours —\naggregate mAP averages over different "
            "denominators, so removing hard classes inflates it without improving\n"
            "any prediction. For those comparisons read fig_class_ablation instead.",
            fontsize=7, color="#8a4b1f",
        )

    p = out / f"fig_leaderboard.{fmt}"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_per_class(runs: list[Run], experiments: list[str], out: Path,
                  fmt: str) -> Optional[Path]:
    """
    Grouped per-class AP@50 bars.

    Only classes present in EVERY plotted experiment are drawn. When the runs have
    different class sets that intersection is the honest comparison, and the caption
    records what was excluded.
    """
    plt = _mpl()
    if plt is None:
        return None

    per_exp = {e: mean_per_class(runs, e) for e in experiments}
    per_exp = {e: v for e, v in per_exp.items() if v}
    if not per_exp:
        print("[figures] no per_class_ap.json found - skipping per-class figure",
              file=sys.stderr)
        return None

    shared = set.intersection(*(set(v) for v in per_exp.values()))
    if not shared:
        print("[figures] experiments share no classes - skipping", file=sys.stderr)
        return None

    excluded = sorted(set().union(*(set(v) for v in per_exp.values())) - shared)
    # Order by the first experiment's AP so the weak classes are visually grouped.
    first = next(iter(per_exp.values()))
    classes = sorted(shared, key=lambda c: first.get(c, 0.0))

    n_exp = len(per_exp)
    width = 0.8 / n_exp
    ys = range(len(classes))

    fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.36 * len(classes) + 1.6)))
    for i, (exp, vals) in enumerate(per_exp.items()):
        offs = [y + (i - (n_exp - 1) / 2) * width for y in ys]
        ax.barh(offs, [vals.get(c, 0.0) for c in classes], height=width,
                label=exp, color=PALETTE[i % len(PALETTE)], alpha=0.9)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [c + ("  *" if c in THIN_CLASSES else "") for c in classes], fontsize=8
    )
    ax.set_xlabel("AP@50")
    title = "Per-class AP@50"
    if excluded:
        title += f"  (shared classes only; {len(excluded)} excluded)"
    ax.set_title(title)
    ax.legend(fontsize=7)

    # Captions go in FIGURE coordinates below the axes. Placing them in axes
    # coordinates collides with the x-label once tight bbox shrinks the margins.
    note = "* elongated classes (the E4 mechanism claim)"
    if excluded:
        note += ("\nexcluded (not shared by every run): "
                 + ", ".join(excluded[:8]) + ("…" if len(excluded) > 8 else ""))
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.01, 0.012, note, fontsize=6.8, color="#666666")

    p = out / f"fig_per_class.{fmt}"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_class_ablation(runs: list[Run], out: Path, fmt: str) -> Optional[Path]:
    """
    The E8 figure: what removing classes did to the classes that remain.

    Deliberately plots DELTAS on shared classes rather than absolute mAP, because
    absolute aggregate mAP is not comparable across class sets. Every bar is
    'this class got better/worse by X when those classes were removed', which is the
    only claim the data supports.
    """
    plt = _mpl()
    if plt is None:
        return None

    e8 = sorted({r.experiment for r in runs if r.experiment.startswith("E8-")})
    baseline = "E8-all10" if "E8-all10" in e8 else (e8[0] if e8 else None)
    others = [e for e in e8 if e != baseline]
    if not baseline or not others:
        return None

    base_pc = mean_per_class(runs, baseline)
    if not base_pc:
        return None

    fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.34 * len(base_pc) + 1.8)))
    plotted = False

    for i, exp in enumerate(others):
        pc = mean_per_class(runs, exp)
        shared = sorted(set(pc) & set(base_pc), key=lambda c: base_pc[c])
        if not shared:
            continue
        deltas = [pc[c] - base_pc[c] for c in shared]
        offs = [
            y + (i - (len(others) - 1) / 2) * (0.8 / max(len(others), 1))
            for y in range(len(shared))
        ]
        ax.barh(offs, deltas, height=0.8 / max(len(others), 1),
                label=f"{exp} ({len(pc)} classes)",
                color=PALETTE[(i + 1) % len(PALETTE)], alpha=0.9)
        if not plotted:
            ax.set_yticks(range(len(shared)))
            ax.set_yticklabels(shared, fontsize=8)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axvline(0, color="#000000", lw=1.0)
    ax.set_xlabel(f"Change in AP@50 vs {baseline}")
    ax.set_title("Class-set ablation: effect on the classes that remain")
    ax.legend(fontsize=7)

    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.01, 0.012,
        "Aggregate mAP is deliberately not shown: it is not comparable across class "
        "sets.\nPositive = the class improved once the other classes were removed. "
        "Negative = those classes\nwere acting as useful hard negatives.",
        fontsize=6.8, color="#666666",
    )
    p = out / f"fig_class_ablation.{fmt}"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_precision_recall_balance(summary: list[dict], runs: list[Run], out: Path,
                                 fmt: str) -> Optional[Path]:
    """
    Precision against recall per experiment, with the diagonal drawn.

    Included because the project's operating problem is that recall sits BELOW
    precision, which is backwards for municipal survey. Points below the diagonal
    are mis-tuned for the application, and that is easier to see than to read off a
    table.
    """
    plt = _mpl()
    if plt is None:
        return None

    pts = []
    for s in summary:
        rs = [r for r in runs if r.experiment == s["experiment"]]
        ps = [r.metric("precision") for r in rs if r.metric("precision") is not None]
        rc = [r.metric("recall") for r in rs if r.metric("recall") is not None]
        if ps and rc:
            pts.append((s["experiment"], statistics.fmean(ps), statistics.fmean(rc)))
    if not pts:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = max(max(p for _, p, _ in pts), max(r for _, _, r in pts)) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", color="#999999", lw=1.0)
    ax.fill_between([0, lim], [0, 0], [0, lim], color="#D55E00", alpha=0.06)
    # Alternate the label offset. Runs that differ only slightly land almost on top
    # of each other, and two overlapping labels are worse than none.
    offsets = [(7, 3), (7, -11), (-7, 7), (-7, -13)]
    for i, (name, p_, r_) in enumerate(sorted(pts, key=lambda t: (t[1], t[2]))):
        dx, dy = offsets[i % len(offsets)]
        ax.scatter([p_], [r_], s=64, color=PALETTE[i % len(PALETTE)],
                   edgecolors="k", linewidths=0.6, zorder=5, label=name)
        ax.annotate(name, (p_, r_), fontsize=6.8, xytext=(dx, dy),
                    ha="left" if dx > 0 else "right",
                    textcoords="offset points", zorder=6)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Precision vs recall")
    ax.text(0.97, 0.05,
            "shaded: recall < precision\n(wrong side for municipal survey —\n"
            "a missed defect costs more than a\nfalse alarm an operator dismisses)",
            transform=ax.transAxes, ha="right", fontsize=6.8, color="#8a4b1f")
    p = out / f"fig_precision_recall.{fmt}"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _embed(path: Path) -> str:
    """Inline an image as base64 so the HTML file is portable on its own."""
    try:
        b = base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""
    mime = "image/png" if path.suffix == ".png" else "application/pdf"
    if path.suffix != ".png":
        return f'<p><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></p>'
    return f'<img src="data:{mime};base64,{b}" alt="{html.escape(path.name)}">'


def write_html(figures: list[Path], summary: list[dict], runs: list[Run],
               noise: Optional[float], metric: str, out: Path) -> Path:
    rows = "".join(
        f"<tr><td>{html.escape(s['experiment'])}</td><td>{s['n_seeds']}</td>"
        f"<td>{s['mean']:.4f}</td><td>{s['std']:.4f}</td>"
        f"<td>{s['min']:.4f}</td><td>{s['max']:.4f}</td>"
        f"<td>{'yes' if s['any_dirty'] else ''}</td></tr>"
        for s in summary
    )
    noise_note = (
        f"Seed-noise floor <b>{noise:.4f}</b>. Any difference smaller than this is "
        f"not a result."
        if noise is not None else
        "<b>No multi-seed experiment has completed</b>, so the noise floor is unknown "
        "and no difference can yet be judged. Run E0 with its three seeds first."
    )
    imgs = "".join(
        f"<section><h2>{html.escape(f.stem.replace('fig_', '').replace('_', ' ').title())}"
        f"</h2>{_embed(f)}</section>"
        for f in figures if f
    )
    doc = f"""<!doctype html><meta charset="utf-8">
<title>RDDS detector — results</title>
<style>
 body{{font:14px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;max-width:940px;
       margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{margin-bottom:.2rem}} h2{{margin-top:2rem;font-size:1.05rem;color:#333}}
 .sub{{color:#666;margin-top:0}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:.4rem .55rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f5f5f5}}
 img{{max-width:100%;border:1px solid #eee;border-radius:4px}}
 .note{{background:#fff8e6;border-left:3px solid #E69F00;padding:.7rem 1rem;
        margin:1rem 0;font-size:13px}}
</style>
<h1>RDDS detector — results</h1>
<p class="sub">{len(runs)} run(s) · metric <code>{html.escape(metric)}</code> ·
generated by <code>ml/research/visualise.py</code></p>
<div class="note">{noise_note}</div>
<h2>Leaderboard</h2>
<table><tr><th>Experiment</th><th>Seeds</th><th>Mean</th><th>Std</th>
<th>Min</th><th>Max</th><th>Dirty</th></tr>{rows}</table>
<div class="note">Aggregate scores are only comparable between runs with the
<b>same class set</b>. Where class counts differ, read the class-ablation figure,
which plots per-class deltas on the shared classes instead.</div>
{imgs}
<h2>Reproducing any number here</h2>
<p>Each run directory carries <code>run.json</code> (git SHA, seed, dataset hash),
<code>config.json</code> (resolved hyperparameters) and <code>metrics.csv</code>.
A run marked <b>Dirty</b> came from an uncommitted working tree and is not
reportable.</p>
"""
    out.mkdir(parents=True, exist_ok=True)
    p = out / "report.html"
    p.write_text(doc, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Figures and an HTML report from run dirs")
    ap.add_argument("--runs", default="runs/research")
    ap.add_argument("--out", default="runs/research/_figures")
    ap.add_argument("--metric", default="mAP50-95")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--compare", nargs="+", metavar="EXP",
                    help="restrict the per-class figure to these experiments")
    ap.add_argument("--curve-metric", default="val_mAP50-95")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs))
    if not runs:
        print(f"no runs found under {args.runs}\n"
              f"Have any jobs finished? Pull them first:\n"
              f"  aws s3 sync s3://<bucket>/rdds-research/models {args.runs}",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(runs, args.metric, args.split)
    noise = estimate_seed_noise(runs, args.metric, args.split)

    experiments = args.compare or [s["experiment"] for s in summary]

    figures = [
        fig_training_curves(runs, out, args.format, args.curve_metric),
        fig_leaderboard(summary, runs, out, args.format, args.metric, noise),
        fig_per_class(runs, experiments, out, args.format),
        fig_class_ablation(runs, out, args.format),
        fig_precision_recall_balance(summary, runs, out, args.format),
    ]
    made = [f for f in figures if f]

    report = write_html(made, summary, runs, noise, args.metric, out)

    print(f"[figures] {len(runs)} run(s), {len(made)} figure(s)")
    for f in made:
        print(f"  {f}")
    print(f"\nOpen this: {report}")
    if noise is None:
        print("\nNote: no multi-seed experiment yet, so no difference in these "
              "figures can be called a result. Run E0 first.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
