"""
ml/research/anisotropy.py
-------------------------
E1 core analysis: does per-class detection accuracy fall as the class's boxes get
more elongated?

This is the experiment the whole E4 contribution is gated on. It tests the
hypothesis stated in ml/research/RESEARCH_PROGRAM.md section 2, Gap B:

    Road damage classes are separated by geometry and orientation, not appearance.
    A square-kernel, square-input detector is structurally mismatched to elongated
    targets, so elongated classes should score worse.

The suggestive evidence is the old 5-class RDD-2022 run, where the two elongated
classes (longitudinal 0.174, transverse 0.126) were the two worst and the two compact
classes (pothole 0.313, alligator 0.231) were the two best. Four points on a different
dataset is a hypothesis, not a result. This script tests it properly on N-RDD2024.

What it does NOT do: general class-balance, box-size and leakage auditing. Run the
`dataset-audit` skill for that - it covers those and this script deliberately does not
duplicate them.

Statistics note:
    Spearman rank correlation is used because the relationship is expected to be
    monotonic but not linear, and because n = 10 classes is far too small to trust a
    parametric assumption. The p-value comes from a random permutation test rather
    than a t-approximation, which is the honest choice at this sample size
    (Ernst, 2004, "Permutation Methods: A Basis for Exact Inference",
    doi:10.1214/088342304000000396).

Dependencies: PIL for image headers, matplotlib for the figure (optional).
No scipy required - the rank correlation and permutation test are implemented here
so this runs in a bare SageMaker processing container.

Usage:
    # Aspect-ratio statistics only (no AP available yet)
    python ml/research/anisotropy.py \
        --labels /data/nrdd/train/labels --images /data/nrdd/train/images \
        --out runs/research/E1_anisotropy

    # The actual hypothesis test, once E0 has produced per-class AP
    python ml/research/anisotropy.py \
        --labels /data/nrdd/test/labels --images /data/nrdd/test/images \
        --per-class-ap runs/research/<E0 run>/per_class_ap.json \
        --out runs/research/E1_anisotropy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

# N-RDD2024 10-class schema (Kaya & Codur, doi:10.17632/27c8pwsd6v.3).
# Order must match the training class ids - mirrors pipeline/detector.py CLASS_NAMES.
CLASS_NAMES: list[str] = [
    "longitudinal_crack",        # 0  D00
    "transverse_crack",          # 1  D10
    "alligator_crack",           # 2  D20
    "repaired_crack",            # 3  D30
    "pothole",                   # 4  D40
    "pedestrian_crossing_blur",  # 5  D50
    "lane_line_blur",            # 6  D60
    "manhole_cover",             # 7  D70
    "patchy_road",               # 8  D80
    "rutting",                   # 9  D90
]

# Classes the hypothesis predicts are elongated. Declared UP FRONT, before looking at
# the data, so the prediction is falsifiable rather than fitted after the fact.
PREDICTED_ELONGATED = {"longitudinal_crack", "transverse_crack", "rutting", "lane_line_blur"}
PREDICTED_COMPACT = {"pothole", "alligator_crack", "manhole_cover"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Statistics (no scipy)
# ---------------------------------------------------------------------------
def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks, averaging ties (the standard Spearman tie correction)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks."""
    return _pearson(_ranks(x), _ranks(y))


def permutation_p(
    x: Sequence[float], y: Sequence[float], n_perm: int = 100_000, seed: int = 1337
) -> tuple[float, int]:
    """
    Two-sided p-value for Spearman's rho by random permutation.

    Shuffles y against x n_perm times and counts how often |rho_permuted| >= |rho|.
    At n = 10 classes an exact test would need 10! = 3.6M permutations; 100k random
    ones give a p-value accurate to about +/-0.002, which is ample.

    Returns (p_value, n_perm_used).
    """
    observed = abs(spearman(x, y))
    if math.isnan(observed):
        return float("nan"), 0
    rng = random.Random(seed)
    y_list = list(y)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(y_list)
        if abs(spearman(x, y_list)) >= observed:
            hits += 1
    # Add-one correction: a permutation p-value should never be exactly 0.
    return (hits + 1) / (n_perm + 1), n_perm


# ---------------------------------------------------------------------------
# Data scanning
# ---------------------------------------------------------------------------
def _image_size(path: Path) -> Optional[tuple[int, int]]:
    """Read image dimensions from the header only (fast - no pixel decode)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan_boxes(labels_dir: Path, images_dir: Path) -> dict[int, list[dict]]:
    """
    Read YOLO labels and convert each box to PIXEL aspect ratio.

    This conversion is the whole point and is easy to get wrong. YOLO stores
    ``cls cx cy w h`` with w and h normalised to image WIDTH and HEIGHT separately.
    On a 16:9 frame a normalised w == h is NOT a square box. Pixel aspect ratio is
    therefore (w * W) / (h * H), and using the normalised values directly would
    manufacture a spurious 1.78x anisotropy in every class.

    Returns {class_id: [ {ar, log2_ar, area_frac, w_px, h_px}, ... ]}
    Raises if the directories do not exist - it will not silently return empty.
    """
    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    by_class: dict[int, list[dict]] = defaultdict(list)
    size_cache: dict[str, tuple[int, int]] = {}
    n_labels = n_boxes = n_skipped_no_image = n_malformed = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise RuntimeError(f"no .txt label files in {labels_dir}")

    for lf in label_files:
        n_labels += 1
        img = _find_image(images_dir, lf.stem)
        if img is None:
            n_skipped_no_image += 1
            continue

        if lf.stem not in size_cache:
            size = _image_size(img)
            if size is None:
                n_skipped_no_image += 1
                continue
            size_cache[lf.stem] = size
        W, H = size_cache[lf.stem]
        if W <= 0 or H <= 0:
            continue

        for line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                if line.strip():
                    n_malformed += 1
                continue
            try:
                cls = int(float(parts[0]))
                wn, hn = float(parts[3]), float(parts[4])
            except ValueError:
                n_malformed += 1
                continue
            if wn <= 0 or hn <= 0:
                n_malformed += 1
                continue

            w_px, h_px = wn * W, hn * H
            ar = w_px / h_px
            by_class[cls].append({
                "ar": ar,
                "log2_ar": math.log2(ar),
                "area_frac": wn * hn,
                "w_px": w_px,
                "h_px": h_px,
            })
            n_boxes += 1

    print(
        f"[scan] {n_labels} label files, {n_boxes} boxes, "
        f"{len(by_class)} classes present, "
        f"{n_skipped_no_image} skipped (no readable image), "
        f"{n_malformed} malformed lines",
        file=sys.stderr,
    )
    if n_boxes == 0:
        raise RuntimeError(
            "no valid boxes parsed - check that labels are YOLO format and that "
            "image stems match label stems"
        )
    return dict(by_class)


def per_class_stats(by_class: dict[int, list[dict]], names: list[str]) -> list[dict]:
    """
    Summarise each class's geometry.

    The anisotropy measure is median |log2(aspect ratio)|. Absolute value because a
    class elongated *vertically* (log2 AR = -2) is exactly as anisotropic as one
    elongated *horizontally* (log2 AR = +2) - and D00 vs D10 are precisely that pair,
    so signing it would cancel the effect being measured. Log because AR is a ratio:
    2:1 and 1:2 should be equidistant from square.
    """
    rows = []
    for cls in sorted(by_class):
        boxes = by_class[cls]
        logs = [b["log2_ar"] for b in boxes]
        abs_logs = [abs(v) for v in logs]
        areas = [b["area_frac"] for b in boxes]
        rows.append({
            "class_id": cls,
            "class_name": names[cls] if 0 <= cls < len(names) else f"class_{cls}",
            "n_boxes": len(boxes),
            "median_abs_log2_ar": statistics.median(abs_logs),
            "mean_abs_log2_ar": statistics.fmean(abs_logs),
            "median_log2_ar": statistics.median(logs),   # signed: which way it points
            "median_ar": statistics.median([b["ar"] for b in boxes]),
            "p10_ar": _quantile([b["ar"] for b in boxes], 0.10),
            "p90_ar": _quantile([b["ar"] for b in boxes], 0.90),
            "median_area_frac": statistics.median(areas),
            "pct_boxes_under_1pct_area": 100.0 * sum(a < 0.01 for a in areas) / len(areas),
        })
    return rows


def _quantile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    idx = q * (len(s) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ---------------------------------------------------------------------------
# The hypothesis test
# ---------------------------------------------------------------------------
def test_hypothesis(rows: list[dict], per_class_ap: dict[str, float]) -> dict:
    """
    Correlate per-class AP against per-class anisotropy.

    A strong NEGATIVE correlation supports the hypothesis: more elongated classes are
    detected worse. Also reports the correlation against median box area, because
    "elongated classes are just small classes" is the obvious confound and it has to
    be addressed before claiming the shape story.
    """
    paired = [(r, per_class_ap[r["class_name"]])
              for r in rows if r["class_name"] in per_class_ap]

    if len(paired) < 4:
        return {
            "status": "insufficient_data",
            "n_classes_matched": len(paired),
            "message": (
                "Fewer than 4 classes matched between the AP file and the label scan. "
                "Check that the AP file's class names match the N-RDD2024 schema."
            ),
        }

    aniso = [r["median_abs_log2_ar"] for r, _ in paired]
    area = [r["median_area_frac"] for r, _ in paired]
    ap = [a for _, a in paired]

    rho_aniso = spearman(aniso, ap)
    p_aniso, n_perm = permutation_p(aniso, ap)
    rho_area = spearman(area, ap)
    p_area, _ = permutation_p(area, ap)

    # The confound check: are anisotropy and size themselves correlated?
    rho_confound = spearman(aniso, area)

    if math.isnan(rho_aniso):
        verdict = "INCONCLUSIVE: correlation undefined (zero variance)."
    elif rho_aniso <= -0.5 and p_aniso < 0.05:
        verdict = (
            "SUPPORTS the anisotropy hypothesis. E4 (shape-aware detection) is "
            "motivated by the data. Check rho_ap_vs_area before claiming the "
            "mechanism is shape and not size."
        )
    elif rho_aniso <= -0.3:
        verdict = (
            "WEAK support. The direction is right but the effect is not significant "
            "at n=10 classes. E4 is defensible as exploratory but the paper cannot "
            "lead with this correlation as evidence."
        )
    else:
        verdict = (
            "DOES NOT SUPPORT the hypothesis. The elongation story is wrong or is "
            "not the dominant factor. Per RESEARCH_PROGRAM.md section 3 E1, the "
            "program should pivot to E2/E3/E5 as its main line rather than running "
            "E4. This is a real finding - report it, do not bury it."
        )

    return {
        "status": "ok",
        "n_classes_matched": len(paired),
        "n_permutations": n_perm,
        "rho_ap_vs_anisotropy": rho_aniso,
        "p_ap_vs_anisotropy": p_aniso,
        "rho_ap_vs_area": rho_area,
        "p_ap_vs_area": p_area,
        "rho_anisotropy_vs_area": rho_confound,
        "confound_warning": (
            "Anisotropy and box area are themselves strongly correlated "
            f"(rho={rho_confound:.3f}); the shape effect cannot be cleanly separated "
            "from the small-object effect on this data alone. E3 (resolution) "
            "partially disentangles them: if resolution alone closes the gap, size "
            "was the driver."
        ) if abs(rho_confound) > 0.6 else None,
        "predicted_elongated_confirmed": _check_prediction(rows),
        "verdict": verdict,
    }


def _check_prediction(rows: list[dict]) -> dict:
    """
    Did the classes predicted elongated in advance actually come out elongated?
    A pre-registered prediction that survives is worth more than a post-hoc grouping.
    """
    by_name = {r["class_name"]: r["median_abs_log2_ar"] for r in rows}
    elong = [by_name[n] for n in PREDICTED_ELONGATED if n in by_name]
    compact = [by_name[n] for n in PREDICTED_COMPACT if n in by_name]
    if not elong or not compact:
        return {"status": "insufficient_classes"}
    return {
        "status": "ok",
        "median_anisotropy_predicted_elongated": statistics.median(elong),
        "median_anisotropy_predicted_compact": statistics.median(compact),
        "prediction_holds": statistics.median(elong) > statistics.median(compact),
        "note": (
            "PREDICTED_ELONGATED / PREDICTED_COMPACT are declared at the top of this "
            "module before any data is read, so this is a genuine pre-registered "
            "prediction rather than a grouping fitted to the result."
        ),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(rows: list[dict], per_class_ap: Optional[dict], out: Path) -> Optional[Path]:
    """The motivating figure: per-class AP against per-class anisotropy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figure] matplotlib not installed - skipping plot", file=sys.stderr)
        return None

    if per_class_ap:
        paired = [(r, per_class_ap[r["class_name"]])
                  for r in rows if r["class_name"] in per_class_ap]
        if not paired:
            return None
        fig, ax = plt.subplots(figsize=(8, 5.5))
        xs = [r["median_abs_log2_ar"] for r, _ in paired]
        ys = [a for _, a in paired]
        sizes = [30 + 220 * (r["n_boxes"] / max(x["n_boxes"] for x, _ in paired))
                 for r, _ in paired]
        colors = ["#d1495b" if r["class_name"] in PREDICTED_ELONGATED else "#2e86ab"
                  for r, _ in paired]
        ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.75, edgecolors="k", linewidths=0.6)
        for (r, a), x in zip(paired, xs):
            ax.annotate(r["class_name"], (x, a), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        rho = spearman(xs, ys)
        ax.set_xlabel("Anisotropy — median |log2(pixel aspect ratio)|")
        ax.set_ylabel("AP@50 (test split)")
        ax.set_title(f"Per-class accuracy vs box elongation  (Spearman rho = {rho:.3f})")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.02, "red = predicted elongated   blue = predicted compact\n"
                            "marker size ∝ box count",
                transform=ax.transAxes, fontsize=7, va="bottom")
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        rows_sorted = sorted(rows, key=lambda r: r["median_abs_log2_ar"])
        names = [r["class_name"] for r in rows_sorted]
        vals = [r["median_abs_log2_ar"] for r in rows_sorted]
        colors = ["#d1495b" if n in PREDICTED_ELONGATED else "#2e86ab" for n in names]
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("Anisotropy — median |log2(pixel aspect ratio)|")
        ax.set_title("Per-class box elongation (no AP supplied)")
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "anisotropy_vs_ap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(rows: list[dict], test: Optional[dict], out: Path, sources: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    L: list[str] = [
        "# E1 — Anisotropy analysis",
        "",
        "Tests whether per-class detection accuracy falls as boxes get more elongated.",
        "Gates experiment E4 in `ml/research/RESEARCH_PROGRAM.md`.",
        "",
        "## Sources",
        "",
    ]
    for k, v in sources.items():
        L.append(f"- **{k}**: `{v}`")
    L += ["", "## Per-class geometry", "",
          "| Class | Boxes | Anisotropy | Median AR | AR p10–p90 | Median area | % boxes <1% area |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: -x["median_abs_log2_ar"]):
        L.append(
            f"| {r['class_name']} | {r['n_boxes']:,} | {r['median_abs_log2_ar']:.3f} | "
            f"{r['median_ar']:.2f} | {r['p10_ar']:.2f}–{r['p90_ar']:.2f} | "
            f"{r['median_area_frac']*100:.3f}% | {r['pct_boxes_under_1pct_area']:.1f}% |"
        )
    L += [
        "",
        "*Anisotropy = median |log2(pixel aspect ratio)|. 0 is square; 1.0 means the "
        "typical box is 2:1 or 1:2. Aspect ratios are in PIXELS, not normalised YOLO "
        "units — see `scan_boxes()` for why that distinction matters.*",
        "",
    ]

    if test and test.get("status") == "ok":
        L += [
            "## Hypothesis test",
            "",
            f"- Classes matched: **{test['n_classes_matched']}**",
            f"- Spearman rho (AP vs anisotropy): **{test['rho_ap_vs_anisotropy']:.3f}** "
            f"(permutation p = {test['p_ap_vs_anisotropy']:.4f}, "
            f"{test['n_permutations']:,} permutations)",
            f"- Spearman rho (AP vs box area): {test['rho_ap_vs_area']:.3f} "
            f"(p = {test['p_ap_vs_area']:.4f})",
            f"- Spearman rho (anisotropy vs area): {test['rho_anisotropy_vs_area']:.3f}",
            "",
        ]
        if test.get("confound_warning"):
            L += ["> **Confound.** " + test["confound_warning"], ""]
        pred = test.get("predicted_elongated_confirmed", {})
        if pred.get("status") == "ok":
            L += [
                "### Pre-registered prediction",
                "",
                f"- Predicted-elongated classes, median anisotropy: "
                f"**{pred['median_anisotropy_predicted_elongated']:.3f}**",
                f"- Predicted-compact classes, median anisotropy: "
                f"**{pred['median_anisotropy_predicted_compact']:.3f}**",
                f"- Prediction holds: **{pred['prediction_holds']}**",
                "",
            ]
        L += ["## Verdict", "", test["verdict"], ""]
    else:
        L += [
            "## Hypothesis test",
            "",
            "Not run — no per-class AP supplied. Run E0 first, then re-run this with "
            "`--per-class-ap <run>/per_class_ap.json`.",
            "",
        ]

    L += ["## Figure", "", "![anisotropy vs AP](anisotropy_vs_ap.png)", ""]

    path = out / "E1_anisotropy_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="E1 anisotropy analysis for RDDS")
    ap.add_argument("--labels", required=True, help="YOLO labels dir (*.txt)")
    ap.add_argument("--images", required=True, help="matching images dir")
    ap.add_argument("--per-class-ap", help="JSON of {class_name: AP50} from evaluation")
    ap.add_argument("--out", default="runs/research/E1_anisotropy")
    ap.add_argument("--names", help="comma-separated class names (default: N-RDD2024)")
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args()

    names = args.names.split(",") if args.names else CLASS_NAMES
    out = Path(args.out)

    by_class = scan_boxes(Path(args.labels), Path(args.images))
    rows = per_class_stats(by_class, names)

    per_class_ap: Optional[dict] = None
    test: Optional[dict] = None
    if args.per_class_ap:
        raw = json.loads(Path(args.per_class_ap).read_text(encoding="utf-8"))
        # Accept either a bare mapping or an eval JSON with a per_class_AP50 key.
        per_class_ap = raw.get("per_class_AP50", raw) if isinstance(raw, dict) else raw
        per_class_ap = {k: float(v) for k, v in per_class_ap.items()}
        test = test_hypothesis(rows, per_class_ap)

    fig = make_figure(rows, per_class_ap, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anisotropy.json").write_text(
        json.dumps({"per_class": rows, "hypothesis_test": test}, indent=2),
        encoding="utf-8",
    )
    report = write_report(rows, test, out, {
        "labels": args.labels,
        "images": args.images,
        "per_class_ap": args.per_class_ap or "(not supplied)",
    })

    print(f"\nwrote {report}")
    if fig:
        print(f"wrote {fig}")
    if test and test.get("status") == "ok":
        print(f"\nrho = {test['rho_ap_vs_anisotropy']:.3f}  "
              f"p = {test['p_ap_vs_anisotropy']:.4f}")
        print(f"\n{test['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
