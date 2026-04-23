"""
scripts/dataset_analysis.py

Comprehensive statistical analysis and visualisation of the merged
RT-DETR training dataset. Reads data/detection/train.json (COCO format)
and produces 12 publication-quality plots covering every aspect of the
dataset: class distribution, bounding box geometry, spatial density,
annotation load, class co-occurrence, and COCO size categories.

NO image files are read — all analysis is derived from annotation metadata.

Usage
-----
    python scripts/dataset_analysis.py
    python scripts/dataset_analysis.py --json  path/to/train.json
    python scripts/dataset_analysis.py --outdir path/to/output

Output
------
    01_class_instance_counts.png
    02_class_share_pie.png
    03_source_contribution_per_class.png
    04_annotations_per_image.png
    05_bbox_width_distribution.png
    06_bbox_height_distribution.png
    07_bbox_area_distribution.png
    08_bbox_aspect_ratio_violin.png
    09_spatial_density_heatmap.png
    10_coco_size_categories.png
    11_class_cooccurrence_heatmap.png
    12_compactness_vs_area_scatter.png

References
----------
COCO metrics (small/medium/large): Lin et al. 2014 (arxiv 1405.0312)
"""

import json
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — derived from project class schema, not hardcoded results
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]

CLASS_COLORS = {
    "longitudinal_crack":  "#3498DB",
    "transverse_crack":    "#2ECC71",
    "alligator_crack":     "#F39C12",
    "pothole":             "#E74C3C",
    "patch_deterioration": "#9B59B6",
}

SOURCE_COLORS = {
    "rdd2022":    "#2980B9",
    "pothole600": "#E67E22",
    "unknown":    "#95A5A6",
}

# COCO size boundaries (area in pixels²)
COCO_SMALL_MAX  = 32 ** 2    #  1,024 px²
COCO_MEDIUM_MAX = 96 ** 2    #  9,216 px²

FIGURE_DPI = 150
STYLE = "seaborn-v0_8-whitegrid"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coco(json_path: Path) -> dict:
    """Load and validate a COCO-format JSON file."""
    log.info(f"Loading {json_path} ...")
    with open(json_path, "r") as f:
        data = json.load(f)

    required = {"images", "annotations", "categories"}
    missing = required - set(data.keys())
    if missing:
        log.error(f"Missing keys in JSON: {missing}")
        sys.exit(1)

    log.info(
        f"Loaded: {len(data['images']):,} images, "
        f"{len(data['annotations']):,} annotations, "
        f"{len(data['categories'])} categories"
    )
    return data


def build_lookups(data: dict) -> tuple:
    """
    Returns
    -------
    id_to_name : dict  category_id -> class name
    img_source : dict  image_id    -> source string ("rdd2022" | "pothole600")
    img_size   : dict  image_id    -> (width, height)
    """
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    img_source = {}
    img_size   = {}
    for img in data["images"]:
        src = img.get("source", "unknown").lower()
        img_source[img["id"]] = src
        img_size[img["id"]]   = (img.get("width", 0), img.get("height", 0))

    return id_to_name, img_source, img_size


def extract_annotation_arrays(data: dict, id_to_name: dict) -> dict:
    """
    Parse all annotations into numpy arrays grouped by class.
    Skips degenerate boxes (w<=0 or h<=0).

    Returns dict with keys matching CLASS_NAMES, each containing:
        x, y, w, h, area, aspect_ratio, compactness, cx_norm, cy_norm
    """
    raw = defaultdict(lambda: defaultdict(list))

    skipped = 0
    for ann in data["annotations"]:
        cname = id_to_name.get(ann["category_id"], "unknown")
        if cname not in CLASS_NAMES:
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            skipped += 1
            continue

        area   = w * h
        aspect = w / h
        # compactness = 4π·area/perimeter² (circle=1, elongated≈0)
        perimeter   = 2 * (w + h)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Normalised centre (requires image size)
        img_id = ann["image_id"]
        raw[cname]["x"].append(x)
        raw[cname]["y"].append(y)
        raw[cname]["w"].append(w)
        raw[cname]["h"].append(h)
        raw[cname]["area"].append(area)
        raw[cname]["aspect"].append(aspect)
        raw[cname]["compactness"].append(compactness)
        raw[cname]["img_id"].append(img_id)

    if skipped:
        log.warning(f"Skipped {skipped} degenerate annotations (w<=0 or h<=0)")

    # Convert to numpy
    result = {}
    for cname in CLASS_NAMES:
        d = raw[cname]
        result[cname] = {k: np.array(v) for k, v in d.items()}

    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, outdir: Path, filename: str) -> None:
    path = outdir / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved -> {filename}")


def class_color_list(names):
    return [CLASS_COLORS.get(n, "#AAAAAA") for n in names]


# ---------------------------------------------------------------------------
# Plot 01 — Class instance counts (horizontal bar)
# ---------------------------------------------------------------------------

def plot_01_class_counts(arrays: dict, outdir: Path) -> None:
    log.info("Plot 01 — Class instance counts")

    names  = [n for n in CLASS_NAMES if len(arrays[n].get("w", [])) > 0 or True]
    counts = [len(arrays[n].get("w", [])) for n in names]
    colors = class_color_list(names)

    fig, ax = plt.subplots(figsize=(9, 5))
    with plt.style.context(STYLE):
        bars = ax.barh(names, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, labels=[f"{c:,}" for c in counts],
                     padding=6, fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of annotated instances", fontsize=11)
        ax.set_title("Dataset — Class Instance Counts", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        # Skewness annotation
        nonzero = [c for c in counts if c > 0]
        if len(nonzero) > 1:
            sk = stats.skew(nonzero)
            ax.text(0.98, 0.04,
                    f"Skewness: {sk:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlim(0, max(counts) * 1.18)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()

    save(fig, outdir, "01_class_instance_counts.png")


# ---------------------------------------------------------------------------
# Plot 02 — Class share pie
# ---------------------------------------------------------------------------

def plot_02_class_share_pie(arrays: dict, outdir: Path) -> None:
    log.info("Plot 02 — Class share pie")

    names  = CLASS_NAMES
    counts = [len(arrays[n].get("w", [])) for n in names]
    colors = class_color_list(names)

    # Separate zero-count classes
    nonzero_names  = [n for n, c in zip(names, counts) if c > 0]
    nonzero_counts = [c for c in counts if c > 0]
    nonzero_colors = [CLASS_COLORS.get(n, "#AAAAAA") for n in nonzero_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        nonzero_counts,
        labels=nonzero_names,
        colors=nonzero_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    total = sum(nonzero_counts)
    ax.set_title(
        f"Class Distribution  (total: {total:,} instances)",
        fontsize=13, fontweight="bold", pad=15
    )

    # Note zero-count classes
    zero_names = [n for n, c in zip(names, counts) if c == 0]
    if zero_names:
        note = "No training data: " + ", ".join(zero_names)
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, color="#888888")

    fig.tight_layout()
    save(fig, outdir, "02_class_share_pie.png")


# ---------------------------------------------------------------------------
# Plot 03 — Source contribution per class (grouped bar)
# ---------------------------------------------------------------------------

def plot_03_source_contribution(arrays: dict, img_source: dict,
                                 outdir: Path) -> None:
    log.info("Plot 03 — Source contribution per class")

    sources = sorted({s for s in img_source.values()})
    x       = np.arange(len(CLASS_NAMES))
    width   = 0.8 / max(len(sources), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    with plt.style.context(STYLE):
        for i, src in enumerate(sources):
            src_counts = []
            for cname in CLASS_NAMES:
                img_ids = arrays[cname].get("img_id", np.array([]))
                if len(img_ids) == 0:
                    src_counts.append(0)
                else:
                    n = sum(img_source.get(iid, "unknown") == src for iid in img_ids)
                    src_counts.append(n)

            offset = (i - len(sources) / 2 + 0.5) * width
            bars = ax.bar(x + offset, src_counts, width * 0.9,
                          label=src,
                          color=SOURCE_COLORS.get(src, "#AAAAAA"),
                          edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Instance count", fontsize=11)
        ax.set_title("Source Contribution per Class", fontsize=13, fontweight="bold")
        ax.legend(title="Dataset source", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        fig.tight_layout()

    save(fig, outdir, "03_source_contribution_per_class.png")


# ---------------------------------------------------------------------------
# Plot 04 — Annotations per image (histogram + statistics)
# ---------------------------------------------------------------------------

def plot_04_annotations_per_image(data: dict, id_to_name: dict,
                                   outdir: Path) -> None:
    log.info("Plot 04 — Annotations per image histogram")

    ann_per_img = defaultdict(int)
    for ann in data["annotations"]:
        cname = id_to_name.get(ann["category_id"], "unknown")
        if cname in CLASS_NAMES:
            ann_per_img[ann["image_id"]] += 1

    # Images with zero annotations
    all_img_ids = {img["id"] for img in data["images"]}
    for iid in all_img_ids:
        if iid not in ann_per_img:
            ann_per_img[iid] = 0

    counts = np.array(list(ann_per_img.values()))
    counts_nonzero = counts[counts > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: full distribution including zeros
    ax = axes[0]
    max_val = int(np.percentile(counts, 99)) + 1
    ax.hist(counts, bins=min(60, max_val), color="#3498DB",
            edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(np.mean(counts), color="#E74C3C", lw=1.8,
               linestyle="--", label=f"Mean: {np.mean(counts):.2f}")
    ax.axvline(np.median(counts), color="#F39C12", lw=1.8,
               linestyle=":", label=f"Median: {np.median(counts):.2f}")
    ax.set_xlabel("Annotations per image", fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("Annotations per Image (all images)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    sk = stats.skew(counts)
    ax.text(0.97, 0.96,
            f"Skewness: {sk:.2f}\nStd: {np.std(counts):.2f}\n"
            f"Max: {int(np.max(counts))}\n"
            f"Zero-ann images: {int(np.sum(counts == 0)):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    # Right: non-zero only, log-scale y
    ax2 = axes[1]
    ax2.hist(counts_nonzero, bins=min(60, int(np.max(counts_nonzero)) + 1),
             color="#2ECC71", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax2.set_yscale("log")
    ax2.axvline(np.mean(counts_nonzero), color="#E74C3C", lw=1.8,
                linestyle="--", label=f"Mean: {np.mean(counts_nonzero):.2f}")
    ax2.set_xlabel("Annotations per image (annotated only)", fontsize=11)
    ax2.set_ylabel("Count (log scale)", fontsize=11)
    ax2.set_title("Annotations per Image (annotated images only, log y)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Annotation Load Distribution", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, outdir, "04_annotations_per_image.png")


# ---------------------------------------------------------------------------
# Plot 05 — Bounding box width distribution per class
# ---------------------------------------------------------------------------

def plot_05_bbox_width(arrays: dict, outdir: Path) -> None:
    log.info("Plot 05 — Bounding box width distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    with plt.style.context(STYLE):
        for cname in CLASS_NAMES:
            w = arrays[cname].get("w", np.array([]))
            if len(w) == 0:
                continue
            ax.hist(w, bins=80, alpha=0.55, label=cname,
                    color=CLASS_COLORS[cname], edgecolor="none")
            ax.axvline(np.mean(w), color=CLASS_COLORS[cname],
                       linestyle="--", linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Bounding box width (pixels)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Bounding Box Width Distribution per Class",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, title="Class")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        fig.tight_layout()

    save(fig, outdir, "05_bbox_width_distribution.png")


# ---------------------------------------------------------------------------
# Plot 06 — Bounding box height distribution per class
# ---------------------------------------------------------------------------

def plot_06_bbox_height(arrays: dict, outdir: Path) -> None:
    log.info("Plot 06 — Bounding box height distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    with plt.style.context(STYLE):
        for cname in CLASS_NAMES:
            h = arrays[cname].get("h", np.array([]))
            if len(h) == 0:
                continue
            ax.hist(h, bins=80, alpha=0.55, label=cname,
                    color=CLASS_COLORS[cname], edgecolor="none")
            ax.axvline(np.mean(h), color=CLASS_COLORS[cname],
                       linestyle="--", linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Bounding box height (pixels)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Bounding Box Height Distribution per Class",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, title="Class")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        fig.tight_layout()

    save(fig, outdir, "06_bbox_height_distribution.png")


# ---------------------------------------------------------------------------
# Plot 07 — Bounding box area distribution (log scale)
# ---------------------------------------------------------------------------

def plot_07_bbox_area(arrays: dict, outdir: Path) -> None:
    log.info("Plot 07 — Bounding box area distribution (log scale)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overlapping histograms (log x)
    ax = axes[0]
    for cname in CLASS_NAMES:
        areas = arrays[cname].get("area", np.array([]))
        if len(areas) == 0:
            continue
        log_areas = np.log10(areas + 1)
        ax.hist(log_areas, bins=60, alpha=0.5, label=cname,
                color=CLASS_COLORS[cname], edgecolor="none")

    ax.set_xlabel("log₁₀(bounding box area)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Bbox Area Distribution (log scale)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: box plot per class
    ax2 = axes[1]
    data_for_box = []
    labels_for_box = []
    for cname in CLASS_NAMES:
        areas = arrays[cname].get("area", np.array([]))
        if len(areas) > 0:
            data_for_box.append(np.log10(areas + 1))
            labels_for_box.append(cname)

    if data_for_box:
        bp = ax2.boxplot(data_for_box, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
        for patch, cname in zip(bp["boxes"], labels_for_box):
            patch.set_facecolor(CLASS_COLORS.get(cname, "#AAAAAA"))
            patch.set_alpha(0.7)
        ax2.set_xticklabels(labels_for_box, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("log₁₀(area)", fontsize=11)
        ax2.set_title("Bbox Area Box Plot per Class", fontsize=12, fontweight="bold")

        # Add COCO size boundaries
        for boundary, label in [(np.log10(COCO_SMALL_MAX), "small/medium"),
                                  (np.log10(COCO_MEDIUM_MAX), "medium/large")]:
            ax2.axhline(boundary, color="red", linestyle=":", alpha=0.6, linewidth=1.2)
            ax2.text(len(data_for_box) + 0.1, boundary, label,
                     va="center", fontsize=7, color="red")

    fig.suptitle("Bounding Box Area Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, outdir, "07_bbox_area_distribution.png")


# ---------------------------------------------------------------------------
# Plot 08 — Bounding box aspect ratio (violin + summary table)
# ---------------------------------------------------------------------------

def plot_08_aspect_ratio(arrays: dict, outdir: Path) -> None:
    log.info("Plot 08 — Bounding box aspect ratio violin")

    valid = [(cname, arrays[cname]["aspect"])
             for cname in CLASS_NAMES
             if len(arrays[cname].get("aspect", [])) > 0]

    if not valid:
        log.warning("No aspect ratio data — skipping plot 08")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: violin plot
    ax = axes[0]
    names_v = [v[0] for v in valid]
    data_v  = [v[1] for v in valid]
    colors_v = [CLASS_COLORS.get(n, "#AAAAAA") for n in names_v]

    parts = ax.violinplot(data_v, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors_v):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(1, len(names_v) + 1))
    ax.set_xticklabels(names_v, rotation=20, ha="right", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="aspect = 1 (square)")
    ax.set_ylabel("width / height", fontsize=11)
    ax.set_title("Bounding Box Aspect Ratio (w/h) per Class",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(np.percentile(np.concatenate(data_v), 99) * 1.2, 15))

    # Right: mean ± std bar chart
    ax2 = axes[1]
    means = [np.mean(d) for d in data_v]
    stds  = [np.std(d)  for d in data_v]
    x     = np.arange(len(names_v))
    bars  = ax2.bar(x, means, color=colors_v, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
    ax2.errorbar(x, means, yerr=stds, fmt="none",
                 color="black", capsize=5, linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_v, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Mean aspect ratio ± std", fontsize=11)
    ax2.set_title("Mean Aspect Ratio per Class", fontsize=12, fontweight="bold")

    # Skewness per class
    for i, (mean_v, data_i, cname) in enumerate(zip(means, data_v, names_v)):
        sk = stats.skew(data_i)
        ax2.text(i, mean_v + stds[i] + 0.05, f"sk={sk:.1f}",
                 ha="center", fontsize=7, color="#333333")

    fig.suptitle("Aspect Ratio Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, outdir, "08_bbox_aspect_ratio_violin.png")


# ---------------------------------------------------------------------------
# Plot 09 — Spatial annotation density heatmap
# ---------------------------------------------------------------------------

def plot_09_spatial_density(data: dict, id_to_name: dict,
                             img_size: dict, outdir: Path) -> None:
    log.info("Plot 09 — Spatial annotation density heatmap")

    GRID = 32  # grid cells per side

    # Accumulate normalised centres per class
    centres = defaultdict(lambda: ([], []))  # cname -> (cx_norm_list, cy_norm_list)

    for ann in data["annotations"]:
        cname = id_to_name.get(ann["category_id"], "unknown")
        if cname not in CLASS_NAMES:
            continue
        x, y, w, h = ann["bbox"]
        img_id = ann["image_id"]
        iw, ih = img_size.get(img_id, (0, 0))
        if iw <= 0 or ih <= 0:
            continue
        cx_norm = (x + w / 2) / iw
        cy_norm = (y + h / 2) / ih
        centres[cname][0].append(cx_norm)
        centres[cname][1].append(cy_norm)

    active_classes = [c for c in CLASS_NAMES if len(centres[c][0]) > 0]
    n = len(active_classes)
    if n == 0:
        log.warning("No spatial data with image sizes — skipping plot 09")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, cname in zip(axes, active_classes):
        cx = np.array(centres[cname][0])
        cy = np.array(centres[cname][1])

        heatmap, _, _ = np.histogram2d(cx, cy,
                                        bins=GRID,
                                        range=[[0, 1], [0, 1]])
        heatmap = heatmap.T  # transpose so x=col, y=row

        im = ax.imshow(heatmap, origin="upper", aspect="equal",
                       cmap="YlOrRd", interpolation="bilinear",
                       extent=[0, 1, 1, 0])
        plt.colorbar(im, ax=ax, shrink=0.8, label="Count")
        ax.set_title(cname, fontsize=10, fontweight="bold",
                     color=CLASS_COLORS.get(cname, "black"))
        ax.set_xlabel("Normalised x", fontsize=8)
        ax.set_ylabel("Normalised y", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Spatial Density of Annotation Centres (normalised image coordinates)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, outdir, "09_spatial_density_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 10 — COCO size categories (small / medium / large) per class
# ---------------------------------------------------------------------------

def plot_10_coco_size_categories(arrays: dict, outdir: Path) -> None:
    log.info("Plot 10 — COCO size categories per class")

    size_labels = ["Small\n(<32²px)", "Medium\n(32²–96²px)", "Large\n(>96²px)"]
    size_colors = ["#3498DB", "#F39C12", "#E74C3C"]

    active = [c for c in CLASS_NAMES if len(arrays[c].get("area", [])) > 0]
    if not active:
        return

    small_counts  = []
    medium_counts = []
    large_counts  = []

    for cname in active:
        areas = arrays[cname]["area"]
        small_counts.append(int(np.sum(areas < COCO_SMALL_MAX)))
        medium_counts.append(int(np.sum(
            (areas >= COCO_SMALL_MAX) & (areas < COCO_MEDIUM_MAX))))
        large_counts.append(int(np.sum(areas >= COCO_MEDIUM_MAX)))

    x     = np.arange(len(active))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bar
    ax = axes[0]
    for i, (counts, label, color) in enumerate(zip(
            [small_counts, medium_counts, large_counts],
            size_labels, size_colors)):
        ax.bar(x + (i - 1) * width, counts, width, label=label,
               color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(active, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Instance count", fontsize=11)
    ax.set_title("COCO Size Categories per Class (grouped)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Right: 100% stacked bar
    ax2 = axes[1]
    totals = [s + m + l for s, m, l in zip(small_counts, medium_counts, large_counts)]
    s_pct = [s / t * 100 if t else 0 for s, t in zip(small_counts, totals)]
    m_pct = [m / t * 100 if t else 0 for m, t in zip(medium_counts, totals)]
    l_pct = [l / t * 100 if t else 0 for l, t in zip(large_counts, totals)]

    ax2.bar(x, s_pct, color=size_colors[0], alpha=0.8,
            label=size_labels[0], edgecolor="white", linewidth=0.5)
    ax2.bar(x, m_pct, bottom=s_pct, color=size_colors[1], alpha=0.8,
            label=size_labels[1], edgecolor="white", linewidth=0.5)
    ax2.bar(x, l_pct,
            bottom=[s + m for s, m in zip(s_pct, m_pct)],
            color=size_colors[2], alpha=0.8,
            label=size_labels[2], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(active, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("COCO Size Categories per Class (normalised)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "COCO Size Category Analysis  "
        f"(small: area<{COCO_SMALL_MAX:,}px², "
        f"medium: <{COCO_MEDIUM_MAX:,}px²)",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    save(fig, outdir, "10_coco_size_categories.png")


# ---------------------------------------------------------------------------
# Plot 11 — Class co-occurrence heatmap
# ---------------------------------------------------------------------------

def plot_11_cooccurrence(data: dict, id_to_name: dict, outdir: Path) -> None:
    log.info("Plot 11 — Class co-occurrence heatmap")

    # Build per-image class presence
    img_classes = defaultdict(set)
    for ann in data["annotations"]:
        cname = id_to_name.get(ann["category_id"], "unknown")
        if cname in CLASS_NAMES:
            img_classes[ann["image_id"]].add(cname)

    n = len(CLASS_NAMES)
    matrix = np.zeros((n, n), dtype=int)
    idx    = {c: i for i, c in enumerate(CLASS_NAMES)}

    for classes in img_classes.values():
        clist = list(classes)
        for i in range(len(clist)):
            for j in range(len(clist)):
                matrix[idx[clist[i]], idx[clist[j]]] += 1

    # Normalise by diagonal (conditional probability P(col | row))
    diag = matrix.diagonal().astype(float)
    diag[diag == 0] = 1  # avoid division by zero
    norm_matrix = matrix / diag[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw counts
    ax = axes[0]
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Co-occurrence count"})
    ax.set_title("Class Co-occurrence (raw counts)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Right: normalised
    ax2 = axes[1]
    sns.heatmap(norm_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax2, vmin=0, vmax=1,
                cbar_kws={"label": "P(column | row)"})
    ax2.set_title("Class Co-occurrence (normalised: P(col | row))",
                  fontsize=12, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle(
        "Class Co-occurrence per Image\n"
        "(diagonal = images containing each class; "
        "off-diagonal = images containing both classes)",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, outdir, "11_class_cooccurrence_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 12 — Compactness vs area scatter (coloured by class)
# ---------------------------------------------------------------------------

def plot_12_compactness_vs_area(arrays: dict, outdir: Path) -> None:
    log.info("Plot 12 — Compactness vs area scatter")

    MAX_POINTS = 2000  # cap per class for readability

    fig, ax = plt.subplots(figsize=(10, 6))
    with plt.style.context(STYLE):
        for cname in CLASS_NAMES:
            areas       = arrays[cname].get("area",        np.array([]))
            compactness = arrays[cname].get("compactness", np.array([]))
            if len(areas) == 0:
                continue

            # Subsample for readability if large
            n = len(areas)
            if n > MAX_POINTS:
                idx = np.random.choice(n, MAX_POINTS, replace=False)
                areas       = areas[idx]
                compactness = compactness[idx]

            ax.scatter(
                np.log10(areas + 1), compactness,
                c=CLASS_COLORS.get(cname, "#AAAAAA"),
                alpha=0.35, s=10, label=f"{cname} (n={n:,})",
                edgecolors="none"
            )

        ax.set_xlabel("log₁₀(bounding box area)", fontsize=11)
        ax.set_ylabel("Compactness  (4π·area / perimeter²)", fontsize=11)
        ax.set_title(
            "Bbox Compactness vs Area per Class\n"
            "(compactness=1.0: circle; ≈0.05: elongated crack)",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=8, markerscale=2, title="Class (subsampled)")

        # Reference lines
        ax.axhline(0.785, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(ax.get_xlim()[0] + 0.05, 0.79, "square ≈ 0.785",
                fontsize=7, color="gray")

        ax.set_ylim(0, 1.05)
        fig.tight_layout()

    save(fig, outdir, "12_compactness_vs_area_scatter.png")


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(arrays: dict, data: dict) -> None:
    log.info("=" * 60)
    log.info("DATASET SUMMARY")
    log.info("=" * 60)
    log.info(f"Total images      : {len(data['images']):,}")
    log.info(f"Total annotations : {len(data['annotations']):,}")
    log.info("")
    log.info(f"{'Class':<25} {'Count':>8}  {'Mean W':>8}  {'Mean H':>8}  "
             f"{'Mean Area':>10}  {'Skew(area)':>10}")
    log.info("-" * 75)
    for cname in CLASS_NAMES:
        d = arrays[cname]
        n = len(d.get("w", []))
        if n == 0:
            log.info(f"  {cname:<23} {'0':>8}  {'—':>8}  {'—':>8}  "
                     f"{'—':>10}  {'—':>10}")
            continue
        mw  = np.mean(d["w"])
        mh  = np.mean(d["h"])
        ma  = np.mean(d["area"])
        sk  = stats.skew(d["area"])
        log.info(f"  {cname:<23} {n:>8,}  {mw:>8.1f}  {mh:>8.1f}  "
                 f"{ma:>10.0f}  {sk:>10.2f}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset analysis — 12 plots from train.json"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("data/detection/train.json"),
        help="Path to COCO-format train.json (default: data/detection/train.json)"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/inspection_plots"),
        help="Output directory for plots (default: data/inspection_plots)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling in scatter plots"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"Input  : {args.json}")
    log.info(f"Output : {args.outdir}")

    # Load
    data                      = load_coco(args.json)
    id_to_name, img_source, img_size = build_lookups(data)
    arrays                    = extract_annotation_arrays(data, id_to_name)

    # Summary
    print_summary(arrays, data)

    # Plots
    log.info("Generating 12 plots ...")
    plot_01_class_counts(arrays, args.outdir)
    plot_02_class_share_pie(arrays, args.outdir)
    plot_03_source_contribution(arrays, img_source, args.outdir)
    plot_04_annotations_per_image(data, id_to_name, args.outdir)
    plot_05_bbox_width(arrays, args.outdir)
    plot_06_bbox_height(arrays, args.outdir)
    plot_07_bbox_area(arrays, args.outdir)
    plot_08_aspect_ratio(arrays, args.outdir)
    plot_09_spatial_density(data, id_to_name, img_size, args.outdir)
    plot_10_coco_size_categories(arrays, args.outdir)
    plot_11_cooccurrence(data, id_to_name, args.outdir)
    plot_12_compactness_vs_area(arrays, args.outdir)

    log.info("=" * 60)
    log.info(f"All 12 plots saved to: {args.outdir.resolve()}")
    log.info("Files generated:")
    for f in sorted(args.outdir.glob("*.png")):
        log.info(f"  {f.name}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()