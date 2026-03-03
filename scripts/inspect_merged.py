"""
scripts/inspect_merged.py

PURPOSE:
    Comprehensive visualization of the merged RT-DETR training dataset.
    Run after merge_datasets.py has produced data/detection/train.json.

PLOTS GENERATED:
    00 - Summary card
    01 - Dataset contribution (images + annotations per source)
    02 - Class distribution (full merged dataset)
    03 - Source vs class (grouped + stacked bar)
    04 - Sample images grid with bounding boxes
    05 - Bbox size 2D density heatmap
    06 - Bbox area distribution per class
    07 - Bbox aspect ratio per class (violin + bar)
    08 - Spatial annotation density heatmap per class
    09 - Annotations per image histogram
    10 - Image size distribution per source
    11 - Class co-occurrence heatmap
    12 - COCO size categories (small/medium/large) per class
    13 - Image quality per source (brightness/contrast/saturation)
    14 - Source comparison grid (one sample per class per source)

USAGE:
    python scripts/inspect_merged.py
    python scripts/inspect_merged.py --split train
    python scripts/inspect_merged.py --split val --max_imgs 200
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "detection"
OUTPUT_DIR = ROOT / "data" / "inspection_plots" / "merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG     = "#0D0F1A"
BG2    = "#12152A"
PANEL  = "#1A1D2E"
BORDER = "#2E3150"
WHITE  = "#E8EAF6"
GREY   = "#8892B0"

CLASS_COLORS = {
    "longitudinal_crack":  "#E63946",
    "transverse_crack":    "#F4A261",
    "alligator_crack":     "#2A9D8F",
    "pothole":             "#457B9D",
    "patch_deterioration": "#9B59B6",
    "unknown":             "#7F8C8D",
}

SOURCE_COLORS = {
    "rdd2022":    "#4FC3F7",
    "pothole600": "#FF8A65",
    "unknown":    "#90A4AE",
}

CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
    4: "patch_deterioration",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": WHITE,
    "xtick.color": GREY, "ytick.color": GREY, "text.color": WHITE,
    "grid.color": BORDER, "grid.linestyle": "--", "grid.alpha": 0.4,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": WHITE,
})


def save(fig, name, dpi=150):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=GREY)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    return ax


def load_merged(split):
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}\nRun merge_datasets.py first.")
    with open(path) as f:
        return json.load(f)


def build_records(coco):
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)
    records = []
    for img in coco["images"]:
        records.append({
            "id": img["id"], "file_name": img["file_name"],
            "width": img.get("width", 0), "height": img.get("height", 0),
            "source": img.get("source", "unknown"),
            "annotations": ann_by_img[img["id"]],
            "n_boxes": len(ann_by_img[img["id"]]),
        })
    return records


def plot_summary_card(records, coco, split):
    n_imgs  = len(records)
    n_anns  = len(coco["annotations"])
    sources = Counter(r["source"] for r in records)
    n_empty = sum(1 for r in records if r["n_boxes"] == 0)
    classes = Counter(CLASS_NAMES.get(a["category_id"], "unknown") for a in coco["annotations"])

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(0.5, 0.93, "MERGED ROAD DAMAGE DATASET", ha="center", fontsize=26,
            fontweight="bold", color=WHITE, transform=ax.transAxes)
    ax.text(0.5, 0.86, f"{split.upper()} SPLIT — Summary Card", ha="center",
            fontsize=14, color=GREY, transform=ax.transAxes)

    divider = mpatches.FancyBboxPatch((0.03, 0.825), 0.94, 0.003,
        boxstyle="round,pad=0", facecolor=BORDER, transform=ax.transAxes, clip_on=False)
    ax.add_patch(divider)

    stats = [
        ("Total Images", f"{n_imgs:,}"),
        ("Total Annotations", f"{n_anns:,}"),
        ("Empty Images", f"{n_empty:,}  ({n_empty/max(n_imgs,1)*100:.1f}%)"),
        ("Avg Boxes/Image", f"{n_anns/max(n_imgs,1):.2f}"),
        ("Damage Classes", str(len(classes))),
        ("Sources", str(len(sources))),
    ]
    for (label, value), x in zip(stats, [0.08, 0.25, 0.42, 0.59, 0.74, 0.88]):
        box = mpatches.FancyBboxPatch((x-0.06, 0.64), 0.13, 0.14,
            boxstyle="round,pad=0.01", facecolor=PANEL, edgecolor=BORDER,
            linewidth=1.5, transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(x, 0.735, value, ha="center", fontsize=14, fontweight="bold",
                color=WHITE, transform=ax.transAxes)
        ax.text(x, 0.665, label, ha="center", fontsize=9, color=GREY, transform=ax.transAxes)

    ax.text(0.06, 0.60, "SOURCE BREAKDOWN", fontsize=10, color=GREY, transform=ax.transAxes)
    x_s = 0.06
    for src, count in sorted(sources.items()):
        color = SOURCE_COLORS.get(src, "#7F8C8D")
        ax.text(x_s, 0.55, f"● {src}", fontsize=12, color=color,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x_s, 0.50, f"{count:,} images", fontsize=11, color=WHITE, transform=ax.transAxes)
        x_s += 0.22

    divider2 = mpatches.FancyBboxPatch((0.03, 0.455), 0.94, 0.002,
        boxstyle="round,pad=0", facecolor=BORDER, transform=ax.transAxes, clip_on=False)
    ax.add_patch(divider2)

    ax.text(0.06, 0.42, "CLASS BREAKDOWN", fontsize=10, color=GREY, transform=ax.transAxes)
    total_anns = sum(classes.values())
    x_bar = 0.04
    for cls_name, count in sorted(classes.items(), key=lambda x: -x[1]):
        w     = (count / max(total_anns, 1)) * 0.92
        color = CLASS_COLORS.get(cls_name, "#7F8C8D")
        rect  = mpatches.FancyBboxPatch((x_bar, 0.31), w, 0.06,
            boxstyle="round,pad=0.003", facecolor=color, edgecolor=BG,
            linewidth=1, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        if w > 0.04:
            ax.text(x_bar + w/2, 0.34, cls_name.replace("_", "\n"),
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x_bar + w/2, 0.285, f"{count:,}", ha="center", fontsize=8,
                    color=color, fontweight="bold", transform=ax.transAxes)
        x_bar += w

    ax.text(0.5, 0.10,
            "RT-DETR-L Training Dataset  ·  Cluj Urban Infrastructure Monitoring System",
            ha="center", fontsize=10, color=GREY, style="italic", transform=ax.transAxes)
    save(fig, f"merged_{split}_00_summary_card.png", dpi=150)


def plot_dataset_contribution(records, coco, split):
    src_imgs = Counter(r["source"] for r in records)
    src_anns = defaultdict(int)
    img_source = {r["id"]: r["source"] for r in records}
    for ann in coco["annotations"]:
        src_anns[img_source.get(ann["image_id"], "unknown")] += 1

    sources = sorted(src_imgs.keys())
    colors  = [SOURCE_COLORS.get(s, "#7F8C8D") for s in sources]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Merged Dataset — Source Contribution ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)

    for ax, data, ylabel, title in zip(
        axes[:2],
        [src_imgs, src_anns],
        ["Image count", "Annotation count"],
        ["Images per Source", "Annotations per Source"]
    ):
        style_ax(ax)
        vals = [data[s] for s in sources]
        bars = ax.bar(sources, vals, color=colors, edgecolor=BG, linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        for bar, s in zip(bars, sources):
            pct = data[s] / max(sum(data.values()), 1) * 100
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{data[s]:,}\n({pct:.1f}%)",
                    ha="center", color=WHITE, fontsize=10, fontweight="bold")

    ax = style_ax(axes[2])
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        [src_imgs[s] for s in sources], labels=sources, colors=colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": BG, "linewidth": 2, "width": 0.5})
    for t in texts + autotexts:
        t.set_color(WHITE)
    ax.set_title("Image Share (donut)")
    fig.tight_layout()
    save(fig, f"merged_{split}_01_dataset_contribution.png")


def plot_class_distribution(records, coco, split):
    classes   = Counter(CLASS_NAMES.get(a["category_id"], "unknown") for a in coco["annotations"])
    cls_names = sorted(classes.keys(), key=lambda x: -classes[x])
    counts    = [classes[c] for c in cls_names]
    colors    = [CLASS_COLORS.get(c, "#7F8C8D") for c in cls_names]
    total     = sum(counts)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Merged Dataset — Class Distribution ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)

    ax = style_ax(axes[0])
    bars = ax.barh(cls_names, counts, color=colors, edgecolor=BG, linewidth=0.8)
    ax.set_xlabel("Annotation count")
    ax.set_title("Absolute counts")
    ax.grid(axis="x")
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + total*0.004, bar.get_y() + bar.get_height()/2,
                f"{count:,}  ({count/total*100:.1f}%)", va="center", color=WHITE, fontsize=10)

    ax2 = style_ax(axes[1])
    ax2.set_facecolor(BG)
    wedges, texts, autotexts = ax2.pie(counts, labels=cls_names, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor": BG, "linewidth": 1.5})
    for t in texts + autotexts:
        t.set_color(WHITE)
        t.set_fontsize(9)
    ax2.set_title("Proportional distribution")
    fig.tight_layout()
    save(fig, f"merged_{split}_02_class_distribution.png")
def plot_source_vs_class(records, coco, split):
    img_source = {r["id"]: r["source"] for r in records}
    src_class  = defaultdict(Counter)
    for ann in coco["annotations"]:
        src_class[img_source.get(ann["image_id"], "unknown")][CLASS_NAMES.get(ann["category_id"], "unknown")] += 1

    sources     = sorted(src_class.keys())
    all_classes = sorted({c for s in src_class.values() for c in s})
    x = np.arange(len(all_classes))
    bar_w = 0.8 / len(sources)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f"Merged Dataset — Source vs Class ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)

    ax = style_ax(axes[0])
    for i, src in enumerate(sources):
        vals = [src_class[src].get(c, 0) for c in all_classes]
        ax.bar(x + (i - len(sources)/2 + 0.5)*bar_w, vals, bar_w,
               label=src, color=SOURCE_COLORS.get(src, "#7F8C8D"), edgecolor=BG, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in all_classes], fontsize=9)
    ax.set_ylabel("Annotation count")
    ax.set_title("Grouped: each source side by side per class")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
    ax.grid(axis="y")

    ax2 = style_ax(axes[1])
    bottoms = np.zeros(len(all_classes))
    for src in sources:
        vals = np.array([src_class[src].get(c, 0) for c in all_classes])
        ax2.bar(range(len(all_classes)), vals, bottom=bottoms,
                label=src, color=SOURCE_COLORS.get(src, "#7F8C8D"), edgecolor=BG, alpha=0.9)
        bottoms += vals
    ax2.set_xticks(range(len(all_classes)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in all_classes], fontsize=9)
    ax2.set_ylabel("Annotation count")
    ax2.set_title("Stacked: total per class coloured by source")
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
    ax2.grid(axis="y")
    fig.tight_layout()
    save(fig, f"merged_{split}_03_source_vs_class.png")


def plot_sample_images(records, split, n=16):
    annotated = [r for r in records if r["n_boxes"] > 0]
    if not annotated:
        return
    sample  = random.sample(annotated, min(n, len(annotated)))
    n_cols  = 4
    n_rows  = (len(sample) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Merged Dataset — Sample Annotated Images ({split} split)",
                 color=WHITE, fontsize=16, fontweight="bold")
    axes_flat = axes.flatten() if n_rows > 1 else list(axes)

    for i, rec in enumerate(sample):
        ax = axes_flat[i]
        ax.set_facecolor(BG2)
        img_path = Path(rec["file_name"])
        if not img_path.exists():
            ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                    color=GREY, transform=ax.transAxes)
            ax.axis("off")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        ax.imshow(img)
        for ann in rec["annotations"]:
            cls_name = CLASS_NAMES.get(ann["category_id"], "unknown")
            color    = CLASS_COLORS.get(cls_name, "white")
            x, y, w, h = ann["bbox"]
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y-3, cls_name.replace("_"," "), color=color, fontsize=6,
                    fontweight="bold", bbox=dict(facecolor=BG, alpha=0.6, pad=1, edgecolor="none"))
        src_color = SOURCE_COLORS.get(rec["source"], WHITE)
        ax.set_title(f"[{rec['source']}]  {img_path.name}", color=src_color, fontsize=7)
        ax.axis("off")

    for j in range(len(sample), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    save(fig, f"merged_{split}_04_sample_images.png", dpi=100)


def plot_bbox_heatmap(records, coco, split):
    img_dims = {r["id"]: (r["width"], r["height"]) for r in records}
    norm_w, norm_h = [], []
    for ann in coco["annotations"]:
        dims = img_dims.get(ann["image_id"])
        if not dims or dims[0] == 0 or dims[1] == 0:
            continue
        bw = ann["bbox"][2] / dims[0]
        bh = ann["bbox"][3] / dims[1]
        if 0 < bw <= 1 and 0 < bh <= 1:
            norm_w.append(bw)
            norm_h.append(bh)
    if not norm_w:
        return
    cmap = LinearSegmentedColormap.from_list("heat", [BG, "#E63946", "#FFD700"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Merged Dataset — Bounding Box Size Density ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    ax = style_ax(axes[0])
    h, xe, ye, img2 = ax.hist2d(norm_w, norm_h, bins=60, range=[[0,1],[0,1]], cmap=cmap)
    plt.colorbar(img2, ax=ax, label="Annotation count")
    ax.set_xlabel("Normalized box width")
    ax.set_ylabel("Normalized box height")
    ax.set_title("Box width vs height density")
    ax.axvline(np.mean(norm_w), color="white", linestyle="--", linewidth=1.2,
               label=f"mean w={np.mean(norm_w):.3f}")
    ax.axhline(np.mean(norm_h), color="#4FC3F7", linestyle="--", linewidth=1.2,
               label=f"mean h={np.mean(norm_h):.3f}")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
    ax2 = style_ax(axes[1])
    sample_idx = random.sample(range(len(norm_w)), min(5000, len(norm_w)))
    ax2.scatter([norm_w[i] for i in sample_idx], [norm_h[i] for i in sample_idx],
                s=3, alpha=0.3, c="#E63946", linewidths=0)
    ax2.set_xlabel("Normalized box width")
    ax2.set_ylabel("Normalized box height")
    ax2.set_title(f"Box size scatter (n={len(sample_idx):,} sampled)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, f"merged_{split}_05_bbox_size_heatmap.png")


def plot_bbox_area_per_class(records, coco, split):
    img_dims  = {r["id"]: (r["width"], r["height"]) for r in records}
    cls_areas = defaultdict(list)
    for ann in coco["annotations"]:
        dims = img_dims.get(ann["image_id"])
        if not dims or dims[0] == 0 or dims[1] == 0:
            continue
        rel = (ann["bbox"][2]*ann["bbox"][3]) / (dims[0]*dims[1]) * 100
        cls_areas[CLASS_NAMES.get(ann["category_id"], "unknown")].append(rel)
    if not cls_areas:
        return
    classes = sorted(cls_areas.keys())
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Merged Dataset — Bbox Area per Class ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    ax = style_ax(axes[0])
    for cls in classes:
        ax.hist(cls_areas[cls], bins=60, range=(0, 20),
                color=CLASS_COLORS.get(cls, "#7F8C8D"), alpha=0.55, label=cls, edgecolor="none")
    ax.set_xlabel("Box area (% of image)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution (0–20% range)")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)
    ax.grid(axis="y")
    ax2 = style_ax(axes[1])
    bp = ax2.boxplot([cls_areas[c] for c in classes], patch_artist=True,
                     labels=[c.replace("_","\n") for c in classes],
                     medianprops={"color":"white","linewidth":2},
                     flierprops={"marker":".","markersize":2,"alpha":0.3,"markerfacecolor":GREY})
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS.get(cls, "#7F8C8D"))
        patch.set_alpha(0.75)
    for elem in ["whiskers","caps"]:
        for item in bp[elem]:
            item.set_color(GREY)
    ax2.set_ylabel("Box area (% of image)")
    ax2.set_title("Boxplot per class")
    ax2.grid(axis="y")
    fig.tight_layout()
    save(fig, f"merged_{split}_06_bbox_area_per_class.png")


def plot_bbox_aspect_ratio(records, coco, split):
    cls_ratios = defaultdict(list)
    for ann in coco["annotations"]:
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        if bh > 0:
            cls_ratios[CLASS_NAMES.get(ann["category_id"], "unknown")].append(bw / bh)
    if not cls_ratios:
        return
    classes = sorted(cls_ratios.keys())
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Merged Dataset — Bbox Aspect Ratio W/H ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    ax = style_ax(axes[0])
    parts = ax.violinplot([cls_ratios[c] for c in classes], positions=range(len(classes)),
                          showmedians=True, showextrema=True)
    for pc, cls in zip(parts["bodies"], classes):
        pc.set_facecolor(CLASS_COLORS.get(cls, "#7F8C8D"))
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    for elem in ["cmaxes","cmins","cbars"]:
        parts[elem].set_color(GREY)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_","\n") for c in classes], fontsize=9)
    ax.set_ylabel("Width / Height ratio")
    ax.set_title("Violin plot")
    ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1, alpha=0.7, label="Square (1.0)")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
    ax.grid(axis="y")
    ax2 = style_ax(axes[1])
    means  = [np.mean(cls_ratios[c]) for c in classes]
    stds   = [np.std(cls_ratios[c])  for c in classes]
    colors = [CLASS_COLORS.get(c, "#7F8C8D") for c in classes]
    bars   = ax2.bar(range(len(classes)), means, yerr=stds, color=colors, edgecolor=BG,
                     linewidth=0.8, capsize=5, error_kw={"ecolor":WHITE,"linewidth":1.5})
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace("_","\n") for c in classes], fontsize=9)
    ax2.set_ylabel("Mean W/H ratio (±std)")
    ax2.set_title("Mean aspect ratio per class")
    ax2.axhline(1.0, color=GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax2.grid(axis="y")
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                 f"{mean:.2f}", ha="center", color=WHITE, fontsize=10)
    fig.tight_layout()
    save(fig, f"merged_{split}_07_bbox_aspect_ratio.png")


def plot_spatial_density(records, coco, split):
    img_dims  = {r["id"]: (r["width"], r["height"]) for r in records}
    cls_grids = defaultdict(lambda: np.zeros((30, 30)))
    all_grid  = np.zeros((30, 30))
    for ann in coco["annotations"]:
        dims = img_dims.get(ann["image_id"])
        if not dims or dims[0] == 0 or dims[1] == 0:
            continue
        cx_n = (ann["bbox"][0] + ann["bbox"][2]/2) / dims[0]
        cy_n = (ann["bbox"][1] + ann["bbox"][3]/2) / dims[1]
        gx = min(int(cx_n*30), 29)
        gy = min(int(cy_n*30), 29)
        cls_name = CLASS_NAMES.get(ann["category_id"], "unknown")
        cls_grids[cls_name][gy, gx] += 1
        all_grid[gy, gx] += 1
    classes = sorted(cls_grids.keys())
    cmap = LinearSegmentedColormap.from_list("spatial", [BG, "#2A9D8F", "#E63946", "#FFD700"])
    n_plots = len(classes) + 1
    n_cols  = 3
    n_rows  = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    fig.suptitle(f"Merged Dataset — Spatial Annotation Density ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    axes_flat = axes.flatten() if n_rows > 1 else list(axes)
    ax = style_ax(axes_flat[0])
    im = ax.imshow(all_grid, cmap=cmap, aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="count")
    ax.set_title("ALL CLASSES COMBINED", color=WHITE)
    for i, cls in enumerate(classes):
        ax = style_ax(axes_flat[i+1])
        im = ax.imshow(cls_grids[cls], cmap=cmap, aspect="equal", origin="upper")
        plt.colorbar(im, ax=ax, label="count")
        ax.set_title(cls.replace("_"," ").title(), color=CLASS_COLORS.get(cls, WHITE))
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    save(fig, f"merged_{split}_08_spatial_density.png")


def plot_annotations_per_image(records, split):
    src_nboxes = defaultdict(list)
    for r in records:
        src_nboxes[r["source"]].append(r["n_boxes"])
    src_nboxes["ALL"] = [r["n_boxes"] for r in records]
    sources = ["ALL"] + [s for s in src_nboxes if s != "ALL"]
    fig, axes = plt.subplots(1, len(sources), figsize=(7*len(sources), 6))
    fig.suptitle(f"Merged Dataset — Annotations per Image ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    if len(sources) == 1:
        axes = [axes]
    for ax, src in zip(axes, sources):
        style_ax(ax)
        data  = src_nboxes[src]
        color = SOURCE_COLORS.get(src, "#4FC3F7")
        ax.hist(data, bins=range(0, min(max(data)+2, 25)), color=color, edgecolor=BG, alpha=0.85)
        ax.axvline(np.mean(data), color="white", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(data):.2f}")
        ax.axvline(np.median(data), color="#FFD700", linestyle=":", linewidth=1.5,
                   label=f"median={np.median(data):.0f}")
        ax.set_xlabel("Annotations per image")
        ax.set_ylabel("Image count")
        ax.set_title(src.upper())
        ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
        ax.grid(axis="y")
        ax.text(0.97, 0.95, f"Empty: {data.count(0)/max(len(data),1)*100:.1f}%",
                ha="right", va="top", color=GREY, fontsize=9, transform=ax.transAxes)
    fig.tight_layout()
    save(fig, f"merged_{split}_09_annotations_per_image.png")


def plot_image_sizes(records, split):
    src_dims = defaultdict(lambda: {"w":[], "h":[], "ar":[]})
    for r in records:
        if r["width"] > 0 and r["height"] > 0:
            src_dims[r["source"]]["w"].append(r["width"])
            src_dims[r["source"]]["h"].append(r["height"])
            src_dims[r["source"]]["ar"].append(r["width"]/r["height"])
    src_dims["ALL"] = {
        "w":  [r["width"]  for r in records if r["width"]  > 0],
        "h":  [r["height"] for r in records if r["height"] > 0],
        "ar": [r["width"]/r["height"] for r in records if r["width"] > 0 and r["height"] > 0],
    }
    sources = ["ALL"] + [s for s in src_dims if s != "ALL"]
    fig, axes = plt.subplots(len(sources), 3, figsize=(18, 5*len(sources)))
    fig.suptitle(f"Merged Dataset — Image Size Distribution ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    for row, src in enumerate(sources):
        dims  = src_dims[src]
        color = SOURCE_COLORS.get(src, "#4FC3F7")
        for col, (data, label) in enumerate(zip(
            [dims["w"], dims["h"], dims["ar"]],
            ["Width (px)", "Height (px)", "Aspect Ratio W/H"]
        )):
            ax = style_ax(axes[row, col] if len(sources) > 1 else axes[col])
            ax.hist(data, bins=40, color=color, edgecolor=BG, alpha=0.85)
            ax.axvline(np.mean(data), color="white", linestyle="--", linewidth=1.5,
                       label=f"mean={np.mean(data):.1f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{src.upper()} — {label}")
            ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
            ax.grid(axis="y")
    fig.tight_layout()
    save(fig, f"merged_{split}_10_image_sizes.png")


def plot_class_cooccurrence(records, split):
    all_classes = sorted(CLASS_NAMES.values())
    n_cls = len(all_classes)
    cls_idx = {c: i for i, c in enumerate(all_classes)}
    comat   = np.zeros((n_cls, n_cls), dtype=int)
    for r in records:
        present = set(CLASS_NAMES.get(a["category_id"],"unknown") for a in r["annotations"])
        for c1 in present:
            for c2 in present:
                if c1 in cls_idx and c2 in cls_idx:
                    comat[cls_idx[c1], cls_idx[c2]] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"Merged Dataset — Class Co-occurrence ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    style_ax(ax)
    cmap = LinearSegmentedColormap.from_list("cooc", [BG, "#2A9D8F", "#E63946"])
    im   = ax.imshow(comat, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=ax, label="Co-occurrence count")
    short = [c.replace("_", "\n") for c in all_classes]
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(short, fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title("How often classes appear together in the same image", color=WHITE)
    for i in range(n_cls):
        for j in range(n_cls):
            val = comat[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=8,
                    color="white" if val < comat.max()*0.6 else BG)
    fig.tight_layout()
    save(fig, f"merged_{split}_11_class_cooccurrence.png")


def plot_coco_size_categories(records, coco, split):
    cls_sizes = defaultdict(lambda: [0, 0, 0])
    for ann in coco["annotations"]:
        area     = ann["bbox"][2] * ann["bbox"][3]
        cls_name = CLASS_NAMES.get(ann["category_id"], "unknown")
        if area < 1024:
            cls_sizes[cls_name][0] += 1
        elif area < 9216:
            cls_sizes[cls_name][1] += 1
        else:
            cls_sizes[cls_name][2] += 1
    classes     = sorted(cls_sizes.keys())
    size_labels = ["small (<32²)", "medium (32²–96²)", "large (≥96²)"]
    size_colors = ["#457B9D", "#2A9D8F", "#E63946"]
    x = np.arange(len(classes))
    bar_w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Merged Dataset — COCO Size Categories ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    ax = style_ax(axes[0])
    for i, (label, color) in enumerate(zip(size_labels, size_colors)):
        ax.bar(x + (i-1)*bar_w, [cls_sizes[c][i] for c in classes], bar_w,
               label=label, color=color, edgecolor=BG, linewidth=0.8, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_","\n") for c in classes], fontsize=9)
    ax.set_ylabel("Annotation count")
    ax.set_title("Size category per class (grouped)")
    ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
    ax.grid(axis="y")
    ax2 = style_ax(axes[1])
    totals  = [sum(cls_sizes[c]) for c in classes]
    bottoms = np.zeros(len(classes))
    for i, (label, color) in enumerate(zip(size_labels, size_colors)):
        vals = np.array([cls_sizes[c][i]/max(totals[j],1)*100 for j,c in enumerate(classes)])
        ax2.bar(range(len(classes)), vals, bottom=bottoms, label=label,
                color=color, edgecolor=BG, linewidth=0.8, alpha=0.9)
        bottoms += vals
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace("_","\n") for c in classes], fontsize=9)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Proportion per class (100% stacked)")
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
    ax2.grid(axis="y")
    fig.tight_layout()
    save(fig, f"merged_{split}_12_coco_size_categories.png")


def plot_image_quality(records, split, max_per_source=300):
    src_quality = defaultdict(lambda: {"brightness":[], "contrast":[], "saturation":[]})
    by_source   = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for src, src_records in by_source.items():
        sample = random.sample(src_records, min(max_per_source, len(src_records)))
        for rec in tqdm(sample, desc=f"  Quality [{src}]"):
            img_path = Path(rec["file_name"])
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            src_quality[src]["brightness"].append(float(np.mean(gray)))
            src_quality[src]["contrast"].append(float(np.std(gray)))
            src_quality[src]["saturation"].append(float(np.mean(hsv[:,:,1])))
    sources = sorted(src_quality.keys())
    if not sources:
        return
    metrics = ["brightness", "contrast", "saturation"]
    labels  = ["Mean Brightness (0–255)", "Contrast (std dev)", "Mean Saturation (0–255)"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5*len(metrics)))
    fig.suptitle(f"Merged Dataset — Image Quality per Source ({split} split)",
                 fontsize=16, fontweight="bold", color=WHITE)
    for ax, metric, label in zip(axes, metrics, labels):
        style_ax(ax)
        for src in sources:
            data  = src_quality[src][metric]
            color = SOURCE_COLORS.get(src, "#7F8C8D")
            ax.hist(data, bins=50, color=color, alpha=0.6,
                    label=f"{src} (mean={np.mean(data):.1f})", edgecolor="none")
            ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Image count")
        ax.set_title(label)
        ax.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
        ax.grid(axis="y")
    fig.tight_layout()
    save(fig, f"merged_{split}_13_image_quality_per_source.png")


def plot_source_comparison(records, split):
    src_cls_records = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["n_boxes"] == 0:
            continue
        for cls in {CLASS_NAMES.get(a["category_id"],"unknown") for a in r["annotations"]}:
            src_cls_records[r["source"]][cls].append(r)

    sources     = sorted(src_cls_records.keys())
    all_classes = sorted(CLASS_NAMES.values())
    n_cols = len(sources)
    n_rows = len(all_classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Merged Dataset — Source Comparison by Class ({split} split)",
                 color=WHITE, fontsize=16, fontweight="bold")

    for row, cls in enumerate(all_classes):
        for col, src in enumerate(sources):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_facecolor(BG2)
            candidates = src_cls_records[src][cls]
            if not candidates:
                ax.text(0.5, 0.5, f"No {cls}\nin {src}", ha="center", va="center",
                        color=GREY, transform=ax.transAxes, fontsize=9)
                ax.axis("off")
                continue
            rec      = random.choice(candidates)
            img_path = Path(rec["file_name"])
            if not img_path.exists():
                ax.axis("off")
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            ax.imshow(img)
            for ann in rec["annotations"]:
                ann_cls = CLASS_NAMES.get(ann["category_id"],"unknown")
                color   = CLASS_COLORS.get(ann_cls, "white")
                lw      = 3 if ann_cls == cls else 1
                alpha   = 1.0 if ann_cls == cls else 0.4
                x, y, w, h = ann["bbox"]
                ax.add_patch(plt.Rectangle((x,y), w, h, linewidth=lw,
                             edgecolor=color, facecolor="none", alpha=alpha))
            if row == 0:
                ax.set_title(src.upper(), color=SOURCE_COLORS.get(src, WHITE),
                             fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(cls.replace("_","\n"),
                              color=CLASS_COLORS.get(cls, WHITE), fontsize=9, fontweight="bold")
            ax.axis("off")

    fig.tight_layout()
    save(fig, f"merged_{split}_14_source_comparison.png", dpi=80)


def main(split, max_imgs):
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)
    print(f"Loading merged {split}.json ...")
    coco    = load_merged(split)
    records = build_records(coco)
    print(f"  {len(records):,} images  |  {len(coco['annotations']):,} annotations")
    print("\nGenerating plots...")

    plot_summary_card(records, coco, split)
    plot_dataset_contribution(records, coco, split)
    plot_class_distribution(records, coco, split)
    plot_source_vs_class(records, coco, split)
    plot_sample_images(records, split, n=16)
    plot_bbox_heatmap(records, coco, split)
    plot_bbox_area_per_class(records, coco, split)
    plot_bbox_aspect_ratio(records, coco, split)
    plot_spatial_density(records, coco, split)
    plot_annotations_per_image(records, split)
    plot_image_sizes(records, split)
    plot_class_cooccurrence(records, split)
    plot_coco_size_categories(records, coco, split)
    plot_image_quality(records, split, max_per_source=max_imgs)
    plot_source_comparison(records, split)

    print("\n" + "=" * 60)
    print(f"Done. All {15} plots saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob(f"merged_{split}_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="train", choices=["train","val","test"])
    parser.add_argument("--max_imgs", type=int, default=300,
                        help="Max images per source for quality analysis")
    args = parser.parse_args()
    main(args.split, args.max_imgs)