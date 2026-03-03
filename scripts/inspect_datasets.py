"""
scripts/inspect_datasets.py

Comprehensive dataset inspection and visualization for documentation.
Generates plots covering:
  - Class distribution
  - Image size/aspect ratio analysis
  - Annotation statistics
  - Sample image grids
  - Bounding box size distributions
  - Dataset overlap/comparison
  - Color/brightness analysis
  - Annotation density maps

Usage:
    python scripts/inspect_datasets.py
    python scripts/inspect_datasets.py --dataset rdd2022
    python scripts/inspect_datasets.py --dataset all
"""

import os
import sys
import json
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_DIR   = BASE_DIR / "data" / "inspection_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "pothole":            "#E74C3C",
    "longitudinal_crack": "#3498DB",
    "transverse_crack":   "#2ECC71",
    "alligator_crack":    "#F39C12",
    "patch_deterioration":"#9B59B6",
    "background":         "#95A5A6",
}

# RDD2022 label mapping → our classes
RDD_CLASS_MAP = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    "D43": "pothole",
    "D44": "pothole",
    "D50": "patch_deterioration",
}

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#B0B0B0",
    "ytick.color":      "#B0B0B0",
    "text.color":       "#E0E0E0",
    "grid.color":       "#2A2D3D",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})

COLORS = list(PALETTE.values())


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, name, dpi=150):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path.name}")
    return path


def load_rdd2022_annotations(dataset_dir: Path):
    """Parse VOC-XML annotations from RDD2022."""
    records = []
    ann_dirs = list(dataset_dir.rglob("annotations/xmls"))
    img_dirs = list(dataset_dir.rglob("images"))

    for ann_dir in tqdm(ann_dirs, desc="Parsing RDD2022 XMLs"):
        country = ann_dir.parts[-3] if len(ann_dir.parts) >= 3 else "unknown"
        split   = ann_dir.parts[-2] if len(ann_dir.parts) >= 2 else "unknown"

        for xml_file in ann_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                size_el = root.find("size")
                w = int(size_el.find("width").text)  if size_el else 0
                h = int(size_el.find("height").text) if size_el else 0

                boxes = []
                for obj in root.findall("object"):
                    name_el = obj.find("name")
                    bnd_el  = obj.find("bndbox")
                    if name_el is None or bnd_el is None:
                        continue
                    raw_cls = name_el.text.strip()
                    cls     = RDD_CLASS_MAP.get(raw_cls, raw_cls)
                    xmin = float(bnd_el.find("xmin").text)
                    ymin = float(bnd_el.find("ymin").text)
                    xmax = float(bnd_el.find("xmax").text)
                    ymax = float(bnd_el.find("ymax").text)
                    bw = xmax - xmin
                    bh = ymax - ymin
                    boxes.append({
                        "class": cls,
                        "raw_class": raw_cls,
                        "xmin": xmin, "ymin": ymin,
                        "xmax": xmax, "ymax": ymax,
                        "bw": bw, "bh": bh,
                        "area": bw * bh,
                        "cx": (xmin + xmax) / 2,
                        "cy": (ymin + ymax) / 2,
                    })

                records.append({
                    "file":    xml_file.stem,
                    "country": country,
                    "split":   split,
                    "width":   w,
                    "height":  h,
                    "aspect":  w / h if h > 0 else 0,
                    "boxes":   boxes,
                    "n_boxes": len(boxes),
                })
            except Exception:
                continue

    return records


def find_images(dataset_dir: Path, exts=(".jpg", ".jpeg", ".png")):
    imgs = []
    for ext in exts:
        imgs.extend(dataset_dir.rglob(f"*{ext}"))
        imgs.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    return imgs


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Class distribution bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(records, dataset_name):
    class_counts = Counter()
    for r in records:
        for b in r["boxes"]:
            class_counts[b["class"]] += 1

    if not class_counts:
        print("  No annotations found — skipping class distribution.")
        return

    classes = list(class_counts.keys())
    counts  = [class_counts[c] for c in classes]
    colors  = [PALETTE.get(c, "#7F8C8D") for c in classes]
    total   = sum(counts)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{dataset_name} — Class Distribution  (n={total:,} annotations)",
                 fontsize=16, fontweight="bold", y=1.02)

    # Bar chart
    ax = axes[0]
    bars = ax.barh(classes, counts, color=colors, edgecolor="#0F1117", linewidth=0.8)
    ax.set_xlabel("Number of annotations")
    ax.set_title("Absolute counts")
    ax.grid(axis="x")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=10)

    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=classes, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "#0F1117", "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Proportional distribution")

    fig.tight_layout()
    save(fig, f"{dataset_name}_01_class_distribution.png")
    return class_counts


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Image size & aspect ratio
# ══════════════════════════════════════════════════════════════════════════════

def plot_image_sizes(records, dataset_name):
    widths  = [r["width"]  for r in records if r["width"]  > 0]
    heights = [r["height"] for r in records if r["height"] > 0]
    aspects = [r["aspect"] for r in records if r["aspect"] > 0]

    if not widths:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset_name} — Image Dimensions", fontsize=16, fontweight="bold")

    axes[0].hist(widths,  bins=40, color=COLORS[1], edgecolor="#0F1117", alpha=0.85)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_title(f"Width distribution  (mean={np.mean(widths):.0f}px)")
    axes[0].grid(axis="y")

    axes[1].hist(heights, bins=40, color=COLORS[2], edgecolor="#0F1117", alpha=0.85)
    axes[1].set_xlabel("Height (px)")
    axes[1].set_title(f"Height distribution  (mean={np.mean(heights):.0f}px)")
    axes[1].grid(axis="y")

    axes[2].hist(aspects, bins=40, color=COLORS[3], edgecolor="#0F1117", alpha=0.85)
    axes[2].set_xlabel("Aspect ratio (W/H)")
    axes[2].set_title(f"Aspect ratio  (mean={np.mean(aspects):.2f})")
    axes[2].grid(axis="y")

    fig.tight_layout()
    save(fig, f"{dataset_name}_02_image_sizes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Bounding box size distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_bbox_sizes(records, dataset_name):
    by_class = defaultdict(list)
    for r in records:
        img_area = r["width"] * r["height"] if r["width"] > 0 else 1
        for b in r["boxes"]:
            rel_area = b["area"] / img_area if img_area > 0 else 0
            by_class[b["class"]].append({
                "bw": b["bw"], "bh": b["bh"],
                "area": b["area"], "rel_area": rel_area,
                "cx": b["cx"] / r["width"]  if r["width"]  > 0 else 0.5,
                "cy": b["cy"] / r["height"] if r["height"] > 0 else 0.5,
            })

    if not by_class:
        return

    classes = list(by_class.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{dataset_name} — Bounding Box Analysis", fontsize=16, fontweight="bold")

    # Box width vs height scatter
    ax = axes[0, 0]
    for cls in classes:
        bws = [b["bw"] for b in by_class[cls]]
        bhs = [b["bh"] for b in by_class[cls]]
        ax.scatter(bws, bhs, alpha=0.2, s=8,
                   color=PALETTE.get(cls, "#7F8C8D"), label=cls)
    ax.set_xlabel("Box width (px)")
    ax.set_ylabel("Box height (px)")
    ax.set_title("Width vs Height")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid()

    # Relative area distribution per class
    ax = axes[0, 1]
    data   = [[b["rel_area"] * 100 for b in by_class[c]] for c in classes]
    colors = [PALETTE.get(c, "#7F8C8D") for c in classes]
    bp = ax.boxplot(data, patch_artist=True, labels=classes,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("% of image area")
    ax.set_title("Box area relative to image")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")

    # Center point heatmap
    ax = axes[1, 0]
    all_cx = [b["cx"] for cls in classes for b in by_class[cls]]
    all_cy = [b["cy"] for cls in classes for b in by_class[cls]]
    h, xedges, yedges = np.histogram2d(all_cx, all_cy, bins=30,
                                        range=[[0, 1], [0, 1]])
    cmap = LinearSegmentedColormap.from_list("heat", ["#1A1D27", "#E74C3C"])
    im = ax.imshow(h.T, origin="lower", aspect="equal",
                   extent=[0, 1, 0, 1], cmap=cmap)
    plt.colorbar(im, ax=ax, label="count")
    ax.set_xlabel("Normalized X center")
    ax.set_ylabel("Normalized Y center")
    ax.set_title("Annotation center point density")

    # Annotations per image histogram
    ax = axes[1, 1]
    n_boxes = [r["n_boxes"] for r in records]
    ax.hist(n_boxes, bins=range(0, min(max(n_boxes) + 2, 30)),
            color=COLORS[4], edgecolor="#0F1117", alpha=0.85)
    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Annotations/image  (mean={np.mean(n_boxes):.1f}, "
                 f"median={np.median(n_boxes):.0f})")
    ax.grid(axis="y")

    fig.tight_layout()
    save(fig, f"{dataset_name}_03_bbox_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Country / split breakdown (RDD2022 specific)
# ══════════════════════════════════════════════════════════════════════════════

def plot_country_breakdown(records, dataset_name):
    country_class = defaultdict(Counter)
    for r in records:
        for b in r["boxes"]:
            country_class[r["country"]][b["class"]] += 1

    if len(country_class) < 2:
        return

    countries = sorted(country_class.keys())
    all_classes = sorted({c for cc in country_class.values() for c in cc})

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"{dataset_name} — Country Breakdown", fontsize=16, fontweight="bold")

    # Stacked bar per country
    ax = axes[0]
    bottoms = np.zeros(len(countries))
    for i, cls in enumerate(all_classes):
        vals = [country_class[c].get(cls, 0) for c in countries]
        bars = ax.bar(countries, vals, bottom=bottoms,
                      color=PALETTE.get(cls, COLORS[i % len(COLORS)]),
                      label=cls, edgecolor="#0F1117", linewidth=0.5)
        bottoms += np.array(vals)
    ax.set_ylabel("Number of annotations")
    ax.set_title("Annotations per country (stacked by class)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y")

    # Images per country
    ax2 = axes[1]
    img_counts = Counter(r["country"] for r in records)
    cts = sorted(img_counts.keys())
    vals = [img_counts[c] for c in cts]
    bars = ax2.bar(cts, vals,
                   color=[COLORS[i % len(COLORS)] for i in range(len(cts))],
                   edgecolor="#0F1117", linewidth=0.8)
    ax2.set_ylabel("Number of images")
    ax2.set_title("Images per country")
    ax2.grid(axis="y")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                 f"{val:,}", ha="center", fontsize=10)

    fig.tight_layout()
    save(fig, f"{dataset_name}_04_country_breakdown.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Sample image grid with annotations drawn
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_images(dataset_dir: Path, records, dataset_name, n=16):
    img_dirs = list(dataset_dir.rglob("images"))
    if not img_dirs:
        print("  No image directories found — skipping sample grid.")
        return

    # Build filename → path index
    img_index = {}
    for img_dir in img_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for p in img_dir.glob(ext):
                img_index[p.stem] = p

    # Pick records that have both image and annotations
    annotated = [r for r in records if r["file"] in img_index and r["n_boxes"] > 0]
    if not annotated:
        print("  Could not match images to annotations — skipping sample grid.")
        return

    sample = random.sample(annotated, min(n, len(annotated)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.suptitle(f"{dataset_name} — Sample Annotated Images (n={len(sample)})",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for ax, rec in zip(axes, sample):
        img_path = img_index[rec["file"]]
        img = cv2.imread(str(img_path))
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        for box in rec["boxes"]:
            color = PALETTE.get(box["class"], "#7F8C8D")
            rect  = mpatches.Rectangle(
                (box["xmin"], box["ymin"]),
                box["bw"], box["bh"],
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(box["xmin"], box["ymin"] - 4,
                    box["class"].replace("_", " "),
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="#0F1117", alpha=0.6, pad=1, edgecolor="none"))

        ax.set_title(f"{rec['country']} | {rec['n_boxes']} ann.", fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    save(fig, f"{dataset_name}_05_sample_images.png", dpi=120)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Brightness / contrast analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_brightness_analysis(dataset_dir: Path, dataset_name, max_imgs=500):
    images = find_images(dataset_dir)
    if not images:
        return

    sample = random.sample(images, min(max_imgs, len(images)))
    brightness, contrast, saturation = [], [], []

    for p in tqdm(sample, desc="Analysing brightness"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))
        saturation.append(float(np.mean(hsv[:, :, 1])))

    if not brightness:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset_name} — Image Quality Analysis  (n={len(brightness)} images)",
                 fontsize=16, fontweight="bold")

    for ax, data, label, color in zip(
        axes,
        [brightness, contrast, saturation],
        ["Mean brightness (0–255)", "Contrast (std dev)", "Mean saturation (0–255)"],
        [COLORS[0], COLORS[1], COLORS[2]]
    ):
        ax.hist(data, bins=40, color=color, edgecolor="#0F1117", alpha=0.85)
        ax.axvline(np.mean(data), color="white", linestyle="--",
                   linewidth=1.5, label=f"mean={np.mean(data):.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Image count")
        ax.set_title(label)
        ax.legend()
        ax.grid(axis="y")

    fig.tight_layout()
    save(fig, f"{dataset_name}_06_image_quality.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Annotation density heatmap (aggregate over all images)
# ══════════════════════════════════════════════════════════════════════════════

def plot_annotation_heatmap(records, dataset_name):
    grid_size = 20
    heatmap = defaultdict(lambda: np.zeros((grid_size, grid_size)))

    for r in records:
        if r["width"] == 0 or r["height"] == 0:
            continue
        for b in r["boxes"]:
            cx_n = b["cx"] / r["width"]
            cy_n = b["cy"] / r["height"]
            gx = min(int(cx_n * grid_size), grid_size - 1)
            gy = min(int(cy_n * grid_size), grid_size - 1)
            heatmap[b["class"]][gy, gx] += 1

    classes = list(heatmap.keys())
    if not classes:
        return

    cols = min(3, len(classes))
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle(f"{dataset_name} — Spatial Annotation Density per Class",
                 fontsize=16, fontweight="bold")

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    cmap = LinearSegmentedColormap.from_list("heat2", ["#1A1D27", "#F39C12", "#E74C3C"])
    for ax, cls in zip(axes_flat, classes):
        im = ax.imshow(heatmap[cls], cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, label="annotation count")
        ax.set_title(cls.replace("_", " ").title(),
                     color=PALETTE.get(cls, "white"))
        ax.set_xlabel("Image X (normalized)")
        ax.set_ylabel("Image Y (normalized)")

    for ax in axes_flat[len(classes):]:
        ax.axis("off")

    fig.tight_layout()
    save(fig, f"{dataset_name}_07_annotation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Train / val split summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_split_summary(records, dataset_name):
    split_counts = Counter(r["split"] for r in records)
    if len(split_counts) < 2:
        return

    splits = list(split_counts.keys())
    counts = [split_counts[s] for s in splits]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{dataset_name} — Train / Val Split", fontsize=16, fontweight="bold")
    bars = ax.bar(splits, counts,
                  color=COLORS[:len(splits)], edgecolor="#0F1117", linewidth=0.8)
    ax.set_ylabel("Number of images")
    ax.grid(axis="y")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", fontsize=12)

    fig.tight_layout()
    save(fig, f"{dataset_name}_08_split_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 9 — Colour channel mean per class (average crop appearance)
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_colour_profiles(dataset_dir: Path, records, dataset_name, max_crops=200):
    img_dirs = list(dataset_dir.rglob("images"))
    img_index = {}
    for img_dir in img_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in img_dir.glob(ext):
                img_index[p.stem] = p

    class_pixels = defaultdict(list)
    annotated = [r for r in records if r["file"] in img_index and r["n_boxes"] > 0]
    random.shuffle(annotated)

    for rec in tqdm(annotated[:max_crops * 3], desc="Sampling crops"):
        img = cv2.imread(str(img_index[rec["file"]]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        for box in rec["boxes"]:
            cls  = box["class"]
            if len(class_pixels[cls]) >= max_crops:
                continue
            x1 = max(0, int(box["xmin"]))
            y1 = max(0, int(box["ymin"]))
            x2 = min(w, int(box["xmax"]))
            y2 = min(h, int(box["ymax"]))
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mean_rgb = crop.reshape(-1, 3).mean(axis=0)
            class_pixels[cls].append(mean_rgb)

    classes = list(class_pixels.keys())
    if not classes:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{dataset_name} — Mean RGB per Damage Class (from crops)",
                 fontsize=16, fontweight="bold")

    # Grouped bar: R G B per class
    ax = axes[0]
    x  = np.arange(len(classes))
    w  = 0.25
    for i, (channel, color) in enumerate(zip(["R", "G", "B"],
                                              ["#E74C3C", "#2ECC71", "#3498DB"])):
        means = [np.mean([v[i] for v in class_pixels[c]]) for c in classes]
        ax.bar(x + i * w, means, w, label=channel, color=color,
               edgecolor="#0F1117", alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("Mean pixel value (0–255)")
    ax.set_title("RGB channel means per class")
    ax.legend()
    ax.grid(axis="y")

    # Colour swatches
    ax2 = axes[1]
    ax2.axis("off")
    for i, cls in enumerate(classes):
        mean_rgb = np.mean(class_pixels[cls], axis=0) / 255.0
        swatch   = np.ones((40, 200, 3)) * mean_rgb
        ax2.imshow(swatch, extent=[0, 1, i - 0.4, i + 0.4], aspect="auto")
        ax2.text(1.05, i, cls.replace("_", " "),
                 va="center", color=PALETTE.get(cls, "white"), fontsize=11)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(-0.8, len(classes) - 0.2)
    ax2.set_title("Average crop colour per class")

    fig.tight_layout()
    save(fig, f"{dataset_name}_09_colour_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 10 — Dataset summary card (for thesis title page)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_card(records, dataset_name, class_counts=None):
    n_images   = len(records)
    n_ann      = sum(r["n_boxes"] for r in records)
    n_empty    = sum(1 for r in records if r["n_boxes"] == 0)
    countries  = sorted(set(r["country"] for r in records))
    splits     = Counter(r["split"] for r in records)

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#0F1117")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0F1117")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, dataset_name.upper(),
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="#FFFFFF", transform=ax.transAxes)
    ax.text(0.5, 0.83, "Dataset Summary Card",
            ha="center", va="center", fontsize=14,
            color="#B0B0B0", transform=ax.transAxes)

    # Divider
    ax.axhline(0.78, color="#3A3D4D", linewidth=1, transform=ax.transAxes)

    # Stats
    stats = [
        ("Total images",      f"{n_images:,}"),
        ("Total annotations", f"{n_ann:,}"),
        ("Empty images",      f"{n_empty:,}  ({n_empty/n_images*100:.1f}%)"),
        ("Countries",         ", ".join(countries) if countries else "—"),
        ("Splits",            "  |  ".join(f"{k}: {v:,}" for k, v in splits.items())),
        ("Damage classes",    str(len(class_counts) if class_counts else "—")),
    ]
    y = 0.68
    for label, value in stats:
        ax.text(0.08, y, f"{label}:", ha="left", va="center", fontsize=12,
                color="#B0B0B0", transform=ax.transAxes)
        ax.text(0.42, y, value, ha="left", va="center", fontsize=12,
                color="#FFFFFF", fontweight="bold", transform=ax.transAxes)
        y -= 0.09

    # Class counts mini bar
    if class_counts:
        ax.axhline(0.10, color="#3A3D4D", linewidth=1, transform=ax.transAxes)
        ax.text(0.5, 0.07, "Class breakdown",
                ha="center", fontsize=10, color="#B0B0B0", transform=ax.transAxes)
        total   = sum(class_counts.values())
        x_start = 0.05
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            w = (count / total) * 0.90
            color = PALETTE.get(cls, "#7F8C8D")
            rect  = mpatches.FancyBboxPatch(
                (x_start, 0.01), w, 0.045,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="#0F1117",
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            if w > 0.06:
                ax.text(x_start + w / 2, 0.033,
                        cls.replace("_", "\n"),
                        ha="center", va="center", fontsize=7,
                        color="white", transform=ax.transAxes)
            x_start += w

    save(fig, f"{dataset_name}_00_summary_card.png", dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# Main per-dataset runner
# ══════════════════════════════════════════════════════════════════════════════

def inspect_rdd2022():
    dataset_dir = DATASETS_DIR / "rdd2022"
    if not dataset_dir.exists():
        print("  RDD2022 not found at data/datasets/rdd2022/ — skipping.")
        return

    print("\n── RDD2022 ─────────────────────────────────────────")
    records     = load_rdd2022_annotations(dataset_dir)
    print(f"  Loaded {len(records):,} annotation files")

    class_counts = plot_class_distribution(records, "RDD2022")
    plot_image_sizes(records, "RDD2022")
    plot_bbox_sizes(records, "RDD2022")
    plot_country_breakdown(records, "RDD2022")
    plot_sample_images(dataset_dir, records, "RDD2022", n=16)
    plot_brightness_analysis(dataset_dir, "RDD2022", max_imgs=500)
    plot_annotation_heatmap(records, "RDD2022")
    plot_split_summary(records, "RDD2022")
    plot_class_colour_profiles(dataset_dir, records, "RDD2022", max_crops=150)
    plot_summary_card(records, "RDD2022", class_counts)
    print(f"  → All RDD2022 plots saved to {OUTPUT_DIR}")


def inspect_cfd():
    dataset_dir = DATASETS_DIR / "cfd"
    if not dataset_dir.exists():
        print("  CFD not found at data/datasets/cfd/ — skipping.")
        return

    print("\n── CFD (Crack Forest Dataset) ──────────────────────")
    images = find_images(dataset_dir)
    print(f"  Found {len(images)} images")

    if not images:
        return

    # Sample grid of crack images
    sample = random.sample(images, min(16, len(images)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("CFD — Sample Images", fontsize=16, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, p in zip(axes_flat, sample):
        img = cv2.imread(str(p))
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(p.name, fontsize=7)
        ax.axis("off")
    for ax in axes_flat[len(sample):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "CFD_01_sample_images.png", dpi=100)

    # Look for masks
    mask_paths = list(dataset_dir.rglob("*.bmp")) + list(dataset_dir.rglob("*.png"))
    mask_paths = [p for p in mask_paths if "GT" in str(p) or "mask" in str(p).lower()
                  or "label" in str(p).lower()]
    if mask_paths:
        sample_masks = random.sample(mask_paths, min(8, len(mask_paths)))
        fig, axes = plt.subplots(2, min(4, len(sample_masks)),
                                 figsize=(16, 8))
        fig.suptitle("CFD — Image + Ground Truth Mask Pairs",
                     fontsize=16, fontweight="bold")
        for i, mask_p in enumerate(sample_masks[:4]):
            img_p = mask_p.parent.parent / "image" / mask_p.name
            if not img_p.exists():
                img_p = mask_p  # fallback
            img  = cv2.imread(str(img_p))
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title("Image", fontsize=9)
                axes[0, i].axis("off")
            if mask is not None:
                axes[1, i].imshow(mask, cmap="hot")
                axes[1, i].set_title("GT Mask", fontsize=9)
                axes[1, i].axis("off")
        fig.tight_layout()
        save(fig, "CFD_02_image_mask_pairs.png", dpi=100)

    plot_brightness_analysis(dataset_dir, "CFD", max_imgs=200)
    print(f"  → CFD plots saved to {OUTPUT_DIR}")


def inspect_pothole600():
    dataset_dir = DATASETS_DIR / "pothole600"
    if not dataset_dir.exists():
        print("  Pothole-600 not found — skipping.")
        return

    print("\n── Pothole-600 ─────────────────────────────────────")
    images = find_images(dataset_dir / "images") if (dataset_dir / "images").exists() \
             else find_images(dataset_dir)
    print(f"  Found {len(images)} images")

    if images:
        sample = random.sample(images, min(12, len(images)))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle("Pothole-600 — Sample Images", fontsize=16, fontweight="bold")
        for ax, p in zip(axes.flatten(), sample):
            img = cv2.imread(str(p))
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(p.name, fontsize=8)
            ax.axis("off")
        for ax in axes.flatten()[len(sample):]:
            ax.axis("off")
        fig.tight_layout()
        save(fig, "Pothole600_01_sample_images.png", dpi=100)

    plot_brightness_analysis(dataset_dir, "Pothole600", max_imgs=200)
    print(f"  → Pothole-600 plots saved to {OUTPUT_DIR}")


def inspect_gaps():
    dataset_dir = DATASETS_DIR / "gaps"
    if not dataset_dir.exists():
        print("  GAPs not found — skipping.")
        return

    print("\n── GAPs ────────────────────────────────────────────")
    images = find_images(dataset_dir)
    print(f"  Found {len(images)} images")
    if images:
        plot_brightness_analysis(dataset_dir, "GAPs", max_imgs=200)
        sample = random.sample(images, min(12, len(images)))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle("GAPs — Sample Images", fontsize=16, fontweight="bold")
        for ax, p in zip(axes.flatten(), sample):
            img = cv2.imread(str(p))
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        for ax in axes.flatten()[len(sample):]:
            ax.axis("off")
        fig.tight_layout()
        save(fig, "GAPs_01_sample_images.png", dpi=100)
    print(f"  → GAPs plots saved to {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dataset inspection and visualization")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "rdd2022", "cfd", "pothole600", "gaps"],
                        help="Which dataset to inspect")
    args = parser.parse_args()

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)

    if args.dataset in ("all", "rdd2022"):
        inspect_rdd2022()
    if args.dataset in ("all", "cfd"):
        inspect_cfd()
    if args.dataset in ("all", "pothole600"):
        inspect_pothole600()
    if args.dataset in ("all", "gaps"):
        inspect_gaps()

    print("\n" + "=" * 60)
    print(f"Done. All plots saved to:  {OUTPUT_DIR}")
    print("Files generated per dataset:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()