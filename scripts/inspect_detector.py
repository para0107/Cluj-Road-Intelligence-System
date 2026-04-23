"""
scripts/inspect_detector.py
-----------------------------
Visualise and analyse the output of pipeline/detector.py (Stage 2).

What this produces
------------------
Console:
  - Detection summary: total boxes, per-class counts, confidence stats,
    frames with / without detections

Saved to --output-dir:
  1. summary.txt                  Plain-text summary
  2. class_distribution.png       Bar chart: detection counts per class
  3. confidence_distribution.png  Histogram of confidence scores per class
  4. detection_rate_timeline.png  Line plot: n_detections per frame over time
  5. detection_grid.png           Grid of frames with bboxes drawn on them
                                  (only frames WITH detections, up to --grid-n)
  6. all_frames_grid.png          Grid of ALL frames (sampled), with or without
                                  detections, so you can see what the model missed
  7. spatial_heatmap.png          2D heatmap of bbox centre positions across all
                                  frames — shows where in the frame detections
                                  cluster (should be lower-centre for dashcam)
  8. confidence_map.png           Same spatial layout, but colour = mean confidence

Usage:
    python scripts/inspect_detector.py \
        --detections data/processed/detections/validation/detections.json \
        --output-dir data/processed/inspection/detector/ \
        [--grid-n 20] [--conf-threshold 0.35] [--no-display]

Dependencies:
    pip install matplotlib numpy opencv-python
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class colours — consistent with RDD2022 convention
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "longitudinal_crack":  "#E74C3C",   # red
    "transverse_crack":    "#E67E22",   # orange
    "alligator_crack":     "#27AE60",   # green
    "pothole":             "#2980B9",   # blue
    "patch_deterioration": "#8E44AD",   # purple
}
DEFAULT_COLOR = "#95A5A6"  # grey for unknown


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_detections(path: str) -> Tuple[List[Dict], Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    meta   = {k: v for k, v in payload.items() if k != "frames"}
    logger.info("Loaded: %s  (%d frames, %d total boxes)",
                path, len(frames), meta.get("n_boxes", 0))
    return frames, meta


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(frames: List[Dict]) -> Dict:
    n_frames = len(frames)
    all_boxes = [b for fr in frames for b in fr.get("boxes", [])]
    n_boxes   = len(all_boxes)

    frames_with = sum(1 for fr in frames if fr.get("n_detections", 0) > 0)

    class_counts: Dict[str, int] = {}
    class_confs:  Dict[str, List[float]] = {}
    for b in all_boxes:
        cn = b["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1
        class_confs.setdefault(cn, []).append(b["confidence"])

    confs = [b["confidence"] for b in all_boxes]
    return {
        "n_frames":         n_frames,
        "n_boxes":          n_boxes,
        "frames_with_detections": frames_with,
        "frames_no_detections":   n_frames - frames_with,
        "detection_rate_pct": 100 * frames_with / max(n_frames, 1),
        "class_counts":     class_counts,
        "class_confs":      class_confs,
        "conf_mean":        float(np.mean(confs)) if confs else None,
        "conf_min":         float(np.min(confs))  if confs else None,
        "conf_max":         float(np.max(confs))  if confs else None,
    }


def print_summary(s: Dict) -> str:
    lines = [
        "=" * 58,
        "  CRIS Stage 2 — RT-DETR Detection Summary",
        "=" * 58,
        f"  Frames processed         : {s['n_frames']}",
        f"  Frames with detections   : {s['frames_with_detections']} "
        f"({s['detection_rate_pct']:.0f}%)",
        f"  Frames with no detection : {s['frames_no_detections']}",
        f"  Total bounding boxes     : {s['n_boxes']}",
        "",
        "  Class breakdown",
    ]
    for cls, cnt in sorted(s["class_counts"].items(), key=lambda x: -x[1]):
        mean_c = np.mean(s["class_confs"][cls])
        lines.append(f"    {cls:<28} : {cnt:>4}  (mean conf {mean_c:.2f})")

    if s["conf_mean"] is not None:
        lines += [
            "",
            "  Confidence (all classes)",
            f"    min / mean / max : {s['conf_min']:.3f} / "
            f"{s['conf_mean']:.3f} / {s['conf_max']:.3f}",
        ]
    else:
        lines += ["", "  No detections found in any frame."]

    lines.append("=" * 58)
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Plot 1: Class distribution bar chart
# ---------------------------------------------------------------------------

def plot_class_distribution(summary: Dict, out_path: str) -> None:
    counts = summary["class_counts"]
    if not counts:
        logger.warning("No detections — skipping class distribution plot")
        return

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [CLASS_COLORS.get(k, DEFAULT_COLOR) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("RT-DETR Detection Counts per Class", fontsize=13, pad=10)
    ax.set_ylabel("Number of detections")
    ax.set_ylim(0, max(values) * 1.2)
    plt.xticks(rotation=15, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 2: Confidence histogram per class
# ---------------------------------------------------------------------------

def plot_confidence_distribution(summary: Dict, out_path: str) -> None:
    class_confs = summary["class_confs"]
    if not class_confs:
        logger.warning("No detections — skipping confidence plot")
        return

    n = len(class_confs)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    # Normalise axes to always be a 2D array
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.axis("off")

    for i, (cls, confs) in enumerate(class_confs.items()):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("on")
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        ax.hist(confs, bins=20, range=(0, 1), color=color, edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline(np.mean(confs), color="black", linewidth=1.2,
                   linestyle="--", label=f"mean={np.mean(confs):.2f}")
        ax.set_title(cls.replace("_", " "), fontsize=10)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Confidence Score Distribution per Class", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3: Detection count over time
# ---------------------------------------------------------------------------

def plot_detection_timeline(frames: List[Dict], out_path: str) -> None:
    t_vals = [fr["timestamp_s"] for fr in frames]
    n_vals = [fr.get("n_detections", 0) for fr in frames]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(t_vals, n_vals, width=0.4, color="#2980B9", alpha=0.8, edgecolor="white")
    ax.set_title("Number of Detections per Frame over Time", fontsize=13, pad=10)
    ax.set_xlabel("Time offset (s)")
    ax.set_ylabel("Detections")
    ax.set_ylim(0, max(n_vals) + 1 if n_vals else 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Helper: draw bboxes on a frame image
# ---------------------------------------------------------------------------

def draw_boxes(
    img_bgr: np.ndarray,
    boxes: List[Dict],
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes with class labels on a copy of img_bgr."""
    out = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cls  = b["class_name"]
        conf = b["confidence"]
        hex_color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        # Convert hex to BGR
        h = hex_color.lstrip("#")
        r, g, bv = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        color_bgr = (bv, g, r)

        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)

        label = f"{cls.replace('_', ' ')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color_bgr, -1)
        cv2.putText(out, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Plot 4: Detection grid — only frames WITH detections
# ---------------------------------------------------------------------------

def plot_detection_grid(
    frames: List[Dict],
    out_path: str,
    n: int = 20,
    frames_dir_override: Optional[str] = None,
) -> None:
    """Grid showing only frames that had at least one detection, with boxes drawn."""
    detected = [fr for fr in frames if fr.get("n_detections", 0) > 0]
    if not detected:
        logger.warning("No frames with detections — skipping detection grid")
        return

    indices = np.linspace(0, len(detected) - 1, min(n, len(detected)), dtype=int)
    selected = [detected[i] for i in indices]

    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.0))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.axis("off")

    for flat_i, fr in enumerate(selected):
        r_i, c_i = divmod(flat_i, cols)
        ax = axes[r_i][c_i]

        frame_path = fr["frame_path"]
        if frames_dir_override and not Path(frame_path).exists():
            frame_path = str(Path(frames_dir_override) / Path(frame_path).name)

        if not Path(frame_path).exists():
            ax.text(0.5, 0.5, "File not\nfound", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=9)
            continue

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue

        img_annotated = draw_boxes(img_bgr, fr.get("boxes", []))
        img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        n_det = fr.get("n_detections", 0)
        t_str = f"{fr['timestamp_s']:.1f}s"
        cls_str = ", ".join(set(b["class_name"].replace("_", " ")
                               for b in fr.get("boxes", [])))
        ax.set_title(f"#{fr['frame_index']} t={t_str}\n{n_det} det: {cls_str}",
                     fontsize=7, pad=3)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Frames WITH Detections  ({len(selected)} shown of {len(detected)} total)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 5: All frames grid (sampled), with and without detections
# ---------------------------------------------------------------------------

def plot_all_frames_grid(
    frames: List[Dict],
    out_path: str,
    n: int = 20,
    frames_dir_override: Optional[str] = None,
) -> None:
    """Sampled grid of ALL frames — shows what the model hit AND missed."""
    if not frames:
        return

    indices = np.linspace(0, len(frames) - 1, min(n, len(frames)), dtype=int)
    selected = [frames[i] for i in indices]

    cols = min(5, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.axis("off")

    for flat_i, fr in enumerate(selected):
        r_i, c_i = divmod(flat_i, cols)
        ax = axes[r_i][c_i]

        frame_path = fr["frame_path"]
        if frames_dir_override and not Path(frame_path).exists():
            frame_path = str(Path(frames_dir_override) / Path(frame_path).name)

        if not Path(frame_path).exists():
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8)
            continue

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue

        n_det = fr.get("n_detections", 0)
        if n_det > 0:
            img_bgr = draw_boxes(img_bgr, fr.get("boxes", []))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

        # Green border = hit, red border = miss
        border = "#27AE60" if n_det > 0 else "#E74C3C"
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3)
            spine.set_visible(True)

        label = f"#{fr['frame_index']} {fr['timestamp_s']:.1f}s\n"
        label += f"✓ {n_det} det" if n_det > 0 else "✗ no det"
        ax.set_title(label, fontsize=7, pad=2,
                     color="#27AE60" if n_det > 0 else "#E74C3C")

    fig.suptitle(
        "All Frames — Green border = detection, Red border = no detection",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 6: Spatial heatmap of bbox centres
# ---------------------------------------------------------------------------

def plot_spatial_heatmap(
    frames: List[Dict],
    out_path: str,
    grid_size: int = 20,
) -> None:
    """
    2D heatmap showing WHERE in the frame detections appear.
    Boxes are normalised to [0,1] relative to frame dimensions.
    Expected: concentration in lower-centre (road surface region).
    """
    all_boxes = [b for fr in frames for b in fr.get("boxes", [])]
    if not all_boxes:
        logger.warning("No detections — skipping spatial heatmap")
        return

    # Get a frame to find image dimensions
    sample_fr = next((fr for fr in frames if fr.get("n_detections", 0) > 0), None)
    if sample_fr is None:
        return

    img_w = sample_fr.get("image_width", 640)
    img_h = sample_fr.get("image_height", 480)

    # Normalised centre positions
    cx_vals = [(b["x1"] + b["x2"]) / 2 / img_w for b in all_boxes]
    cy_vals = [(b["y1"] + b["y2"]) / 2 / img_h for b in all_boxes]

    heatmap, xedges, yedges = np.histogram2d(
        cx_vals, cy_vals,
        bins=grid_size,
        range=[[0, 1], [0, 1]],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(heatmap.T, origin="upper", cmap="YlOrRd",
                   extent=[0, 1, 1, 0], aspect="auto")
    plt.colorbar(im, ax=ax, label="Detection count")
    ax.set_title("Detection Spatial Density", fontsize=11)
    ax.set_xlabel("Normalised x (left→right)")
    ax.set_ylabel("Normalised y (top→bottom)")
    ax.axhline(0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

    # Right: scatter coloured by class
    ax2 = axes[1]
    for cls, color in CLASS_COLORS.items():
        cx = [(b["x1"] + b["x2"]) / 2 / img_w
              for fr in frames for b in fr.get("boxes", [])
              if b["class_name"] == cls]
        cy = [(b["y1"] + b["y2"]) / 2 / img_h
              for fr in frames for b in fr.get("boxes", [])
              if b["class_name"] == cls]
        if cx:
            ax2.scatter(cx, cy, c=color, s=18, alpha=0.7,
                        label=cls.replace("_", " "))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)  # Flip y so top-left is origin
    ax2.set_title("Detection Centres by Class", fontsize=11)
    ax2.set_xlabel("Normalised x")
    ax2.set_ylabel("Normalised y (0=top)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
    ax2.axvline(0.5, color="grey", linewidth=0.5, linestyle="--")

    fig.suptitle(
        f"Spatial Distribution of {len(all_boxes)} Detections "
        f"across {img_w}×{img_h} frames",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Stage 2 (detector) output — print summary and save plots."
    )
    parser.add_argument("--detections", required=True,
                        help="Path to detections.json from detector.py")
    parser.add_argument("--output-dir", default="data/processed/inspection/detector",
                        help="Where to save plots and summary")
    parser.add_argument("--frames-dir", default=None,
                        help="Override frame image directory if paths have changed")
    parser.add_argument("--grid-n", type=int, default=20,
                        help="Max frames to show in detection grid (default 20)")
    parser.add_argument("--no-display", action="store_true",
                        help="Save only, do not call plt.show()")
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, meta = load_detections(args.detections)
    if not frames:
        logger.error("No frames in detections file.")
        sys.exit(1)

    summary = compute_summary(frames)
    text = print_summary(summary)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    logger.info("Saved: %s", out_dir / "summary.txt")

    plot_class_distribution(summary,     str(out_dir / "class_distribution.png"))
    plot_confidence_distribution(summary, str(out_dir / "confidence_distribution.png"))
    plot_detection_timeline(frames,      str(out_dir / "detection_rate_timeline.png"))
    plot_detection_grid(
        frames, str(out_dir / "detection_grid.png"),
        n=args.grid_n, frames_dir_override=args.frames_dir,
    )
    plot_all_frames_grid(
        frames, str(out_dir / "all_frames_grid.png"),
        n=args.grid_n, frames_dir_override=args.frames_dir,
    )
    plot_spatial_heatmap(frames,         str(out_dir / "spatial_heatmap.png"))

    if not args.no_display:
        plt.show()

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    print("Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()