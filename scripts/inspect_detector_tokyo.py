"""
inspect_detector_tokyo.py
=========================
Reads data/processed/detections/tokyo/detections.json and produces:
  - summary.txt
  - class_distribution.png
  - confidence_distribution.png
  - detection_rate_timeline.png
  - detection_grid.png  (20 representative annotated frames)
  - all_frames_grid.png (all frames, one cell each)
  - spatial_heatmap.png
  - annotated_frames/   (one JPEG per detection frame, for manual validation)

This is the EXACT same inspector used for the Cluj video, pointed at the
Tokyo output directory. No modification to logic — this is intentional so
the outputs are directly comparable.

Usage:
    python inspect_detector_tokyo.py
    python inspect_detector_tokyo.py --no-display        # headless / server
    python inspect_detector_tokyo.py --skip-frames       # skip annotated export

Author: CRIS pipeline — Paraschiv Tudor, BBU 2026
Reference: Zhao et al. (2024) https://arxiv.org/abs/2304.08069
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inspect_tokyo.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DETECTIONS_PATH = Path("data/processed/detections/tokyo/detections.json")
OUTPUT_DIR      = Path("data/processed/inspection/tokyo")
EXPORT_SCALE    = 1.5   # same as Cluj

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "patch_deterioration",
]

# Colours — same as Cluj inspector for visual consistency
CLASS_COLORS = {
    "longitudinal_crack"  : "#e74c3c",   # red
    "transverse_crack"    : "#e67e22",   # orange
    "alligator_crack"     : "#9b59b6",   # purple
    "pothole"             : "#3498db",   # blue
    "patch_deterioration" : "#27ae60",   # green
}
DEFAULT_COLOR = "#95a5a6"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_detections(path: Path):
    if not path.exists():
        logger.error("detections.json not found: %s", path)
        logger.error("Run run_tokyo_validation.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        frames = json.load(f)
    total_boxes = sum(r["n_detections"] for r in frames)
    logger.info("Loaded: %s  (%d frames, %d total boxes)", path, len(frames), total_boxes)
    return frames


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def write_summary(frames: list, out_dir: Path):
    frames_with = [r for r in frames if r["n_detections"] > 0]
    total_boxes = sum(r["n_detections"] for r in frames)
    class_counts = {}
    class_confs  = {}
    for r in frames:
        for b in r["boxes"]:
            cn = b["class_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1
            class_confs.setdefault(cn, []).append(b["confidence"])

    all_confs = [b["confidence"] for r in frames for b in r["boxes"]]

    lines = [
        "=" * 58,
        "  CRIS Stage 2 — Tokyo RT-DETR Detection Summary",
        "=" * 58,
        f"  Frames processed         : {len(frames)}",
        f"  Frames with detections   : {len(frames_with)} ({100*len(frames_with)/max(1,len(frames)):.1f}%)",
        f"  Frames with no detection : {len(frames) - len(frames_with)}",
        f"  Total bounding boxes     : {total_boxes}",
        "  Class breakdown",
    ]
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        if count > 0:
            mc = sum(class_confs[cls]) / len(class_confs[cls])
            lines.append(f"    {cls:<30}: {count:4d}  (mean conf {mc:.2f})")
        else:
            lines.append(f"    {cls:<30}:    0")

    if all_confs:
        lines += [
            "  Confidence (all classes)",
            f"    min / mean / max : {min(all_confs):.3f} / "
            f"{sum(all_confs)/len(all_confs):.3f} / {max(all_confs):.3f}",
        ]
    lines.append("=" * 58)

    out_path = out_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved: %s", out_path)
    print("\n" + "\n".join(lines))
    return class_counts, class_confs, all_confs


# ---------------------------------------------------------------------------
# Plots (identical logic to Cluj inspector)
# ---------------------------------------------------------------------------

def plot_class_distribution(class_counts: dict, out_dir: Path):
    classes = [c for c in CLASS_NAMES if class_counts.get(c, 0) > 0]
    counts  = [class_counts[c] for c in classes]
    colors  = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("Detection count")
    ax.set_title("Tokyo — Class distribution (RT-DETR, conf≥0.35)", fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_confidence_distribution(frames: list, out_dir: Path):
    class_confs = {}
    for r in frames:
        for b in r["boxes"]:
            class_confs.setdefault(b["class_name"], []).append(b["confidence"])

    if not class_confs:
        logger.warning("No detections — skipping confidence distribution plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0.30, 1.0, 30)
    for cls, confs in class_confs.items():
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        ax.hist(confs, bins=bins, alpha=0.6, color=color,
                label=cls.replace("_", " "), edgecolor="white", linewidth=0.5)
    ax.axvline(0.35, color="black", linestyle="--", linewidth=1.2, label="threshold (0.35)")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title("Tokyo — Confidence score distribution by class", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / "confidence_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_rate_timeline(frames: list, out_dir: Path):
    if not frames:
        return
    bin_size_s = 30
    max_t = max(r["timestamp_s"] for r in frames)
    n_bins = max(1, int(max_t / bin_size_s) + 1)
    counts = [0] * n_bins

    for r in frames:
        if r["n_detections"] > 0:
            idx = min(int(r["timestamp_s"] / bin_size_s), n_bins - 1)
            counts[idx] += r["n_detections"]

    t_axis = [i * bin_size_s / 60 for i in range(n_bins)]   # minutes

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(t_axis, counts, width=bin_size_s/60 * 0.9,
           color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Detections per 30s bin")
    ax.set_title("Tokyo — Detection rate over time", fontweight="bold")
    plt.tight_layout()
    out = out_dir / "detection_rate_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_spatial_heatmap(frames: list, out_dir: Path):
    """
    Heatmap of bbox centre positions in normalised frame coordinates.
    Reveals where in the frame the model is detecting — bottom = road surface (correct).
    """
    xs, ys = [], []
    for r in frames:
        for b in r["boxes"]:
            cx, cy = b["bbox_xywhn"][0], b["bbox_xywhn"][1]
            xs.append(cx)
            ys.append(cy)

    if not xs:
        logger.warning("No detections — skipping spatial heatmap")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    h, xedges, yedges = np.histogram2d(xs, ys, bins=20, range=[[0,1],[0,1]])
    ax.imshow(h.T, origin="upper", extent=[0,1,0,1],
              cmap="hot", aspect="auto", interpolation="bilinear")
    ax.set_xlabel("Normalised x (left → right)")
    ax.set_ylabel("Normalised y (top → bottom)")
    ax.set_title("Tokyo — Spatial heatmap of bbox centres", fontweight="bold")
    plt.colorbar(ax.images[0], ax=ax, label="Detection count")
    plt.tight_layout()
    out = out_dir / "spatial_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_grid(frames: list, out_dir: Path, n: int = 20):
    """
    Grid of n representative detection frames with bboxes drawn.
    """
    det_frames = [r for r in frames if r["n_detections"] > 0]
    if not det_frames:
        logger.warning("No detection frames — skipping grid")
        return

    # Sample evenly across the video timeline
    step = max(1, len(det_frames) // n)
    sampled = det_frames[::step][:n]

    cols = 5
    rows = math.ceil(len(sampled) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for i, fr in enumerate(sampled):
        img = cv2.imread(fr["frame_path"])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        for b in fr["boxes"]:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            color_hex = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
            color_rgb = tuple(int(color_hex[j:j+2], 16)/255 for j in (0, 2, 4))
            rect = mpatches.FancyBboxPatch(
                (x1/w, 1 - y2/h), (x2-x1)/w, (y2-y1)/h,
                boxstyle="square,pad=0", linewidth=1.5,
                edgecolor=color_rgb, facecolor="none",
                transform=axes[i].transAxes,
            )
            axes[i].add_patch(rect)

        axes[i].imshow(img_rgb)
        axes[i].set_title(
            f"#{fr['frame_index']} t={fr['timestamp_s']:.0f}s\n"
            f"{fr['lighting']} | {fr['n_detections']} det",
            fontsize=7,
        )
        axes[i].axis("off")

    plt.suptitle("Tokyo — Detection grid (sampled)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "detection_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_all_frames_grid(frames: list, out_dir: Path):
    """
    One small cell per frame — green = detection, grey = no detection.
    Gives a temporal overview of where detections cluster.
    """
    n     = len(frames)
    cols  = 80
    rows  = math.ceil(n / cols)
    cell  = 8   # pixels per cell

    img = np.ones((rows * cell, cols * cell, 3), dtype=np.uint8) * 60  # dark grey bg

    for idx, r in enumerate(frames):
        row = idx // cols
        col = idx % cols
        y0, y1 = row * cell, (row + 1) * cell
        x0, x1 = col * cell, (col + 1) * cell
        if r["n_detections"] > 0:
            # Colour by dominant class
            dominant = max(r["boxes"], key=lambda b: b["confidence"])["class_name"]
            hex_c = CLASS_COLORS.get(dominant, DEFAULT_COLOR).lstrip("#")
            color = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))  # BGR
            img[y0:y1, x0:x1] = color
        else:
            img[y0:y1, x0:x1] = 35  # no detection — very dark

    out = out_dir / "all_frames_grid.png"
    cv2.imwrite(str(out), img)
    logger.info("Saved: %s  (%d frames, %d×%d cells)", out, n, rows, cols)


# ---------------------------------------------------------------------------
# Annotated frame export (manual validation)
# ---------------------------------------------------------------------------

def export_annotated_frames(frames: list, out_dir: Path, scale: float = 1.5):
    """
    Export one JPEG per detection frame with bboxes, labels, confidence,
    and a top info banner. Same format as Cluj annotated frames.
    """
    out = out_dir / "annotated_frames"
    out.mkdir(parents=True, exist_ok=True)

    det_frames   = [r for r in frames if r["n_detections"] > 0]
    n_written    = 0
    font_thickness = max(1, int(scale))

    for fr in det_frames:
        img = cv2.imread(fr["frame_path"])
        if img is None:
            continue

        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)
        h_img, w_img = img.shape[:2]

        # Draw bounding boxes
        for b in fr["boxes"]:
            x1, y1, x2, y2 = [int(v * scale) for v in b["bbox_xyxy"]]
            hex_c = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
            color_bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, max(1, int(2 * scale)))

            label = f"{b['class_name'].replace('_', ' ')} {b['confidence']:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5 * scale, font_thickness)[0]
            cv2.rectangle(img, (x1, y2 - lh - 6), (x1 + lw + 4, y2), color_bgr, -1)
            cv2.putText(img, label, (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Crop thumbnail in top-left corner
            crop_h = max(40, int(50 * scale))
            crop_w = max(60, int(80 * scale))
            c_x1, c_y1 = max(0, x1), max(0, y1)
            c_x2, c_y2 = min(w_img, x2), min(h_img, y2)
            if c_x2 > c_x1 and c_y2 > c_y1:
                crop = img[c_y1:c_y2, c_x1:c_x2]
                crop_resized = cv2.resize(crop, (crop_w, crop_h))
                # find free spot at bottom-left
                idx_box = fr["boxes"].index(b)
                ty = h_img - (crop_h + 2) * (idx_box + 1)
                if ty >= 0:
                    # label background
                    cv2.rectangle(img, (0, ty - int(22*scale)),
                                  (int(130*scale), ty), color_bgr, -1)
                    cv2.putText(img,
                                f"{b['class_name'].replace('_',' ')} {b['confidence']:.2f}",
                                (4, ty - int(5*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45*scale,
                                (255,255,255), font_thickness, cv2.LINE_AA)
                    img[ty:ty+crop_h, 0:crop_w] = crop_resized

        # Top info banner
        n_det  = fr["n_detections"]
        banner = (f"#{fr['frame_index']}  t={fr['timestamp_s']}s  "
                  f"{fr['lighting']}  | {n_det} detection(s)  [TOKYO]")
        bh = int(30 * scale)
        cv2.rectangle(img, (0, 0), (w_img, bh), (20, 20, 20), -1)
        cv2.putText(img, banner, (int(8*scale), int(20*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale,
                    (220, 220, 220), font_thickness, cv2.LINE_AA)

        # Detection list overlay
        for i, b in enumerate(fr["boxes"]):
            hex_c = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
            color_bgr = (int(hex_c[4:6],16), int(hex_c[2:4],16), int(hex_c[0:2],16))
            line = f"[{i+1}] {b['class_name'].replace('_',' ')} {b['confidence']:.2f}"
            ly = int((bh + 5)*scale) + i * int(22*scale)
            cv2.putText(img, line, (int(8*scale), ly + int(20*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45*scale,
                        color_bgr, font_thickness, cv2.LINE_AA)

        # Filename
        det_names = "_".join(sorted(set(
            b["class_name"]
            .replace("longitudinal_crack", "lng")
            .replace("transverse_crack", "trs")
            .replace("alligator_crack", "alg")
            .replace("patch_deterioration", "pat")
            for b in fr["boxes"]
        )))
        fname = (f"frame_{fr['frame_index']:06d}"
                 f"_t{fr['timestamp_s']:.0f}s"
                 f"_{n_det}det"
                 f"_{det_names}.jpg")
        cv2.imwrite(str(out / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_written += 1

    logger.info("Exported %d annotated frames → %s", n_written, out)
    print(f"  Annotated frames: {n_written} files in {out.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect Tokyo RT-DETR detections")
    parser.add_argument("--detections", default=str(DETECTIONS_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip annotated frame export (faster, no disk space needed)")
    parser.add_argument("--export-scale", type=float, default=EXPORT_SCALE)
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_detections(Path(args.detections))
    if not frames:
        logger.error("No frames found in detections file.")
        sys.exit(1)

    class_counts, class_confs, all_confs = write_summary(frames, out_dir)
    plot_class_distribution(class_counts, out_dir)
    plot_confidence_distribution(frames, out_dir)
    plot_detection_rate_timeline(frames, out_dir)
    plot_spatial_heatmap(frames, out_dir)
    plot_detection_grid(frames, out_dir, n=20)
    plot_all_frames_grid(frames, out_dir)

    if not args.skip_frames:
        logger.info("Exporting annotated frames (scale=%.1f)...", args.export_scale)
        export_annotated_frames(frames, out_dir, scale=args.export_scale)
    else:
        logger.info("Skipping annotated frame export (--skip-frames)")

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    logger.info("Inspection complete.")


if __name__ == "__main__":
    main()