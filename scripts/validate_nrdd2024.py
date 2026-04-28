"""
validate_nrdd2024.py
====================
Runs the N-RDD2024 fine-tuned RT-DETR-L (runs/detect/nrdd_2024/best.pt)
on two real-world dashcam videos and produces a full inspection report
for each, mirroring the structure of inspect_detector_tokyo.py.

Videos:
  Tokyo : data/raw/footage/Tokyo Drive 4K | Takanawa - Azabudai Hills - Shinjuku [v7JZ9DSSRsY].mp4
  Cluj  : data/raw/footage/validation_cluj.mp4

Outputs per video  →  data/validation_nrdd_2024/<video>/
  detections.json
  summary.txt
  summary_comparison.txt          (Tokyo vs Cluj side-by-side)
  class_distribution.png
  confidence_distribution.png
  detection_rate_timeline.png
  spatial_heatmap.png
  detection_grid.png
  all_frames_grid.png

Bounding box frames  →  data/validation_nrdd_2024/bounding_boxes/<video>/

Model: RT-DETR-L fine-tuned on N-RDD2024 ten-class taxonomy
       (Kaya & Codur, 2024 — doi:10.17632/27c8pwsd6v.3)
Detection: Zhao et al. (2024) https://arxiv.org/abs/2304.08069

Author: RIDS pipeline — Paraschiv Tudor, BBU 2026
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Logging — file + stdout, mirrors existing inspector
# ---------------------------------------------------------------------------
LOG_PATH = Path("data/validation_nrdd_2024/validate_nrdd2024.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT   = Path("runs/detect/nrdd_2024/best.pt")
FRAME_RATE   = 2.0        # frames per second to extract (1 frame / 0.5 s)
CONF_THRESH  = 0.35       # operational RIDS threshold
IOU_THRESH   = 0.6
IMGSZ        = 640
EXPORT_SCALE = 1.5

VIDEOS = {
    "tokyo": Path(
        "data/raw/footage/"
        "Tokyo Drive 4K ｜ Takanawa - Azabudai Hills - Shinjuku [v7JZ9DSSRsY].mp4"
    ),
    "cluj": Path("data/raw/footage/validation_cluj.mp4"),
}

OUTPUT_BASE  = Path("data/validation_nrdd_2024")
BBOX_BASE    = OUTPUT_BASE / "bounding_boxes"

# ---------------------------------------------------------------------------
# N-RDD2024 ten-class taxonomy
# (Kaya & Codur, 2024 — doi:10.17632/27c8pwsd6v.3)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "longitudinal_crack",        # D00 — ID 0
    "transverse_crack",          # D10 — ID 1
    "alligator_crack",           # D20 — ID 2
    "repaired_crack",            # D30 — ID 3
    "pothole",                   # D40 — ID 4
    "pedestrian_crossing_blur",  # D50 — ID 5
    "lane_line_blur",            # D60 — ID 6
    "manhole_cover",             # D70 — ID 7
    "patchy_road",               # D80 — ID 8
    "rutting",                   # D90 — ID 9
]

CLASS_COLORS = {
    "longitudinal_crack":        "#e74c3c",   # red
    "transverse_crack":          "#e67e22",   # orange
    "alligator_crack":           "#9b59b6",   # purple
    "repaired_crack":            "#1abc9c",   # teal
    "pothole":                   "#3498db",   # blue
    "pedestrian_crossing_blur":  "#f39c12",   # amber
    "lane_line_blur":            "#d35400",   # dark orange
    "manhole_cover":             "#7f8c8d",   # grey
    "patchy_road":               "#27ae60",   # green
    "rutting":                   "#8e44ad",   # violet
}
DEFAULT_COLOR = "#95a5a6"

# ---------------------------------------------------------------------------
# Lighting classification (mirrors preprocessor.py)
# ---------------------------------------------------------------------------

def classify_lighting(frame: np.ndarray) -> str:
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum  = float(gray.mean())
    if mean_lum < 60:
        return "low_light"
    elif mean_lum < 160:
        return "overcast"
    return "daylight"


# ---------------------------------------------------------------------------
# Frame extraction + detection
# ---------------------------------------------------------------------------

def run_detection(video_name: str, video_path: Path, model) -> list:
    """
    Extract frames at FRAME_RATE fps and run RT-DETR-L inference on each.
    Returns list of per-frame dicts matching the detections.json schema used
    by the existing inspector scripts.
    """
    out_dir = OUTPUT_BASE / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps_native
    frame_step   = max(1, int(round(fps_native / FRAME_RATE)))
    n_expected   = total_frames // frame_step

    logger.info("=" * 62)
    logger.info("Video     : %s", video_path.name)
    logger.info("Native FPS: %.2f  |  Duration: %.1f s  |  Total frames: %d",
                fps_native, duration_s, total_frames)
    logger.info("Sampling  : 1 frame / %d native frames  (~%.1f fps)",
                frame_step, FRAME_RATE)
    logger.info("Expected  : ~%d frames to process", n_expected)
    logger.info("Checkpoint: %s", CHECKPOINT)
    logger.info("=" * 62)

    frames_data  = []
    frame_idx_raw = 0
    sampled_idx   = 0
    t_start       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_raw % frame_step == 0:
            timestamp_s = round(frame_idx_raw / fps_native, 3)
            lighting    = classify_lighting(frame)

            # RT-DETR inference (Zhao et al., 2024)
            results = model.predict(
                source  = frame,
                conf    = CONF_THRESH,
                iou     = IOU_THRESH,
                imgsz   = IMGSZ,
                verbose = False,
            )

            boxes_data = []
            if results and results[0].boxes is not None:
                res = results[0]
                for box in res.boxes:
                    cls_id   = int(box.cls.item())
                    cls_name = (CLASS_NAMES[cls_id]
                                if cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
                    conf     = round(float(box.conf.item()), 4)
                    xyxy     = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    xywhn    = [round(float(v), 4) for v in box.xywhn[0].tolist()]
                    boxes_data.append({
                        "class_id":    cls_id,
                        "class_name":  cls_name,
                        "confidence":  conf,
                        "bbox_xyxy":   xyxy,
                        "bbox_xywhn":  xywhn,
                    })

            frames_data.append({
                "frame_index":  sampled_idx,
                "frame_raw":    frame_idx_raw,
                "timestamp_s":  timestamp_s,
                "lighting":     lighting,
                "n_detections": len(boxes_data),
                "boxes":        boxes_data,
                # frame_path filled later during bbox export
                "frame_path":   "",
            })

            sampled_idx += 1

            if sampled_idx % 200 == 0:
                elapsed = time.time() - t_start
                rate    = sampled_idx / elapsed
                eta     = (n_expected - sampled_idx) / max(rate, 1e-6)
                logger.info(
                    "  [%s] %d / ~%d frames  |  %.1f fps  |  ETA %.0f s",
                    video_name, sampled_idx, n_expected, rate, eta,
                )

        frame_idx_raw += 1

    cap.release()

    elapsed = time.time() - t_start
    n_det   = sum(1 for r in frames_data if r["n_detections"] > 0)
    logger.info(
        "[%s] Done — %d frames in %.1f s  |  %d with detections (%.1f%%)",
        video_name, len(frames_data), elapsed,
        n_det, 100 * n_det / max(1, len(frames_data)),
    )

    # Save detections.json
    json_path = out_dir / "detections.json"
    json_path.write_text(
        json.dumps(frames_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved detections.json → %s", json_path)

    return frames_data


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(video_name: str, frames: list, out_dir: Path) -> dict:
    frames_with  = [r for r in frames if r["n_detections"] > 0]
    total_boxes  = sum(r["n_detections"] for r in frames)
    class_counts = {}
    class_confs  = {}
    for r in frames:
        for b in r["boxes"]:
            cn = b["class_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1
            class_confs.setdefault(cn, []).append(b["confidence"])

    all_confs = [b["confidence"] for r in frames for b in r["boxes"]]

    title = video_name.upper()
    lines = [
        "=" * 62,
        f"  RIDS N-RDD2024 Validation — {title}",
        f"  Checkpoint: runs/detect/nrdd_2024/best.pt (epoch 47)",
        f"  Model     : RT-DETR-L, N-RDD2024 10-class fine-tune",
        f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 62,
        f"  Frames processed         : {len(frames)}",
        f"  Frames with detections   : {len(frames_with)}"
        f" ({100*len(frames_with)/max(1,len(frames)):.1f}%)",
        f"  Frames with no detection : {len(frames) - len(frames_with)}",
        f"  Total bounding boxes     : {total_boxes}",
        f"  Confidence threshold     : {CONF_THRESH}",
        "  Class breakdown (N-RDD2024 taxonomy):",
    ]
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        if count > 0:
            mc = sum(class_confs[cls]) / len(class_confs[cls])
            lines.append(f"    {cls:<32}: {count:4d}  (mean conf {mc:.3f})")
        else:
            lines.append(f"    {cls:<32}:    0")

    if all_confs:
        lines += [
            "  Confidence (all classes):",
            f"    min / mean / max : {min(all_confs):.3f} / "
            f"{sum(all_confs)/len(all_confs):.3f} / {max(all_confs):.3f}",
        ]
    lines.append("=" * 62)

    out_path = out_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved summary.txt → %s", out_path)
    print("\n" + "\n".join(lines))

    return {
        "n_frames":       len(frames),
        "n_with_det":     len(frames_with),
        "total_boxes":    total_boxes,
        "class_counts":   class_counts,
        "class_confs":    class_confs,
        "all_confs":      all_confs,
        "det_rate":       len(frames_with) / max(1, len(frames)),
    }


# ---------------------------------------------------------------------------
# Comparison summary (Tokyo vs Cluj)
# ---------------------------------------------------------------------------

def write_comparison(summaries: dict, out_dir: Path):
    lines = [
        "=" * 72,
        "  RIDS N-RDD2024 — Tokyo vs Cluj Comparison",
        f"  Checkpoint: runs/detect/nrdd_2024/best.pt (epoch 47)",
        f"  Conf threshold: {CONF_THRESH}",
        "=" * 72,
        f"  {'Metric':<35} {'Tokyo':>10} {'Cluj':>10}",
        f"  {'-' * 57}",
    ]

    def fmt(s, key, as_pct=False):
        v = s.get(key, 0)
        return f"{v*100:.1f}%" if as_pct else str(v)

    tok = summaries.get("tokyo", {})
    clj = summaries.get("cluj",  {})

    rows = [
        ("Frames processed",          "n_frames",   False),
        ("Frames with detections",    "n_with_det", False),
        ("Detection rate",            "det_rate",   True),
        ("Total bounding boxes",      "total_boxes",False),
    ]
    for label, key, pct in rows:
        tv = fmt(tok, key, pct)
        cv = fmt(clj, key, pct)
        lines.append(f"  {label:<35} {tv:>10} {cv:>10}")

    lines.append(f"  {'-' * 57}")
    lines.append("  Per-class detections:")
    for cls in CLASS_NAMES:
        tc = tok.get("class_counts", {}).get(cls, 0)
        cc = clj.get("class_counts", {}).get(cls, 0)
        if tc > 0 or cc > 0:
            lines.append(f"    {cls:<32} {tc:>10} {cc:>10}")

    lines.append(f"  {'-' * 57}")
    lines.append("  Mean confidence per class:")
    for cls in CLASS_NAMES:
        tc_confs = tok.get("class_confs", {}).get(cls, [])
        cc_confs = clj.get("class_confs", {}).get(cls, [])
        tm = f"{sum(tc_confs)/len(tc_confs):.3f}" if tc_confs else "  —  "
        cm = f"{sum(cc_confs)/len(cc_confs):.3f}" if cc_confs else "  —  "
        if tc_confs or cc_confs:
            lines.append(f"    {cls:<32} {tm:>10} {cm:>10}")

    lines.append("=" * 72)

    out_path = out_dir / "summary_comparison.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved summary_comparison.txt → %s", out_path)
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Plots — identical logic to inspect_detector_tokyo.py
# ---------------------------------------------------------------------------

def plot_class_distribution(video_name: str, class_counts: dict, out_dir: Path):
    classes = [c for c in CLASS_NAMES if class_counts.get(c, 0) > 0]
    if not classes:
        logger.warning("[%s] No detections — skipping class distribution plot", video_name)
        return
    counts = [class_counts[c] for c in classes]
    colors = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(classes)), counts, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_ylabel("Detection count")
    ax.set_title(
        f"{video_name.upper()} — Class distribution\n"
        f"(RT-DETR-L N-RDD2024, conf≥{CONF_THRESH})",
        fontweight="bold",
    )
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_confidence_distribution(video_name: str, frames: list, out_dir: Path):
    class_confs = {}
    for r in frames:
        for b in r["boxes"]:
            class_confs.setdefault(b["class_name"], []).append(b["confidence"])
    if not class_confs:
        logger.warning("[%s] No detections — skipping confidence plot", video_name)
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(CONF_THRESH - 0.05, 1.0, 35)
    for cls, confs in class_confs.items():
        ax.hist(confs, bins=bins, alpha=0.6,
                color=CLASS_COLORS.get(cls, DEFAULT_COLOR),
                label=cls.replace("_", " "), edgecolor="white", linewidth=0.4)
    ax.axvline(CONF_THRESH, color="black", linestyle="--",
               linewidth=1.2, label=f"threshold ({CONF_THRESH})")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{video_name.upper()} — Confidence distribution by class",
        fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = out_dir / "confidence_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_rate_timeline(video_name: str, frames: list, out_dir: Path):
    if not frames:
        return
    bin_size_s = 30
    max_t  = max(r["timestamp_s"] for r in frames)
    n_bins = max(1, int(max_t / bin_size_s) + 1)
    counts = [0] * n_bins
    for r in frames:
        if r["n_detections"] > 0:
            idx = min(int(r["timestamp_s"] / bin_size_s), n_bins - 1)
            counts[idx] += r["n_detections"]
    t_axis = [i * bin_size_s / 60 for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.bar(t_axis, counts, width=bin_size_s / 60 * 0.9,
           color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Detections per 30 s bin")
    ax.set_title(
        f"{video_name.upper()} — Detection rate over time",
        fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "detection_rate_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_spatial_heatmap(video_name: str, frames: list, out_dir: Path):
    xs, ys = [], []
    for r in frames:
        for b in r["boxes"]:
            xs.append(b["bbox_xywhn"][0])
            ys.append(b["bbox_xywhn"][1])
    if not xs:
        logger.warning("[%s] No detections — skipping spatial heatmap", video_name)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    h, xedges, yedges = np.histogram2d(xs, ys, bins=20, range=[[0, 1], [0, 1]])
    im = ax.imshow(h.T, origin="upper", extent=[0, 1, 0, 1],
                   cmap="hot", aspect="auto", interpolation="bilinear")
    ax.set_xlabel("Normalised x (left → right)")
    ax.set_ylabel("Normalised y (top → bottom)")
    ax.set_title(
        f"{video_name.upper()} — Spatial heatmap of bbox centres\n"
        "(bottom = road surface, expected for true positives)",
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Detection count")
    plt.tight_layout()
    out = out_dir / "spatial_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_grid(video_name: str, frames: list, out_dir: Path, n: int = 20):
    det_frames = [r for r in frames if r["n_detections"] > 0]
    if not det_frames:
        logger.warning("[%s] No detection frames — skipping grid", video_name)
        return
    step    = max(1, len(det_frames) // n)
    sampled = det_frames[::step][:n]
    cols    = 5
    rows    = math.ceil(len(sampled) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.axis("off")

    for i, fr in enumerate(sampled):
        img = cv2.imread(fr["frame_path"])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w    = img.shape[:2]
        for b in fr["boxes"]:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            hex_c     = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
            color_rgb = tuple(int(hex_c[j:j+2], 16) / 255 for j in (0, 2, 4))
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

    plt.suptitle(
        f"{video_name.upper()} — Detection grid (sampled, N-RDD2024 model)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "detection_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_all_frames_grid(video_name: str, frames: list, out_dir: Path):
    n    = len(frames)
    cols = 80
    rows = math.ceil(n / cols)
    cell = 8
    img  = np.ones((rows * cell, cols * cell, 3), dtype=np.uint8) * 60

    for idx, r in enumerate(frames):
        row = idx // cols
        col = idx % cols
        y0, y1 = row * cell, (row + 1) * cell
        x0, x1 = col * cell, (col + 1) * cell
        if r["n_detections"] > 0:
            dominant = max(r["boxes"], key=lambda b: b["confidence"])["class_name"]
            hex_c    = CLASS_COLORS.get(dominant, DEFAULT_COLOR).lstrip("#")
            color    = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
            img[y0:y1, x0:x1] = color
        else:
            img[y0:y1, x0:x1] = 35

    out = out_dir / "all_frames_grid.png"
    cv2.imwrite(str(out), img)
    logger.info("Saved: %s  (%d frames, %d×%d cells)", out, n, rows, cols)


# ---------------------------------------------------------------------------
# Bounding box export
# ---------------------------------------------------------------------------

def export_bounding_boxes(
    video_name: str,
    frames: list,
    scale: float = EXPORT_SCALE,
) -> list:
    """
    Export one JPEG per detection frame with bboxes drawn.
    Saved to data/validation_nrdd_2024/bounding_boxes/<video_name>/
    Updates frame_path in frames list in-place.
    """
    out_dir = BBOX_BASE / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    font_thickness = max(1, int(scale))
    det_frames     = [r for r in frames if r["n_detections"] > 0]
    n_written      = 0

    # We need to re-read from video since we didn't save raw frames
    video_path = VIDEOS[video_name]
    cap        = cv2.VideoCapture(str(video_path))
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(fps_native / FRAME_RATE)))

    # Build lookup: sampled_idx → frame data
    det_lookup = {r["frame_index"]: r for r in det_frames}
    sampled_idx   = 0
    frame_idx_raw = 0

    logger.info("[%s] Exporting %d annotated frames → %s",
                video_name, len(det_frames), out_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_raw % frame_step == 0:
            if sampled_idx in det_lookup:
                fr  = det_lookup[sampled_idx]
                img = frame.copy()

                if scale != 1.0:
                    h, w = img.shape[:2]
                    img  = cv2.resize(img, (int(w * scale), int(h * scale)),
                                      interpolation=cv2.INTER_LINEAR)
                h_img, w_img = img.shape[:2]

                # Draw boxes
                for b in fr["boxes"]:
                    x1, y1, x2, y2 = [int(v * scale) for v in b["bbox_xyxy"]]
                    hex_c     = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
                    color_bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
                    cv2.rectangle(img, (x1, y1), (x2, y2),
                                  color_bgr, max(1, int(2 * scale)))
                    label = f"{b['class_name'].replace('_', ' ')} {b['confidence']:.2f}"
                    lw, lh = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, font_thickness
                    )[0]
                    cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 4, y1),
                                  color_bgr, -1)
                    cv2.putText(img, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                                (255, 255, 255), font_thickness, cv2.LINE_AA)

                # Top info banner
                n_det  = fr["n_detections"]
                banner = (f"#{fr['frame_index']}  t={fr['timestamp_s']}s  "
                          f"{fr['lighting']}  |  {n_det} detection(s)"
                          f"  [{video_name.upper()}]  [N-RDD2024 model]")
                bh = int(30 * scale)
                cv2.rectangle(img, (0, 0), (w_img, bh), (20, 20, 20), -1)
                cv2.putText(img, banner, (int(8 * scale), int(20 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                            (220, 220, 220), font_thickness, cv2.LINE_AA)

                # Detection list overlay
                for i, b in enumerate(fr["boxes"]):
                    hex_c     = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
                    color_bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
                    line = (f"[{i+1}] {b['class_name'].replace('_', ' ')} "
                            f"{b['confidence']:.2f}")
                    ly = bh + 8 + i * int(22 * scale)
                    cv2.putText(img, line, (int(8 * scale), ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                                color_bgr, font_thickness, cv2.LINE_AA)

                # Filename
                det_names = "_".join(sorted(set(
                    b["class_name"]
                    .replace("longitudinal_crack",       "lng")
                    .replace("transverse_crack",         "trs")
                    .replace("alligator_crack",          "alg")
                    .replace("repaired_crack",           "rep")
                    .replace("pothole",                  "pot")
                    .replace("pedestrian_crossing_blur", "ped")
                    .replace("lane_line_blur",           "lan")
                    .replace("manhole_cover",            "man")
                    .replace("patchy_road",              "pat")
                    .replace("rutting",                  "rut")
                    for b in fr["boxes"]
                )))
                fname = (f"frame_{fr['frame_index']:06d}"
                         f"_t{fr['timestamp_s']:.0f}s"
                         f"_{n_det}det"
                         f"_{det_names}.jpg")
                fpath = out_dir / fname
                cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

                # Update frame_path in-place for detection_grid
                fr["frame_path"] = str(fpath)
                n_written += 1

            sampled_idx += 1
        frame_idx_raw += 1

    cap.release()
    logger.info("[%s] Exported %d annotated frames → %s", video_name, n_written, out_dir)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CONF_THRESH  # must be declared before any use in this scope

    parser = argparse.ArgumentParser(
        description="Validate N-RDD2024 RT-DETR-L on Tokyo and Cluj dashcam videos"
    )
    parser.add_argument("--videos", nargs="+", choices=["tokyo", "cluj"],
                        default=["tokyo", "cluj"],
                        help="Which videos to process (default: both)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — use Agg backend")
    parser.add_argument("--skip-bbox-export", action="store_true",
                        help="Skip bounding box frame export (faster, less disk)")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help=f"Confidence threshold (default: {CONF_THRESH})")
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    CONF_THRESH = args.conf

    # ── Verify checkpoint ────────────────────────────────────────────────────
    if not CHECKPOINT.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT)
        logger.error("Expected: runs/detect/nrdd_2024/best.pt")
        sys.exit(1)
    logger.info("Checkpoint: %s", CHECKPOINT.resolve())

    # ── Load model once — shared across both videos ──────────────────────────
    logger.info("Loading RT-DETR-L from %s ...", CHECKPOINT)
    from ultralytics import RTDETR
    model = RTDETR(str(CHECKPOINT))
    logger.info("Model loaded.")

    # ── Process each video ───────────────────────────────────────────────────
    summaries = {}
    for video_name in args.videos:
        video_path = VIDEOS[video_name]
        out_dir    = OUTPUT_BASE / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n%s\nProcessing: %s\n%s", "=" * 62, video_name.upper(), "=" * 62)

        # Detection pass
        frames = run_detection(video_name, video_path, model)

        # Bounding box export (fills frame_path in-place)
        if not args.skip_bbox_export:
            frames = export_bounding_boxes(video_name, frames, scale=EXPORT_SCALE)
        else:
            logger.info("[%s] Skipping bbox export (--skip-bbox-export)", video_name)

        # Re-save detections.json with updated frame_paths
        json_path = out_dir / "detections.json"
        json_path.write_text(
            json.dumps(frames, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Summary + plots
        summary = write_summary(video_name, frames, out_dir)
        summaries[video_name] = summary

        plot_class_distribution(video_name, summary["class_counts"], out_dir)
        plot_confidence_distribution(video_name, frames, out_dir)
        plot_detection_rate_timeline(video_name, frames, out_dir)
        plot_spatial_heatmap(video_name, frames, out_dir)
        plot_detection_grid(video_name, frames, out_dir, n=20)
        plot_all_frames_grid(video_name, frames, out_dir)

    # ── Comparison (only if both videos were processed) ──────────────────────
    if len(summaries) == 2:
        write_comparison(summaries, OUTPUT_BASE)

    logger.info("\nAll outputs saved to: %s", OUTPUT_BASE.resolve())
    logger.info("Bounding boxes:       %s", BBOX_BASE.resolve())
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()