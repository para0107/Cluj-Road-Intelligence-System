"""
validate_chain_finetune.py
==========================
Runs the chain fine-tuned RT-DETR-L checkpoint
(COCO -> RDD2022 -> N-RDD2024, best.pt from Phase 2)
on the pre-extracted Cluj-Napoca dashcam frames and
produces a full inspection report, mirroring validate_nrdd2024.py.

This is Run 3 in the three-checkpoint ablation study.
Ref:
  RT-DETR   — Zhao et al. (2024) https://arxiv.org/abs/2304.08069
  N-RDD2024 — Kaya & Codur (2024) doi:10.17632/27c8pwsd6v.3
  Chain run — COCO -> RDD2022 -> N-RDD2024, ep48/49, mAP50=0.419 val

Frames (Cluj only):
  data/processed/frames/cluj/
  manifest read from data/processed/frames/cluj/manifest.json

Outputs:
  Stats / plots  ->  data/validation_chain_fine_tune/cluj/
  Bbox frames    ->  data/validation_chain_fine_tune/bounding_boxes/
  Log            ->  data/validation_chain_fine_tune/validate_chain.log

Run:
  python validate_chain_finetune.py
  python validate_chain_finetune.py --no-display           # headless / SSH
  python validate_chain_finetune.py --skip-bbox-export     # faster, less disk
  python validate_chain_finetune.py --conf 0.5             # higher threshold

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
# Paths — absolute Windows paths; Path() normalises separators automatically
# ---------------------------------------------------------------------------
CHECKPOINT = Path(
    r"C:\Facultate\pothole-detection\Pothole-Detection"
    r"\runs\detect\chain_fine_tune\Phase2\best.pt"
)

CLUJ_FRAME_DIR = Path(
    r"C:\Facultate\pothole-detection\Pothole-Detection"
    r"\data\processed\frames\cluj"
)
CLUJ_MANIFEST = CLUJ_FRAME_DIR / "manifest.json"

OUTPUT_DIR = Path(
    r"C:\Facultate\pothole-detection\Pothole-Detection"
    r"\data\validation_chain_fine_tune\cluj"
)
BBOX_DIR = Path(
    r"C:\Facultate\pothole-detection\Pothole-Detection"
    r"\data\validation_chain_fine_tune\bounding_boxes"
)
LOG_PATH = OUTPUT_DIR.parent / "validate_chain.log"

# ---------------------------------------------------------------------------
# Detection config — mirrors validate_nrdd2024.py exactly
# ---------------------------------------------------------------------------
CONF_THRESH  = 0.35   # operational RIDS threshold
IOU_THRESH   = 0.6
IMGSZ        = 640
EXPORT_SCALE = 1.5    # resize factor for annotated frame export
EXTRACT_FPS  = 2.0    # assumed frame rate if manifest is absent

# ---------------------------------------------------------------------------
# N-RDD2024 ten-class taxonomy
# The chain checkpoint outputs 10 classes (head rebuilt 4->10 by Ultralytics).
# Ref: Kaya & Codur (2024) doi:10.17632/27c8pwsd6v.3
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
    "longitudinal_crack":        "#e74c3c",
    "transverse_crack":          "#e67e22",
    "alligator_crack":           "#9b59b6",
    "repaired_crack":            "#1abc9c",
    "pothole":                   "#3498db",
    "pedestrian_crossing_blur":  "#f39c12",
    "lane_line_blur":            "#d35400",
    "manhole_cover":             "#7f8c8d",
    "patchy_road":               "#27ae60",
    "rutting":                   "#8e44ad",
}
DEFAULT_COLOR = "#95a5a6"


# ---------------------------------------------------------------------------
# Logging — file + stdout
# ---------------------------------------------------------------------------
def _setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BBOX_DIR.mkdir(parents=True, exist_ok=True)
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
# Lighting classification (mirrors preprocessor.py / validate_nrdd2024.py)
# ---------------------------------------------------------------------------
def classify_lighting(frame: np.ndarray) -> str:
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(gray.mean())
    if mean_lum < 60:
        return "low_light"
    elif mean_lum < 160:
        return "overcast"
    return "daylight"


# ---------------------------------------------------------------------------
# Frame list loader
# ---------------------------------------------------------------------------
def load_frame_list() -> list:
    """
    Load Cluj frame metadata from manifest.json if it exists,
    otherwise fall back to globbing frame_*.jpg.
    Mirrors load_frame_list() in validate_nrdd2024.py exactly.
    """
    if not CLUJ_FRAME_DIR.exists():
        logger.error("Frame directory not found: %s", CLUJ_FRAME_DIR)
        sys.exit(1)

    if CLUJ_MANIFEST.exists():
        logger.info("Loading manifest: %s", CLUJ_MANIFEST)
        with open(CLUJ_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)

        # The manifest stores Windows-style relative paths
        # (e.g. "data\processed\frames\cluj\frame_000000_t0.000.jpg").
        # On Windows, Path() parses these correctly but resolves them
        # against the CWD which may be wrong.
        # Solution: use PureWindowsPath to reliably extract just the
        # filename, then join with the absolute CLUJ_FRAME_DIR.
        from pathlib import PureWindowsPath
        for entry in manifest:
            raw = entry["frame_path"]
            fname = PureWindowsPath(raw).name   # "frame_000000_t0.000.jpg"
            entry["frame_path"] = str(CLUJ_FRAME_DIR / fname)

        logger.info("Manifest: %d entries", len(manifest))
        logger.info("First frame path: %s", manifest[0]["frame_path"])
        return manifest

    # Fallback — no manifest present
    logger.warning("manifest.json not found — falling back to glob")
    jpgs = sorted(CLUJ_FRAME_DIR.glob("frame_*.jpg"))
    if not jpgs:
        logger.error("No frame_*.jpg files found in %s", CLUJ_FRAME_DIR)
        sys.exit(1)

    logger.info("Found %d frames via glob (no manifest)", len(jpgs))
    return [
        {
            "frame_path":    str(jpg.resolve()),  # absolute path
            "frame_index":   idx,
            "timestamp_s":   round(idx / EXTRACT_FPS, 3),
            "lighting":      None,
            "latitude":      None,
            "longitude":     None,
            "sun_elevation": None,
        }
        for idx, jpg in enumerate(jpgs)
    ]


# ---------------------------------------------------------------------------
# Detection pass
# ---------------------------------------------------------------------------
def run_detection(model) -> list:
    """
    Run the chain fine-tune checkpoint on all Cluj frames.
    Returns a list of per-frame dicts with detection results.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_list = load_frame_list()
    n_total    = len(frame_list)

    logger.info("=" * 62)
    logger.info("Video       : CLUJ  (chain fine-tune validation)")
    logger.info("Frames      : %d pre-extracted JPEGs", n_total)
    logger.info("Checkpoint  : %s", CHECKPOINT)
    logger.info("conf=%.2f  iou=%.2f  imgsz=%d", CONF_THRESH, IOU_THRESH, IMGSZ)
    logger.info("Model       : RT-DETR-L  COCO->RDD2022->N-RDD2024  ep48/49")
    logger.info("mAP50 val   : 0.4186  (best.pt)")
    logger.info("=" * 62)

    frames_data = []
    t_start     = time.time()

    for i, meta in enumerate(frame_list):
        frame_path = Path(meta["frame_path"])

        if not frame_path.exists():
            logger.warning("Frame not found, skipping: %s", frame_path)
            continue

        # Use manifest lighting if present, else classify live from pixels
        lighting = meta.get("lighting") or None
        if lighting is None:
            img_bgr  = cv2.imread(str(frame_path))
            if img_bgr is not None:
                lighting = classify_lighting(img_bgr)
            else:
                lighting = "unknown"

        # RT-DETR inference (Zhao et al., 2024)
        results = model.predict(
            source  = str(frame_path),
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            imgsz   = IMGSZ,
            verbose = False,
        )

        boxes_data = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id   = int(box.cls.item())
                cls_name = (CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
                conf     = round(float(box.conf.item()), 4)
                xyxy     = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                xywhn    = [round(float(v), 4) for v in box.xywhn[0].tolist()]
                boxes_data.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_xyxy":  xyxy,
                    "bbox_xywhn": xywhn,
                })

        frames_data.append({
            "frame_index":   meta.get("frame_index", i),
            "frame_path":    str(frame_path),
            "timestamp_s":   meta.get("timestamp_s", round(i / EXTRACT_FPS, 3)),
            "lighting":      lighting,
            "latitude":      meta.get("latitude"),
            "longitude":     meta.get("longitude"),
            "sun_elevation": meta.get("sun_elevation"),
            "n_detections":  len(boxes_data),
            "boxes":         boxes_data,
        })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (n_total - i - 1) / max(rate, 1e-6)
            n_det   = sum(1 for r in frames_data if r["n_detections"] > 0)
            logger.info(
                "  [cluj] %d / %d  |  %.1f fps  |  ETA %.0f s  |  "
                "%d frames with detections",
                i + 1, n_total, rate, eta, n_det,
            )

    elapsed = time.time() - t_start
    n_det   = sum(1 for r in frames_data if r["n_detections"] > 0)
    logger.info(
        "[cluj] Done — %d frames in %.1f s  |  %d with detections (%.1f%%)",
        len(frames_data), elapsed,
        n_det, 100 * n_det / max(1, len(frames_data)),
    )

    json_path = OUTPUT_DIR / "detections.json"
    json_path.write_text(
        json.dumps(frames_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved detections.json -> %s", json_path)
    return frames_data


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(frames: list) -> dict:
    frames_with  = [r for r in frames if r["n_detections"] > 0]
    total_boxes  = sum(r["n_detections"] for r in frames)
    class_counts: dict = {}
    class_confs:  dict = {}
    for r in frames:
        for b in r["boxes"]:
            cn = b["class_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1
            class_confs.setdefault(cn, []).append(b["confidence"])

    all_confs = [b["confidence"] for r in frames for b in r["boxes"]]

    lines = [
        "=" * 62,
        "  RIDS Chain Fine-Tune Validation — CLUJ",
        f"  Checkpoint : {CHECKPOINT}",
        "  Model      : RT-DETR-L  COCO->RDD2022->N-RDD2024",
        "  mAP50 val  : 0.4186  (best.pt, ep48/49)",
        f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 62,
        f"  Frames processed         : {len(frames)}",
        (
            f"  Frames with detections   : {len(frames_with)}"
            f" ({100 * len(frames_with) / max(1, len(frames)):.1f}%)"
        ),
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
            (
                f"    min / mean / max : {min(all_confs):.3f} / "
                f"{sum(all_confs) / len(all_confs):.3f} / {max(all_confs):.3f}"
            ),
        ]
    lines.append("=" * 62)

    out_path = OUTPUT_DIR / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved summary.txt -> %s", out_path)
    print("\n" + "\n".join(lines))

    return {
        "n_frames":     len(frames),
        "n_with_det":   len(frames_with),
        "total_boxes":  total_boxes,
        "class_counts": class_counts,
        "class_confs":  class_confs,
        "all_confs":    all_confs,
        "det_rate":     len(frames_with) / max(1, len(frames)),
    }


# ---------------------------------------------------------------------------
# Plots — identical logic to validate_nrdd2024.py
# ---------------------------------------------------------------------------
def plot_class_distribution(class_counts: dict):
    classes = [c for c in CLASS_NAMES if class_counts.get(c, 0) > 0]
    if not classes:
        logger.warning("No detections — skipping class distribution plot")
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
        f"CLUJ — Class distribution\n"
        f"(RT-DETR-L chain COCO\u2192RDD2022\u2192N-RDD2024, conf\u2265{CONF_THRESH})",
        fontweight="bold",
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count), ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    out = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_confidence_distribution(frames: list):
    class_confs: dict = {}
    for r in frames:
        for b in r["boxes"]:
            class_confs.setdefault(b["class_name"], []).append(b["confidence"])
    if not class_confs:
        logger.warning("No detections — skipping confidence distribution plot")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(CONF_THRESH - 0.05, 1.0, 35)
    for cls, confs in class_confs.items():
        ax.hist(
            confs, bins=bins, alpha=0.6,
            color=CLASS_COLORS.get(cls, DEFAULT_COLOR),
            label=cls.replace("_", " "), edgecolor="white", linewidth=0.4,
        )
    ax.axvline(CONF_THRESH, color="black", linestyle="--",
               linewidth=1.2, label=f"threshold ({CONF_THRESH})")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title(
        "CLUJ — Confidence distribution by class\n"
        "(RT-DETR-L chain fine-tune)",
        fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = OUTPUT_DIR / "confidence_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_rate_timeline(frames: list):
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
        "CLUJ — Detection rate over time  (chain fine-tune)",
        fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "detection_rate_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_spatial_heatmap(frames: list):
    xs, ys = [], []
    for r in frames:
        for b in r["boxes"]:
            xs.append(b["bbox_xywhn"][0])
            ys.append(b["bbox_xywhn"][1])
    if not xs:
        logger.warning("No detections — skipping spatial heatmap")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    h, xedges, yedges = np.histogram2d(xs, ys, bins=20, range=[[0, 1], [0, 1]])
    im = ax.imshow(
        h.T, origin="upper", extent=[0, 1, 0, 1],
        cmap="hot", aspect="auto", interpolation="bilinear",
    )
    ax.set_xlabel("Normalised x (left \u2192 right)")
    ax.set_ylabel("Normalised y (top \u2192 bottom)")
    ax.set_title(
        "CLUJ — Spatial heatmap of bbox centres\n"
        "(bottom = road surface, expected for true positives)",
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Detection count")
    plt.tight_layout()
    out = OUTPUT_DIR / "spatial_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_detection_grid(frames: list, n: int = 20):
    det_frames = [r for r in frames if r["n_detections"] > 0]
    if not det_frames:
        logger.warning("No detection frames — skipping detection grid")
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
            color_rgb = tuple(int(hex_c[j:j + 2], 16) / 255 for j in (0, 2, 4))
            rect = mpatches.FancyBboxPatch(
                (x1 / w, 1 - y2 / h), (x2 - x1) / w, (y2 - y1) / h,
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
        "CLUJ — Detection grid (sampled, chain fine-tune)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "detection_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_all_frames_grid(frames: list):
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

    out = OUTPUT_DIR / "all_frames_grid.png"
    cv2.imwrite(str(out), img)
    logger.info(
        "Saved all_frames_grid: %s  (%d frames, %d\u00d7%d cells)",
        out, n, rows, cols,
    )


# ---------------------------------------------------------------------------
# Bounding box export
# ---------------------------------------------------------------------------
def export_bounding_boxes(frames: list, scale: float = EXPORT_SCALE):
    """
    Write one annotated JPEG per detection frame to BBOX_DIR.
    Reads directly from the pre-extracted frame JPEGs.
    """
    BBOX_DIR.mkdir(parents=True, exist_ok=True)
    font_thickness = max(1, int(scale))
    det_frames     = [r for r in frames if r["n_detections"] > 0]
    n_written      = 0

    logger.info("Exporting %d annotated frames -> %s", len(det_frames), BBOX_DIR)

    for fr in det_frames:
        img = cv2.imread(fr["frame_path"])
        if img is None:
            logger.warning("Cannot read frame: %s", fr["frame_path"])
            continue

        if scale != 1.0:
            h, w = img.shape[:2]
            img  = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        h_img, w_img = img.shape[:2]

        # Draw bounding boxes and labels
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
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA,
            )

        # Top info banner
        lat   = fr.get("latitude")
        lon   = fr.get("longitude")
        gps_s = f" | GPS {lat:.4f},{lon:.4f}" if lat and lon else ""
        banner = (
            f"#{fr['frame_index']}  t={fr['timestamp_s']}s  "
            f"{fr['lighting']}  |  {fr['n_detections']} det"
            f"  [CLUJ]  [CHAIN FINE-TUNE ep48/49]{gps_s}"
        )
        bh = int(30 * scale)
        cv2.rectangle(img, (0, 0), (w_img, bh), (20, 20, 20), -1)
        cv2.putText(
            img, banner, (int(8 * scale), int(20 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
            (220, 220, 220), font_thickness, cv2.LINE_AA,
        )

        # Detection list overlay (left side, below banner)
        for i, b in enumerate(fr["boxes"]):
            hex_c     = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR).lstrip("#")
            color_bgr = (int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16))
            line      = f"[{i + 1}] {b['class_name'].replace('_', ' ')} {b['confidence']:.2f}"
            ly = bh + 8 + i * int(22 * scale)
            cv2.putText(
                img, line, (int(8 * scale), ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                color_bgr, font_thickness, cv2.LINE_AA,
            )

        # Output filename encodes frame index, timestamp, n detections, classes
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
        fname = (
            f"frame_{fr['frame_index']:06d}"
            f"_t{fr['timestamp_s']:.0f}s"
            f"_{fr['n_detections']}det"
            f"_{det_names}.jpg"
        )
        cv2.imwrite(str(BBOX_DIR / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_written += 1

    logger.info("Exported %d annotated frames -> %s", n_written, BBOX_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global CONF_THRESH

    _setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Validate chain fine-tune RT-DETR-L "
            "(COCO->RDD2022->N-RDD2024) on Cluj dashcam frames"
        )
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Headless mode — use Agg backend (for SSH / servers)",
    )
    parser.add_argument(
        "--skip-bbox-export", action="store_true",
        help="Skip bounding box frame export (faster, less disk)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help=f"Confidence threshold (default: {CONF_THRESH})",
    )
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    CONF_THRESH = args.conf

    # ── Verify checkpoint ────────────────────────────────────────────────────
    if not CHECKPOINT.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT)
        sys.exit(1)
    logger.info("Checkpoint : %s", CHECKPOINT.resolve())
    logger.info("Size       : %.1f MB", CHECKPOINT.stat().st_size / 1e6)

    # ── Verify frame directory ───────────────────────────────────────────────
    if not CLUJ_FRAME_DIR.exists():
        logger.error("Frame directory not found: %s", CLUJ_FRAME_DIR)
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading RT-DETR-L from checkpoint ...")
    from ultralytics import RTDETR  # noqa: PLC0415  (local import — Ultralytics optional dep)
    model = RTDETR(str(CHECKPOINT))
    logger.info("Model loaded.")

    # ── Detection ────────────────────────────────────────────────────────────
    frames = run_detection(model)

    # ── Bounding box export ──────────────────────────────────────────────────
    if not args.skip_bbox_export:
        export_bounding_boxes(frames)
    else:
        logger.info("Skipping bbox export (--skip-bbox-export)")

    # Re-save detections.json (frame_paths updated in-place during export)
    (OUTPUT_DIR / "detections.json").write_text(
        json.dumps(frames, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = write_summary(frames)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_class_distribution(summary["class_counts"])
    plot_confidence_distribution(frames)
    plot_detection_rate_timeline(frames)
    plot_spatial_heatmap(frames)
    plot_detection_grid(frames, n=20)
    plot_all_frames_grid(frames)

    logger.info("\nAll outputs saved to : %s", OUTPUT_DIR.resolve())
    logger.info("Bounding boxes       : %s", BBOX_DIR.resolve())
    logger.info("Log                  : %s", LOG_PATH.resolve())
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()