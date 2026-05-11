"""
scripts/generate_kitti_report.py
---------------------------------
Visual report generator for KITTI pipeline test runs.

Reads the intermediate JSON files written by run_kitti_pipeline.py and
produces a self-contained HTML report with embedded matplotlib figures
(base64-encoded PNG) covering:

  1. DBSCAN cluster map       — GPS scatter, retained (green) vs removed (red),
                                 cluster boundaries shown as convex hulls
  2. Before/after class chart — stacked bar: detections per class before and
                                 after DBSCAN deduplication, per drive
  3. Severity distribution    — S1–S5 pie chart and count table, per drive
  4. Detection heatmap        — 2D hexbin density plot of GPS coordinates
  5. Confidence distribution  — per-class violin/box plot of RT-DETR scores
  6. Stage timing breakdown   — horizontal bar chart of elapsed seconds per stage
  7. Summary statistics table — total frames, detections, inserted, updated,
                                 removed by DBSCAN, GPS rate, per drive and total

All figures are generated with matplotlib and embedded as base64 PNG — the
output is a single .html file with no external dependencies.

Data sources (all read from disk — no reprocessing):
  data/processed/sessions/kitti_<drive>/
      01_manifest/manifest.json          → frame count, GPS rate, lighting
      02_detections/detections.json      → raw detection count, class dist
      06_deduplicated/deduplicated.json  → DBSCAN results, before/after counts
      05_severity/severity_estimates.json → S1–S5 distribution
      session.json                       → stage timings, final counts

Usage
-----
    python scripts/generate_kitti_report.py
    python scripts/generate_kitti_report.py --drive 0001   # single drive
    python scripts/generate_kitti_report.py --output reports/kitti_report.html
    python scripts/generate_kitti_report.py --verbose

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kitti_report")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Facultate\pothole-detection\Pothole-Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SESSIONS_DIR = PROJECT_ROOT / "data" / "processed" / "sessions"
REPORTS_DIR  = PROJECT_ROOT / "data" / "reports"

KITTI_DRIVES: List[str] = ["0001", "0002", "0018", "0057"]

# Colour palette — consistent across all charts
CLASS_COLOURS = {
    "longitudinal_crack":        "#e74c3c",
    "transverse_crack":          "#e67e22",
    "alligator_crack":           "#f39c12",
    "repaired_crack":            "#27ae60",
    "pothole":                   "#2980b9",
    "pedestrian_crossing_blur":  "#8e44ad",
    "lane_line_blur":            "#16a085",
    "manhole_cover":             "#2c3e50",
    "patchy_road":               "#d35400",
    "rutting":                   "#c0392b",
}

SEVERITY_COLOURS = {
    "S1": "#27ae60",
    "S2": "#f1c40f",
    "S3": "#e67e22",
    "S4": "#e74c3c",
    "S5": "#8e44ad",
}

SEVERITY_LABELS = {
    "S1": "S1 — Monitor",
    "S2": "S2 — Schedule",
    "S3": "S3 — Priority",
    "S4": "S4 — Urgent",
    "S5": "S5 — Emergency",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file. Returns None if missing, logs a warning."""
    if not path.exists():
        logger.warning("File not found (skipped): %s", path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_drive_data(drive_id: str) -> dict:
    """
    Load all intermediate JSON files for one KITTI drive.
    Returns a dict with keys: session, manifest, detections, dedup, severity.
    Any missing file is stored as None; calling code must handle None gracefully.
    """
    base = SESSIONS_DIR / f"kitti_{drive_id}"
    return {
        "drive_id":   drive_id,
        "session":    _load_json(base / "session.json"),
        "manifest":   _load_json(base / "01_manifest" / "manifest.json"),
        "detections": _load_json(base / "02_detections" / "detections.json"),
        "severity":   _load_json(base / "05_severity" / "severity_estimates.json"),
        "dedup":      _load_json(base / "06_deduplicated" / "deduplicated.json"),
    }


# ---------------------------------------------------------------------------
# Figure helpers — each returns a base64-encoded PNG string
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _style_fig(fig: plt.Figure, ax: plt.Axes, title: str) -> None:
    """Apply consistent styling to a figure."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f5f5f5")


# ---------------------------------------------------------------------------
# Figure 1 — DBSCAN GPS scatter: retained vs removed, per drive
# ---------------------------------------------------------------------------

def fig_dbscan_scatter(drive_data: dict) -> Optional[str]:
    """
    GPS scatter plot showing:
      - Green circles: retained detections
      - Red crosses:   removed duplicates
      - Cluster ID annotated for multi-detection clusters
    One subplot per drive. Returns base64 PNG or None if no GPS data.
    """
    drives_with_gps = [
        d for d in drive_data
        if d["dedup"] is not None
        and not d["dedup"].get("dbscan_skipped", True)
    ]

    if not drives_with_gps:
        logger.info("DBSCAN scatter: no GPS data available, skipping figure")
        return None

    n = len(drives_with_gps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("DBSCAN Spatial Deduplication — Retained vs Removed",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, d in zip(axes[0], drives_with_gps):
        drive_id = d["drive_id"]
        frames   = d["dedup"].get("frames", [])

        ret_lats, ret_lons = [], []
        rem_lats, rem_lons = [], []
        cluster_centres: Dict[int, Tuple[List, List]] = defaultdict(lambda: ([], []))

        for frame in frames:
            lat = frame.get("latitude")
            lon = frame.get("longitude")
            if lat is None or lon is None:
                continue
            for box in frame.get("boxes", []):
                dedup = box.get("dedup", {})
                cid   = dedup.get("cluster_id", -1)
                csize = dedup.get("cluster_size", 1)
                is_dup = dedup.get("is_duplicate", False)

                if is_dup:
                    rem_lats.append(lat)
                    rem_lons.append(lon)
                else:
                    ret_lats.append(lat)
                    ret_lons.append(lon)

                if csize > 1 and not is_dup:
                    cluster_centres[cid][0].append(lat)
                    cluster_centres[cid][1].append(lon)

        # Plot retained
        if ret_lons:
            ax.scatter(ret_lons, ret_lats, c="#27ae60", s=18,
                       alpha=0.75, label=f"Retained ({len(ret_lats)})",
                       zorder=3, marker="o")

        # Plot removed
        if rem_lons:
            ax.scatter(rem_lons, rem_lats, c="#e74c3c", s=18,
                       alpha=0.75, label=f"Removed ({len(rem_lats)})",
                       zorder=3, marker="x")

        # Annotate cluster centres for clusters with >1 detection
        for cid, (clats, clons) in cluster_centres.items():
            if len(clats) > 1:
                c_lat = np.mean(clats)
                c_lon = np.mean(clons)
                ax.annotate(
                    f"C{cid}\n({len(clats)+1})",
                    xy=(c_lon, c_lat),
                    fontsize=6, ha="center", va="bottom",
                    color="#2c3e50",
                    xytext=(0, 6), textcoords="offset points",
                )

        n_ret = len(ret_lats)
        n_rem = len(rem_lats)
        n_tot = n_ret + n_rem
        pct   = n_rem / max(n_tot, 1) * 100

        ax.set_title(
            f"Drive {drive_id}\n"
            f"Total: {n_tot} | Retained: {n_ret} | Removed: {n_rem} ({pct:.1f}%)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#f5f5f5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Before/after class bar chart
# ---------------------------------------------------------------------------

def fig_class_before_after(drive_data: dict) -> Optional[str]:
    """
    Grouped bar chart: detections per class before and after DBSCAN,
    one group of bars per drive.
    """
    all_classes = set()
    drive_stats = []

    for d in drive_data:
        if d["dedup"] is None:
            continue
        frames   = d["dedup"].get("frames", [])
        before_c: Dict[str, int] = defaultdict(int)
        after_c:  Dict[str, int] = defaultdict(int)

        for frame in frames:
            for box in frame.get("boxes", []):
                cls = box.get("class_name", "unknown")
                all_classes.add(cls)
                is_dup = box.get("dedup", {}).get("is_duplicate", False)
                before_c[cls] += 1
                if not is_dup:
                    after_c[cls] += 1

        drive_stats.append({
            "drive_id": d["drive_id"],
            "before":   dict(before_c),
            "after":    dict(after_c),
        })

    if not drive_stats:
        return None

    classes = sorted(all_classes)
    n_cls   = len(classes)
    n_drv   = len(drive_stats)

    if n_cls == 0:
        return None

    fig, axes = plt.subplots(1, n_drv, figsize=(max(7, 3.5 * n_drv), 5),
                             squeeze=False)
    fig.suptitle("Detections per Class — Before vs After DBSCAN Deduplication",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(n_cls)
    w = 0.38

    for ax, ds in zip(axes[0], drive_stats):
        before_vals = [ds["before"].get(c, 0) for c in classes]
        after_vals  = [ds["after"].get(c, 0)  for c in classes]

        bars_b = ax.bar(x - w / 2, before_vals, w,
                        label="Before", color="#3498db", alpha=0.8)
        bars_a = ax.bar(x + w / 2, after_vals, w,
                        label="After",  color="#27ae60", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("_", "\n") for c in classes],
            fontsize=7, rotation=0,
        )
        ax.set_ylabel("Detections", fontsize=9)
        n_tot  = sum(before_vals)
        n_kept = sum(after_vals)
        n_rem  = n_tot - n_kept
        ax.set_title(
            f"Drive {ds['drive_id']}\n"
            f"Before: {n_tot} → After: {n_kept} (−{n_rem})",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.set_facecolor("#f5f5f5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Value labels on bars
        for bar in list(bars_b) + list(bars_a):
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=6,
                )

    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Severity distribution
# ---------------------------------------------------------------------------

def fig_severity_distribution(drive_data: dict) -> Optional[str]:
    """
    One subplot per drive: horizontal bar chart of S1–S5 counts with
    percentage labels. An additional combined subplot is appended.
    """
    drive_sev = []
    combined: Dict[str, int] = defaultdict(int)

    for d in drive_data:
        if d["severity"] is None:
            continue
        counts: Dict[str, int] = defaultdict(int)
        for frame in d["severity"].get("frames", []):
            for box in frame.get("boxes", []):
                sev = box.get("severity", {})
                if isinstance(sev, dict):
                    lvl = sev.get("severity_level", "S1")
                    counts[lvl] += 1
                    combined[lvl] += 1
        drive_sev.append({"drive_id": d["drive_id"], "counts": dict(counts)})

    if not drive_sev:
        return None

    levels  = ["S1", "S2", "S3", "S4", "S5"]
    n_plots = len(drive_sev) + 1  # +1 for combined
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)
    fig.suptitle("Severity Level Distribution (S1–S5)",
                 fontsize=13, fontweight="bold", y=1.02)

    all_plots = drive_sev + [{"drive_id": "COMBINED", "counts": dict(combined)}]

    for ax, ds in zip(axes[0], all_plots):
        vals    = [ds["counts"].get(l, 0) for l in levels]
        total   = sum(vals)
        colours = [SEVERITY_COLOURS[l] for l in levels]

        bars = ax.barh(levels, vals, color=colours, edgecolor="white",
                       linewidth=0.5)

        for bar, v in zip(bars, vals):
            if v > 0:
                pct = v / max(total, 1) * 100
                ax.text(
                    bar.get_width() + max(total, 1) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v} ({pct:.0f}%)",
                    va="center", fontsize=8,
                )

        label = ("Combined\n(all drives)"
                 if ds["drive_id"] == "COMBINED"
                 else f"Drive {ds['drive_id']}")
        ax.set_title(f"{label}\n(n={total})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Detections", fontsize=9)
        ax.set_xlim(0, max(vals + [1]) * 1.3)
        ax.set_facecolor("#f5f5f5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Detection GPS heatmap (hexbin)
# ---------------------------------------------------------------------------

def fig_gps_heatmap(drive_data: dict) -> Optional[str]:
    """
    2D hexbin density plot of all retained detection GPS coordinates.
    All drives combined into one plot; each drive also gets a subplot.
    """
    drive_coords = []
    all_lats, all_lons = [], []

    for d in drive_data:
        if d["dedup"] is None:
            continue
        lats, lons = [], []
        for frame in d["dedup"].get("frames", []):
            lat = frame.get("latitude")
            lon = frame.get("longitude")
            if lat is None or lon is None:
                continue
            for box in frame.get("boxes", []):
                if not box.get("dedup", {}).get("is_duplicate", False):
                    lats.append(lat)
                    lons.append(lon)
                    all_lats.append(lat)
                    all_lons.append(lon)
        drive_coords.append({"drive_id": d["drive_id"], "lats": lats, "lons": lons})

    valid_drives = [dc for dc in drive_coords if dc["lats"]]
    if not valid_drives or not all_lats:
        logger.info("GPS heatmap: no retained GPS detections, skipping")
        return None

    n = len(valid_drives) + 1  # +1 combined
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle("Retained Detection Density Heatmap (GPS)",
                 fontsize=13, fontweight="bold", y=1.02)

    def _hexbin_plot(ax, lons, lats, title):
        if len(lons) < 3:
            ax.scatter(lons, lats, c="#2980b9", s=20, alpha=0.8)
            ax.set_title(f"{title}\n(n={len(lons)})", fontsize=10, fontweight="bold")
        else:
            gridsize = max(10, min(30, len(lons) // 3))
            hb = ax.hexbin(
                lons, lats,
                gridsize=gridsize,
                cmap="YlOrRd",
                mincnt=1,
                edgecolors="none",
            )
            plt.colorbar(hb, ax=ax, label="Detection count", shrink=0.8)
            ax.set_title(f"{title}\n(n={len(lons)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.set_facecolor("#eeeeee")
        ax.tick_params(labelsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax, dc in zip(axes[0], valid_drives):
        _hexbin_plot(ax, dc["lons"], dc["lats"], f"Drive {dc['drive_id']}")

    # Combined subplot
    _hexbin_plot(axes[0][-1], all_lons, all_lats, "Combined (all drives)")

    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Confidence distribution per class
# ---------------------------------------------------------------------------

def fig_confidence_distribution(drive_data: dict) -> Optional[str]:
    """
    Box plot of RT-DETR confidence scores per class,
    combining all drives. Shows median, IQR, and outliers.
    """
    class_confs: Dict[str, List[float]] = defaultdict(list)

    for d in drive_data:
        if d["detections"] is None:
            continue
        for frame in d["detections"].get("frames", []):
            for box in frame.get("boxes", []):
                cls  = box.get("class_name", "unknown")
                conf = box.get("confidence")
                if conf is not None:
                    class_confs[cls].append(float(conf))

    if not class_confs:
        return None

    # Sort classes by median confidence descending
    classes = sorted(class_confs.keys(),
                     key=lambda c: np.median(class_confs[c]),
                     reverse=True)
    data    = [class_confs[c] for c in classes]
    colours = [CLASS_COLOURS.get(c, "#95a5a6") for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.1), 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        vert=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in classes],
        fontsize=8,
    )
    ax.set_ylabel("Confidence score", fontsize=10)
    ax.set_ylim(0.30, 1.02)
    _style_fig(fig, ax,
               "RT-DETR Confidence Score Distribution per Class (all drives)")

    # Annotate n= per class
    for i, cls in enumerate(classes):
        ax.text(i + 1, 0.315, f"n={len(class_confs[cls])}",
                ha="center", va="bottom", fontsize=7, color="#555")

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Stage timing breakdown
# ---------------------------------------------------------------------------

def fig_stage_timing(drive_data: dict) -> Optional[str]:
    """
    Horizontal stacked bar chart: time spent in each pipeline stage,
    one bar per drive.
    """
    stage_order = [
        "kitti_frame_builder",
        "detector",
        "segmentor",
        "depth_estimator",
        "severity_classifier",
        "deduplicator",
        "db_writer",
    ]
    stage_labels = {
        "kitti_frame_builder": "S1* Frame Builder",
        "detector":            "S2  Detector",
        "segmentor":           "S3  Segmentor (SAM)",
        "depth_estimator":     "S4  Depth Estimator",
        "severity_classifier": "S5  Severity",
        "deduplicator":        "S6  Deduplicator",
        "db_writer":           "S7  DB Writer",
    }
    stage_colours = [
        "#3498db", "#e74c3c", "#8e44ad", "#27ae60",
        "#f39c12", "#1abc9c", "#2c3e50",
    ]

    drive_timings = []
    for d in drive_data:
        sess = d.get("session")
        if sess is None:
            continue
        stages = {s["name"]: s.get("elapsed_s", 0.0)
                  for s in sess.get("stages", [])}
        drive_timings.append({
            "drive_id": d["drive_id"],
            "stages":   stages,
        })

    if not drive_timings:
        return None

    n_drv    = len(drive_timings)
    fig, ax  = plt.subplots(figsize=(10, max(3, n_drv * 1.2)))

    y_pos    = np.arange(n_drv)
    lefts    = np.zeros(n_drv)
    drive_labels = [dt["drive_id"] for dt in drive_timings]

    for stage, colour in zip(stage_order, stage_colours):
        vals = [dt["stages"].get(stage, 0.0) for dt in drive_timings]
        bars = ax.barh(
            y_pos, vals, left=lefts,
            color=colour, label=stage_labels.get(stage, stage),
            edgecolor="white", linewidth=0.4, height=0.6,
        )
        # Label segments that are wide enough to read
        for bar, v, left in zip(bars, vals, lefts):
            if v > 2.0:
                ax.text(
                    left + v / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}s",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold",
                )
        lefts += np.array(vals)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Drive {l}" for l in drive_labels], fontsize=10)
    ax.set_xlabel("Elapsed time (seconds)", fontsize=10)
    ax.legend(
        loc="lower right", fontsize=8,
        ncol=2, framealpha=0.9,
    )
    _style_fig(fig, ax, "Pipeline Stage Timing Breakdown")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Figure 7 — DBSCAN cluster size distribution
# ---------------------------------------------------------------------------

def fig_cluster_size_distribution(drive_data: dict) -> Optional[str]:
    """
    Histogram of DBSCAN cluster sizes (how many detections were in each cluster)
    across all drives combined. Shows what the 2 m radius typically collapses.
    """
    all_sizes: List[int] = []

    for d in drive_data:
        if d["dedup"] is None or d["dedup"].get("dbscan_skipped", True):
            continue
        seen_clusters: Dict[int, int] = {}
        for frame in d["dedup"].get("frames", []):
            for box in frame.get("boxes", []):
                dedup = box.get("dedup", {})
                cid   = dedup.get("cluster_id", -1)
                csize = dedup.get("cluster_size", 1)
                if cid != -1 and cid not in seen_clusters:
                    seen_clusters[cid] = csize
        all_sizes.extend(seen_clusters.values())

    if not all_sizes:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    max_size = max(all_sizes)
    bins     = range(1, max_size + 2)

    ax.hist(all_sizes, bins=bins, color="#3498db", edgecolor="white",
            alpha=0.85, align="left")
    ax.set_xlabel("Cluster size (number of detections)", fontsize=10)
    ax.set_ylabel("Number of clusters", fontsize=10)
    ax.set_xticks(range(1, max_size + 1))

    # Annotate the singleton bar separately
    n_singletons = sum(1 for s in all_sizes if s == 1)
    n_multi      = sum(1 for s in all_sizes if s > 1)
    ax.text(
        0.97, 0.95,
        f"Singletons: {n_singletons}\nMulti-det clusters: {n_multi}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    _style_fig(fig, ax, "DBSCAN Cluster Size Distribution (all drives combined)")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Summary statistics table (HTML)
# ---------------------------------------------------------------------------

def build_stats_table(drive_data: list) -> str:
    """
    Build an HTML table of key statistics per drive plus a totals row.
    All values read from the JSON files — no hardcoded numbers.
    """

    def _count_detections(dedup_data: Optional[dict], retained_only: bool) -> int:
        if dedup_data is None:
            return 0
        total = 0
        for frame in dedup_data.get("frames", []):
            for box in frame.get("boxes", []):
                is_dup = box.get("dedup", {}).get("is_duplicate", False)
                if retained_only and is_dup:
                    continue
                total += 1
        return total

    def _gps_rate(manifest: Optional[list]) -> str:
        if not manifest:
            return "—"
        n_gps = sum(1 for f in manifest if f.get("latitude") is not None)
        pct   = n_gps / max(len(manifest), 1) * 100
        return f"{n_gps}/{len(manifest)} ({pct:.0f}%)"

    def _sev_breakdown(severity: Optional[dict]) -> str:
        if severity is None:
            return "—"
        counts: Dict[str, int] = defaultdict(int)
        for frame in severity.get("frames", []):
            for box in frame.get("boxes", []):
                sev = box.get("severity", {})
                if isinstance(sev, dict):
                    counts[sev.get("severity_level", "S1")] += 1
        if not counts:
            return "—"
        return " ".join(f"{l}:{counts.get(l,0)}" for l in ["S1","S2","S3","S4","S5"])

    rows = []
    totals = {
        "n_frames": 0, "n_det_raw": 0, "n_det_dedup": 0,
        "n_removed": 0, "n_inserted": 0, "n_updated": 0,
    }

    for d in drive_data:
        sess     = d.get("session") or {}
        manifest = d.get("manifest")
        if isinstance(manifest, dict):
            manifest = manifest if "frames" not in manifest else None
        # manifest.json is a list of frame dicts, not wrapped
        manifest_frames = d.get("manifest")
        if isinstance(manifest_frames, dict):
            manifest_frames = manifest_frames.get("frames", [])

        det_raw    = _count_detections(d["dedup"], retained_only=False)
        det_dedup  = _count_detections(d["dedup"], retained_only=True)
        n_removed  = det_raw - det_dedup
        n_inserted = sess.get("n_inserted", 0)
        n_updated  = sess.get("n_updated",  0)
        n_frames   = sess.get("n_frames",   0)

        totals["n_frames"]    += n_frames
        totals["n_det_raw"]   += det_raw
        totals["n_det_dedup"] += det_dedup
        totals["n_removed"]   += n_removed
        totals["n_inserted"]  += n_inserted
        totals["n_updated"]   += n_updated

        pct_rem = n_removed / max(det_raw, 1) * 100
        status  = sess.get("status", "unknown")
        status_colour = (
            "#27ae60" if status == "complete"
            else "#e74c3c" if status == "failed"
            else "#f39c12"
        )

        # Stage timing summary
        stage_times = {
            s["name"]: s.get("elapsed_s", 0)
            for s in sess.get("stages", [])
        }
        total_s = sum(stage_times.values())

        rows.append(f"""
        <tr>
          <td><strong>Drive {d['drive_id']}</strong></td>
          <td style="color:{status_colour};font-weight:bold">{status.upper()}</td>
          <td>{n_frames}</td>
          <td>{_gps_rate(manifest_frames if isinstance(manifest_frames, list) else [])}</td>
          <td>{det_raw}</td>
          <td>{det_dedup}</td>
          <td style="color:#e74c3c">{n_removed} ({pct_rem:.0f}%)</td>
          <td style="color:#27ae60">{n_inserted}</td>
          <td style="color:#f39c12">{n_updated}</td>
          <td style="font-size:11px">{_sev_breakdown(d['severity'])}</td>
          <td>{total_s:.0f} s</td>
        </tr>""")

    pct_rem_tot = totals["n_removed"] / max(totals["n_det_raw"], 1) * 100
    rows.append(f"""
        <tr style="background:#2c3e50;color:white;font-weight:bold">
          <td>TOTAL</td>
          <td>—</td>
          <td>{totals['n_frames']}</td>
          <td>—</td>
          <td>{totals['n_det_raw']}</td>
          <td>{totals['n_det_dedup']}</td>
          <td>{totals['n_removed']} ({pct_rem_tot:.0f}%)</td>
          <td>{totals['n_inserted']}</td>
          <td>{totals['n_updated']}</td>
          <td>—</td>
          <td>—</td>
        </tr>""")

    header = """
    <table>
      <thead>
        <tr>
          <th>Drive</th>
          <th>Status</th>
          <th>Frames</th>
          <th>GPS rate</th>
          <th>Dets (raw)</th>
          <th>Dets (dedup)</th>
          <th>Removed</th>
          <th>DB inserted</th>
          <th>DB updated</th>
          <th>Severity breakdown</th>
          <th>Total time</th>
        </tr>
      </thead>
      <tbody>
    """ + "\n".join(rows) + "\n  </tbody>\n</table>"

    return header


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

CSS = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5;
    margin: 0;
    padding: 0;
    color: #2c3e50;
}
.container {
    max-width: 1300px;
    margin: 0 auto;
    padding: 30px 20px;
}
h1 {
    font-size: 26px;
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 6px;
}
h2 {
    font-size: 17px;
    color: #34495e;
    margin-top: 36px;
    margin-bottom: 10px;
    border-left: 4px solid #3498db;
    padding-left: 10px;
}
.meta {
    font-size: 12px;
    color: #7f8c8d;
    margin-bottom: 20px;
}
.card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
img.figure {
    width: 100%;
    border-radius: 4px;
    margin-top: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}
th {
    background: #2c3e50;
    color: white;
    padding: 8px 12px;
    text-align: left;
}
td {
    padding: 7px 12px;
    border-bottom: 1px solid #ecf0f1;
}
tr:hover td { background: #f8f9fa; }
.note {
    font-size: 11px;
    color: #7f8c8d;
    font-style: italic;
    margin-top: 6px;
}
"""


def build_html_report(
    drive_data: list,
    drives:     List[str],
    generated_at: str,
) -> str:
    """Build the full HTML report string."""

    figures = {}

    logger.info("Generating Figure 1: DBSCAN GPS scatter...")
    figures["dbscan_scatter"]    = fig_dbscan_scatter(drive_data)

    logger.info("Generating Figure 2: class before/after...")
    figures["class_before_after"] = fig_class_before_after(drive_data)

    logger.info("Generating Figure 3: severity distribution...")
    figures["severity_dist"]     = fig_severity_distribution(drive_data)

    logger.info("Generating Figure 4: GPS heatmap...")
    figures["gps_heatmap"]       = fig_gps_heatmap(drive_data)

    logger.info("Generating Figure 5: confidence distribution...")
    figures["conf_dist"]         = fig_confidence_distribution(drive_data)

    logger.info("Generating Figure 6: stage timing...")
    figures["stage_timing"]      = fig_stage_timing(drive_data)

    logger.info("Generating Figure 7: cluster size distribution...")
    figures["cluster_size"]      = fig_cluster_size_distribution(drive_data)

    logger.info("Building statistics table...")
    stats_table = build_stats_table(drive_data)

    def _section(title: str, key: str, note: str = "") -> str:
        b64 = figures.get(key)
        if b64 is None:
            return f"""
            <div class="card">
              <h2>{title}</h2>
              <p class="note">Not available — no GPS data or no runs found for the selected drives.</p>
            </div>"""
        note_html = f'<p class="note">{note}</p>' if note else ""
        return f"""
        <div class="card">
          <h2>{title}</h2>
          {note_html}
          <img class="figure" src="data:image/png;base64,{b64}" alt="{title}">
        </div>"""

    drives_str = ", ".join(drives)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RIDS — KITTI Pipeline Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

  <h1>RIDS — KITTI Pipeline Test Report</h1>
  <p class="meta">
    Generated: {generated_at} &nbsp;|&nbsp;
    Drives: {drives_str} &nbsp;|&nbsp;
    Dataset: KITTI 2011_09_26 image_03 (right colour camera, focal length 721 px)
  </p>

  <div class="card">
    <h2>1. Summary Statistics</h2>
    <p class="note">
      All values read from session.json and intermediate JSON files.
      No values are hardcoded.
    </p>
    {stats_table}
  </div>

  {_section(
      "2. DBSCAN Spatial Deduplication — GPS Map",
      "dbscan_scatter",
      "Green = retained detections. Red × = removed duplicates (within 2 m of a higher-severity "
      "detection of the same class). Cluster IDs and sizes are annotated for multi-detection clusters.",
  )}

  {_section(
      "3. Before vs After Deduplication — Class Breakdown",
      "class_before_after",
      "Blue bars show raw detection count per class; green bars show count after DBSCAN "
      "deduplication. The difference is the number of temporal duplicates collapsed.",
  )}

  {_section(
      "4. DBSCAN Cluster Size Distribution",
      "cluster_size",
      "How many detections were in each DBSCAN cluster before deduplication. "
      "Singletons (size=1) are isolated detections with no spatial neighbour within 2 m. "
      "Multi-detection clusters represent the same physical damage seen in consecutive frames.",
  )}

  {_section(
      "5. Severity Level Distribution (S1–S5)",
      "severity_dist",
      "S1 = Monitor · S2 = Schedule maintenance · S3 = Priority repair · "
      "S4 = Urgent repair · S5 = Emergency closure. "
      "Based on the rule-based multi-signal severity classifier (Stage 5).",
  )}

  {_section(
      "6. Detection Density Heatmap (GPS)",
      "gps_heatmap",
      "Hexbin density of retained detections after DBSCAN. Darker cells indicate "
      "spatial hotspots of road damage along the KITTI survey routes.",
  )}

  {_section(
      "7. RT-DETR Confidence Score Distribution per Class",
      "conf_dist",
      "Box plot of raw RT-DETR confidence scores per class (all drives combined). "
      "Medians above 0.5 indicate reliable detection; classes with wide IQR show "
      "higher variability across road conditions.",
  )}

  {_section(
      "8. Pipeline Stage Timing Breakdown",
      "stage_timing",
      "Elapsed time per stage per drive. Stage 3 (SAM segmentation) typically dominates "
      "on CPU; on CUDA the bottleneck shifts to Stage 2 (RT-DETR, model loading) "
      "and Stage 4 (Monodepth2).",
  )}

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual HTML report for KITTI RIDS pipeline runs."
    )
    parser.add_argument(
        "--drive", default=None,
        choices=KITTI_DRIVES,
        help="Report for a single drive (default: all four drives)",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output HTML path "
            "(default: data/reports/kitti_report_YYYYMMDD_HHMMSS.html)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    drives = [args.drive] if args.drive else KITTI_DRIVES

    # Load data for selected drives
    logger.info("Loading data for drives: %s", drives)
    drive_data = [load_drive_data(d) for d in drives]

    # Check at least one drive has data
    if all(d["session"] is None for d in drive_data):
        logger.error(
            "No session.json found for any of the selected drives. "
            "Run scripts/run_kitti_pipeline.py first."
        )
        sys.exit(1)

    # Build report
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Building HTML report...")
    html = build_html_report(drive_data, drives, generated_at)

    # Write output
    if args.output:
        out_path = Path(args.output)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = REPORTS_DIR / f"kitti_report_{ts}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report saved: %s", out_path)
    print(f"\nReport ready: {out_path}")


if __name__ == "__main__":
    main()