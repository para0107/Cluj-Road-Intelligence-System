"""
scripts/inspect_preprocessor.py
---------------------------------
Inspect and visualise the output of pipeline/preprocessor.py (Stage 1).

What this produces
------------------
Console:
  - Per-field summary table (total frames, GPS coverage, lighting breakdown,
    sun elevation stats, shadow score stats, timestamp span)

Saved to --output-dir (default: data/processed/inspection/preprocessor/):
  1. summary.txt              Plain-text version of the console summary
  2. lighting_distribution.png  Bar chart: daylight / overcast / low_light counts
  3. sun_elevation.png          Line plot of sun elevation over time
  4. shadow_score.png           Line plot of shadow_geometry_score over time
  5. gps_track.png              Scatter plot of the GPS track (lat/lon)
  6. frame_grid.png             Grid of N sampled frames, labelled with
                                lighting class and sun elevation
  7. lighting_timeline.png      Colour-coded timeline strip: each frame is
                                one column, coloured by lighting class

Usage
-----
    python scripts/inspect_preprocessor.py \
        --manifest data/processed/frames/bdd_sample/manifest.json \
        [--frames-dir  data/processed/frames/bdd_sample/]  # optional override
        [--output-dir  data/processed/inspection/preprocessor/]
        [--grid-n      16]   # how many frames to show in the grid
        [--no-display]       # skip plt.show(), only save files (useful on servers)

Dependencies
------------
    pip install matplotlib numpy pillow
    (opencv-python already required by preprocessor)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lighting colour palette (consistent across all plots)
# ---------------------------------------------------------------------------
LIGHTING_COLORS = {
    "daylight":  "#F5A623",   # amber
    "overcast":  "#7B8D9E",   # slate blue-grey
    "low_light": "#2C3E50",   # dark navy
}

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded manifest: %s (%d frames)", path, len(records))
    return records


# ---------------------------------------------------------------------------
# Console / text summary
# ---------------------------------------------------------------------------

def compute_summary(records: List[Dict]) -> Dict:
    n = len(records)
    if n == 0:
        return {"error": "No frames in manifest"}

    lighting_counts = {"daylight": 0, "overcast": 0, "low_light": 0}
    for r in records:
        lighting_counts[r.get("lighting", "unknown")] = \
            lighting_counts.get(r.get("lighting", "unknown"), 0) + 1

    sun_elevs = [r["sun_elevation"] for r in records if r.get("sun_elevation") is not None]
    shadows   = [r["shadow_geometry_score"] for r in records if r.get("shadow_geometry_score") is not None]
    lats      = [r["latitude"]  for r in records if r.get("latitude")  is not None]
    lons      = [r["longitude"] for r in records if r.get("longitude") is not None]
    t_vals    = [r["timestamp_s"] for r in records]

    summary = {
        "total_frames":      n,
        "duration_s":        max(t_vals) if t_vals else 0,
        "fps_effective":     n / max(max(t_vals), 1e-9) if t_vals else 0,
        "lighting_counts":   lighting_counts,
        "gps_coverage_pct":  100 * len(lats) / n if n else 0,
        "sun_elev_min":      min(sun_elevs) if sun_elevs else None,
        "sun_elev_max":      max(sun_elevs) if sun_elevs else None,
        "sun_elev_mean":     float(np.mean(sun_elevs)) if sun_elevs else None,
        "shadow_min":        min(shadows) if shadows else None,
        "shadow_max":        max(shadows) if shadows else None,
        "shadow_mean":       float(np.mean(shadows)) if shadows else None,
        "lat_range":         (min(lats), max(lats)) if lats else None,
        "lon_range":         (min(lons), max(lons)) if lons else None,
        "focal_length_px":   records[0].get("focal_length_px"),
        "n_gps_interpolated": sum(1 for r in records if r.get("gps_interpolated", True)),
    }
    return summary


def print_summary(summary: Dict) -> str:
    lines = [
        "=" * 56,
        "  CRIS Stage 1 — Preprocessor Output Summary",
        "=" * 56,
        f"  Total frames extracted : {summary['total_frames']}",
        f"  Video duration covered : {summary['duration_s']:.1f} s",
        f"  Effective rate         : {summary['fps_effective']:.2f} fps",
        f"  Focal length (config)  : {summary['focal_length_px']} px",
        "",
        "  Lighting breakdown",
        f"    daylight   : {summary['lighting_counts'].get('daylight', 0)}",
        f"    overcast   : {summary['lighting_counts'].get('overcast', 0)}",
        f"    low_light  : {summary['lighting_counts'].get('low_light', 0)}",
        "",
        "  GPS",
        f"    coverage   : {summary['gps_coverage_pct']:.1f}%",
        f"    interpolated: {summary['n_gps_interpolated']} frames",
    ]

    if summary.get("lat_range"):
        lines += [
            f"    lat range  : {summary['lat_range'][0]:.5f} → {summary['lat_range'][1]:.5f}",
            f"    lon range  : {summary['lon_range'][0]:.5f} → {summary['lon_range'][1]:.5f}",
        ]

    if summary.get("sun_elev_mean") is not None:
        lines += [
            "",
            "  Sun elevation (°)",
            f"    min / mean / max : {summary['sun_elev_min']:.1f} / "
            f"{summary['sun_elev_mean']:.1f} / {summary['sun_elev_max']:.1f}",
        ]
    else:
        lines += ["", "  Sun elevation : not available (pysolar not installed or no GPS)"]

    if summary.get("shadow_mean") is not None:
        lines += [
            "",
            "  Shadow geometry score",
            f"    min / mean / max : {summary['shadow_min']:.2f} / "
            f"{summary['shadow_mean']:.2f} / {summary['shadow_max']:.2f}",
        ]

    lines.append("=" * 56)
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Plot 1: Lighting distribution bar chart
# ---------------------------------------------------------------------------

def plot_lighting_distribution(records: List[Dict], out_path: str) -> None:
    counts = {"daylight": 0, "overcast": 0, "low_light": 0}
    for r in records:
        key = r.get("lighting", "unknown")
        counts[key] = counts.get(key, 0) + 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [LIGHTING_COLORS.get(k, "#AAAAAA") for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Lighting Classification — Frame Counts", fontsize=13, pad=12)
    ax.set_ylabel("Number of frames")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 2: Sun elevation over time
# ---------------------------------------------------------------------------

def plot_sun_elevation(records: List[Dict], out_path: str) -> None:
    t_vals  = []
    e_vals  = []
    colors  = []

    for r in records:
        if r.get("sun_elevation") is None:
            continue
        t_vals.append(r["timestamp_s"])
        e_vals.append(r["sun_elevation"])
        colors.append(LIGHTING_COLORS.get(r.get("lighting", "daylight"), "#888888"))

    if not t_vals:
        logger.warning("No sun elevation data — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t_vals, e_vals, c=colors, s=12, alpha=0.85, zorder=3)
    ax.axhline(0, color="#CC0000", linewidth=0.8, linestyle="--", label="Horizon (0°)")
    ax.axhline(6, color="#FF8800", linewidth=0.6, linestyle=":", label="Civil twilight (6°)")

    # Legend for lighting colours
    patches = [mpatches.Patch(color=c, label=k) for k, c in LIGHTING_COLORS.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
              fontsize=8, loc="upper right")

    ax.set_title("Solar Elevation Angle over Time", fontsize=13, pad=10)
    ax.set_xlabel("Time offset (s)")
    ax.set_ylabel("Sun elevation (°)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3: Shadow geometry score over time
# ---------------------------------------------------------------------------

def plot_shadow_score(records: List[Dict], out_path: str) -> None:
    t_vals = [r["timestamp_s"] for r in records]
    s_vals = [r.get("shadow_geometry_score", 0.0) for r in records]
    colors = [LIGHTING_COLORS.get(r.get("lighting", "daylight"), "#888888") for r in records]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(t_vals, s_vals, c=colors, s=10, alpha=0.8, zorder=3)
    ax.plot(t_vals, s_vals, color="#CCCCCC", linewidth=0.5, zorder=2)

    patches = [mpatches.Patch(color=c, label=k) for k, c in LIGHTING_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    ax.set_title("Shadow Geometry Score over Time", fontsize=13, pad=10)
    ax.set_xlabel("Time offset (s)")
    ax.set_ylabel("Mean Sobel gradient\n(S-channel)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 4: GPS track scatter
# ---------------------------------------------------------------------------

def plot_gps_track(records: List[Dict], out_path: str) -> None:
    lats = [r["latitude"]  for r in records if r.get("latitude")  is not None]
    lons = [r["longitude"] for r in records if r.get("longitude") is not None]

    if not lats:
        logger.warning("No GPS data — skipping GPS track plot")
        return

    colors = [
        LIGHTING_COLORS.get(r.get("lighting", "daylight"), "#888888")
        for r in records if r.get("latitude") is not None
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(lons, lats, c=colors, s=14, alpha=0.85, zorder=3)

    # Start / end markers
    ax.plot(lons[0],  lats[0],  "g^", markersize=10, label="Start", zorder=5)
    ax.plot(lons[-1], lats[-1], "rs", markersize=10, label="End",   zorder=5)

    patches = [mpatches.Patch(color=c, label=k) for k, c in LIGHTING_COLORS.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0], fontsize=8)

    ax.set_title("GPS Track (coloured by lighting class)", fontsize=13, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.spines[["top", "right"]].set_visible(False)
    # Equal aspect ratio so track isn't distorted
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 5: Frame grid (sampled N frames with labels)
# ---------------------------------------------------------------------------

def plot_frame_grid(
    records: List[Dict],
    out_path: str,
    n: int = 16,
    frames_dir_override: Optional[str] = None,
) -> None:
    """
    Sample n frames evenly from the manifest and display them in a grid.
    Each frame shows: index, lighting label, sun elevation.
    """
    if not records:
        logger.warning("No records — skipping frame grid")
        return

    # Evenly sample n indices across the full manifest
    indices = np.linspace(0, len(records) - 1, min(n, len(records)), dtype=int)
    selected = [records[i] for i in indices]

    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for flat_i, rec in enumerate(selected):
        r_i, c_i = divmod(flat_i, cols)
        ax = axes[r_i][c_i]

        # Resolve frame path (allow override for portability)
        frame_path = rec["frame_path"]
        if frames_dir_override and not Path(frame_path).exists():
            frame_path = str(Path(frames_dir_override) / Path(frame_path).name)

        if not Path(frame_path).exists():
            ax.text(0.5, 0.5, "File not\nfound", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            continue

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            ax.text(0.5, 0.5, "Read\nerror", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        # Build label
        lighting = rec.get("lighting", "?")
        sun = rec.get("sun_elevation")
        sun_str = f"{sun:.1f}°" if sun is not None else "n/a"
        t_str = f"{rec['timestamp_s']:.1f}s"
        label = f"#{rec['frame_index']}  t={t_str}\n{lighting}  ☀ {sun_str}"

        border_color = LIGHTING_COLORS.get(lighting, "#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        ax.set_title(label, fontsize=7.5, pad=3, color="black")

    fig.suptitle(
        f"Frame Sample Grid  ({len(selected)} of {len(records)} frames)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 6: Lighting timeline strip
# ---------------------------------------------------------------------------

def plot_lighting_timeline(records: List[Dict], out_path: str) -> None:
    """
    Horizontal colour strip: one thin column per frame, coloured by lighting.
    Gives an immediate visual of how lighting changes across the video.
    """
    colors = [
        LIGHTING_COLORS.get(r.get("lighting", "daylight"), "#888888")
        for r in records
    ]

    # Build an Nx1 image
    n = len(colors)
    strip = np.zeros((1, n, 3), dtype=np.uint8)
    for i, hex_color in enumerate(colors):
        h = hex_color.lstrip("#")
        strip[0, i] = [int(h[j:j+2], 16) for j in (0, 2, 4)]

    fig, ax = plt.subplots(figsize=(12, 1.2))
    ax.imshow(strip, aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Frame index →", fontsize=9)
    ax.set_title("Lighting Timeline  (amber=daylight | grey=overcast | dark=low_light)",
                 fontsize=10, pad=6)

    # Add lighting class tick labels at transitions
    prev = None
    for i, r in enumerate(records):
        cur = r.get("lighting")
        if cur != prev:
            ax.axvline(i, color="white", linewidth=0.5, alpha=0.5)
            prev = cur

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Stage 1 (preprocessor) output — prints summary and saves plots."
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest.json produced by preprocessor.py",
    )
    parser.add_argument(
        "--frames-dir", default=None,
        help="Override base directory for frame images (useful if manifest was "
             "generated on a different machine)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/inspection/preprocessor",
        help="Directory to save plots and summary (default: data/processed/inspection/preprocessor/)",
    )
    parser.add_argument(
        "--grid-n", type=int, default=16,
        help="Number of frames to show in the frame grid (default: 16)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not call plt.show() — only save files. Use on headless servers.",
    )
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    records = load_manifest(args.manifest)
    if not records:
        logger.error("Manifest is empty — nothing to inspect.")
        sys.exit(1)

    # Summary
    summary = compute_summary(records)
    summary_text = print_summary(summary)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Saved: %s", summary_path)

    # Plots
    plot_lighting_distribution(records, str(out_dir / "lighting_distribution.png"))
    plot_sun_elevation(records,         str(out_dir / "sun_elevation.png"))
    plot_shadow_score(records,          str(out_dir / "shadow_score.png"))
    plot_gps_track(records,             str(out_dir / "gps_track.png"))
    plot_frame_grid(
        records,
        str(out_dir / "frame_grid.png"),
        n=args.grid_n,
        frames_dir_override=args.frames_dir,
    )
    plot_lighting_timeline(records,     str(out_dir / "lighting_timeline.png"))

    if not args.no_display:
        plt.show()

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    print("Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()