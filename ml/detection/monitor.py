"""
ml/detection/monitor.py

PURPOSE:
    Live training monitor — reads results.csv and renders a clean
    dashboard of all key metrics. Run it in a second terminal while
    training is running, or after training completes.

USAGE:
    # Live mode: auto-refreshes every 30s while training runs
    python ml/detection/monitor.py

    # One-shot: render once and save to PNG
    python ml/detection/monitor.py --save

    # Point at a specific results.csv
    python ml/detection/monitor.py --csv runs/detect/rtdetr_road/results.csv

    # Custom refresh interval (seconds)
    python ml/detection/monitor.py --interval 60
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "runs" / "detect" / "rtdetr_road" / "results.csv"

# ── Style ──────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
BG2       = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"
ORANGE    = "#f0883e"
RED       = "#ff7b72"
PURPLE    = "#bc8cff"
YELLOW    = "#d29922"

LOSS_COLOR  = RED
VAL_COLOR   = BLUE
MAP50_COLOR = GREEN
MAP95_COLOR = PURPLE
LR_COLOR    = YELLOW


def style_ax(ax, title="", ylabel="", xlabel="Epoch"):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)


def plot_line(ax, x, y, color, label="", lw=1.8, fill=True):
    if len(y) == 0:
        return
    # Replace nan with None for clean plotting
    y_clean = pd.to_numeric(y, errors="coerce")
    mask = ~np.isnan(y_clean)
    if mask.sum() < 1:
        return
    xv = np.array(x)[mask]
    yv = y_clean[mask]
    ax.plot(xv, yv, color=color, linewidth=lw, label=label, zorder=3)
    if fill:
        ax.fill_between(xv, yv, alpha=0.08, color=color, zorder=2)
    # Mark best point
    if len(yv) > 2:
        best_idx = np.argmax(yv) if "mAP" in label or "P" == label or "R" == label else np.argmin(yv)
        ax.scatter(xv[best_idx], yv[best_idx], color=color, s=40, zorder=5, edgecolors="white", linewidths=0.5)


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def render(df: pd.DataFrame, save_path: Path = None):
    epochs = df["epoch"] + 1  # 0-indexed → 1-indexed

    matplotlib.rcParams.update({
        "font.family":      "monospace",
        "text.color":       TEXT,
        "axes.labelcolor":  TEXT_DIM,
        "xtick.color":      TEXT_DIM,
        "ytick.color":      TEXT_DIM,
    })

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── Title bar ──────────────────────────────────────────────────────────────
    n_epochs = len(df)
    best_map95 = df["metrics/mAP50-95(B)"].max()
    best_epoch = df["metrics/mAP50-95(B)"].idxmax() + 1
    latest_loss = df["train/giou_loss"].iloc[-1] + df["train/cls_loss"].iloc[-1]

    fig.text(0.02, 0.97, "RT-DETR-L  //  Training Monitor",
             color=TEXT, fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.935,
             f"epochs: {n_epochs}   │   best mAP50-95: {best_map95:.4f} @ epoch {best_epoch}"
             f"   │   latest total loss: {latest_loss:.4f}",
             color=TEXT_DIM, fontsize=9, va="top", fontfamily="monospace")

    # ── Grid layout ───────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        top=0.90, bottom=0.06,
        left=0.05, right=0.98,
        hspace=0.45, wspace=0.35
    )

    # ── 1. Train losses (combined) ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, title="Train Losses", ylabel="Loss")
    plot_line(ax1, epochs, df["train/giou_loss"], LOSS_COLOR,  label="GIoU")
    plot_line(ax1, epochs, df["train/cls_loss"],  ORANGE,      label="Cls")
    plot_line(ax1, epochs, df["train/l1_loss"],   BLUE,        label="L1")
    ax1.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)

    # ── 2. Val losses ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    style_ax(ax2, title="Val Losses", ylabel="Loss")
    if "val/giou_loss" in df.columns:
        plot_line(ax2, epochs, df["val/giou_loss"], LOSS_COLOR, label="GIoU")
        plot_line(ax2, epochs, df["val/cls_loss"],  ORANGE,     label="Cls")
        plot_line(ax2, epochs, df["val/l1_loss"],   BLUE,       label="L1")
        ax2.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)
    else:
        ax2.text(0.5, 0.5, "val losses not available\nin this Ultralytics version",
                 ha="center", va="center", color=TEXT_DIM, fontsize=8,
                 transform=ax2.transAxes)

    # ── 3. mAP50 + mAP50-95 ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, title="mAP (validation)", ylabel="mAP")
    plot_line(ax3, epochs, df["metrics/mAP50(B)"],    MAP50_COLOR, label="mAP50")
    plot_line(ax3, epochs, df["metrics/mAP50-95(B)"], MAP95_COLOR, label="mAP50-95")
    ax3.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)

    # ── 4. Precision + Recall ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    style_ax(ax4, title="Precision & Recall", ylabel="Score")
    plot_line(ax4, epochs, df["metrics/precision(B)"], BLUE,  label="Precision")
    plot_line(ax4, epochs, df["metrics/recall(B)"],    GREEN, label="Recall")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)

    # ── 5. Learning rate ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, title="Learning Rate", ylabel="LR")
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    for i, col in enumerate(lr_cols[:3]):
        colors = [LR_COLOR, ORANGE, BLUE]
        plot_line(ax5, epochs, df[col], colors[i], label=col.replace("lr/", ""), fill=False)
    if lr_cols:
        ax5.legend(fontsize=6, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)
    ax5.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2e"))

    # ── 6. GIoU loss detail ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, title="GIoU Loss Detail", ylabel="Loss")
    plot_line(ax6, epochs, df["train/giou_loss"], LOSS_COLOR, label="train", lw=2)
    ax6.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)

    # ── 7. Cls loss detail ────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    style_ax(ax7, title="Cls Loss Detail", ylabel="Loss")
    plot_line(ax7, epochs, df["train/cls_loss"], ORANGE, label="train", lw=2)
    ax7.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT_DIM)

    # ── 8. Stats table ────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.set_facecolor(BG2)
    for spine in ax8.spines.values():
        spine.set_edgecolor(BORDER)
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_title("Summary", color=TEXT, fontsize=9, fontweight="bold", pad=6)

    latest = df.iloc[-1]
    best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]

    stats = [
        ("Epochs done",    f"{n_epochs}"),
        ("GPU mem (GB)",   f"{latest.get('train/gpu_mem', 0):.2f}" if "train/gpu_mem" in df.columns else "—"),
        ("",               ""),
        ("── Latest ──",   ""),
        ("mAP50",          f"{latest['metrics/mAP50(B)']:.4f}"),
        ("mAP50-95",       f"{latest['metrics/mAP50-95(B)']:.4f}"),
        ("Precision",      f"{latest['metrics/precision(B)']:.4f}"),
        ("Recall",         f"{latest['metrics/recall(B)']:.4f}"),
        ("GIoU loss",      f"{latest['train/giou_loss']:.4f}"),
        ("Cls loss",       f"{latest['train/cls_loss']:.4f}"),
        ("",               ""),
        ("── Best ──",     ""),
        ("mAP50-95",       f"{best_map95:.4f}"),
        ("@ epoch",        f"{best_epoch}"),
        ("mAP50",          f"{best_row['metrics/mAP50(B)']:.4f}"),
    ]

    y_pos = 0.97
    for label, value in stats:
        if label.startswith("──"):
            color = TEXT
            fw = "bold"
        elif label == "":
            y_pos -= 0.04
            continue
        else:
            color = TEXT_DIM
            fw = "normal"
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes,
                 color=color, fontsize=7.5, va="top", fontweight=fw,
                 fontfamily="monospace")
        ax8.text(0.98, y_pos, value, transform=ax8.transAxes,
                 color=GREEN if "mAP" in label else TEXT,
                 fontsize=7.5, va="top", ha="right", fontfamily="monospace")
        y_pos -= 0.062

    plt.suptitle("")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    plt.close(fig)


def main(csv_path: Path, live: bool, interval: int, save: bool):
    if not csv_path.exists():
        print(f"[ERROR] results.csv not found: {csv_path}")
        print("        Training hasn't started yet, or path is wrong.")
        sys.exit(1)

    if save:
        df = load_csv(csv_path)
        out = csv_path.parent / "training_monitor.png"
        render(df, save_path=out)
        return

    if live:
        print(f"[MONITOR] Live mode — refreshing every {interval}s")
        print(f"          Reading: {csv_path}")
        print(f"          Press Ctrl+C to stop\n")
        matplotlib.use("TkAgg")  # interactive backend
        try:
            while True:
                df = load_csv(csv_path)
                print(f"  Epoch {len(df)} — mAP50-95: {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}  "
                      f"GIoU: {df['train/giou_loss'].iloc[-1]:.4f}  "
                      f"Cls: {df['train/cls_loss'].iloc[-1]:.4f}")
                render(df)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Monitor stopped.")
    else:
        df = load_csv(csv_path)
        render(df)


def parse_args():
    p = argparse.ArgumentParser(description="RT-DETR training monitor")
    p.add_argument("--csv",      type=str, default=str(DEFAULT_CSV),
                   help="Path to results.csv")
    p.add_argument("--live",     action="store_true",
                   help="Live mode: auto-refresh while training runs")
    p.add_argument("--interval", type=int, default=30,
                   help="Refresh interval in seconds for live mode (default: 30)")
    p.add_argument("--save",     action="store_true",
                   help="Save plot to PNG instead of showing interactively")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        csv_path = Path(args.csv),
        live     = args.live,
        interval = args.interval,
        save     = args.save,
    )