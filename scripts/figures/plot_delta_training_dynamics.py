# -*- coding: utf-8 -*-
"""
plot_delta_training_dynamics.py
Publication-grade Figure: Training Dynamics & Convergence for DimeNet++ Delta.
- Wide canvas (255 x 145 mm) eliminating text squeeze and overlap
- GridSpec with dedicated legend columns
- Gray box on plot 'a' completely removed
- Legends placed in upper right position outside each panel
- Larger, high-contrast axis fonts and concise axis labels
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

# =============================================================================
# STYLE SETTINGS (Canvas = 255 x 145 mm)
# =============================================================================
MM_TO_INCH = 1.0 / 25.4
FIG_W = 255 * MM_TO_INCH
FIG_H = 145 * MM_TO_INCH

BLACK = "#111111"
AXIS_LW = 0.75
STRONG_LW = 1.25
THIN_LW = 0.55

FONT_MAIN = 7.0
FONT_LABEL = 7.5
FONT_TICK = 6.8
FONT_LEGEND = 6.4
FONT_PANEL = 8.5

COLORS = {
    "train": "#1B4F72",       # глубокий синий
    "val": "#C0392B",         # приглушенный кораллово-красный
    "r2": "#4A2E76",          # темно-фиолетовый
    "mae": "#2874A6",         # синий
    "polar": "#1B4F72",       # темно-синий
    "mol_mae": "#7858A6",     # фиолетовый
    "emd": "#4A2E76",         # темно-фиолетовый
    "cosine": "#1E8449",      # глубокий зеленый
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_LABEL,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.fontsize": FONT_LEGEND,
    "axes.linewidth": AXIS_LW,
    "xtick.major.width": AXIS_LW,
    "ytick.major.width": AXIS_LW,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": STRONG_LW,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

def apply_nature_axes(ax):
    """Классические открытые оси чисто чёрного цвета."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.grid(False)

def apply_nature_twinx(ax_twin):
    """Правая ось строго чёрного цвета."""
    ax_twin.spines["top"].set_visible(False)
    ax_twin.spines["left"].set_visible(False)
    ax_twin.spines["bottom"].set_visible(False)
    ax_twin.spines["right"].set_visible(True)
    ax_twin.spines["right"].set_color(BLACK)
    ax_twin.spines["right"].set_linewidth(AXIS_LW)
    ax_twin.tick_params(axis="y", colors=BLACK)
    ax_twin.grid(False)

def add_panel_label(ax, text):
    ax.text(-0.16, 1.08, text, transform=ax.transAxes, fontsize=FONT_PANEL, fontweight="bold", color=BLACK)

def main():
    json_path = REPO_ROOT / "results/physics_v2/metrics/dimenet_pp_delta_seed0_physics.json"
    print(f"--> Reading metrics from {json_path}")
    with open(json_path, "r") as f:
        run_data = json.load(f)

    history = run_data["history"]
    best_ep = run_data["best_epoch"]

    epochs = np.array([h["epoch"] for h in history])
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss = np.array([h["loss"] for h in history])
    weighted_r2 = np.array([h["weighted_r2"] for h in history])
    weighted_mae = np.array([h["weighted_mae"] for h in history])
    polar_mae = np.array([h["polar_mae"] for h in history])
    mol_mae = np.array([h["molecular_mae"] for h in history])
    emd = np.array([h["emd_raw"] * 1e4 for h in history])
    mol_cos = np.array([h["molecular_cosine"] for h in history])

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Сетка: 2 строки, 4 колонки с широким промежутком wspace=0.52
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1.0, 0.42, 1.0, 0.42],
        wspace=0.52, hspace=0.42,
        left=0.07, right=0.97, bottom=0.09, top=0.93
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax_leg1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_leg2 = fig.add_subplot(gs[0, 3])

    ax3 = fig.add_subplot(gs[1, 0])
    ax_leg3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax_leg4 = fig.add_subplot(gs[1, 3])

    for leg_ax in (ax_leg1, ax_leg2, ax_leg3, ax_leg4):
        leg_ax.axis("off")

    for ax in (ax1, ax2, ax3, ax4):
        apply_nature_axes(ax)

    add_panel_label(ax1, "a")
    add_panel_label(ax2, "b")
    add_panel_label(ax3, "c")
    add_panel_label(ax4, "d")

    # =========================================================================
    # Panel a: Loss Dynamics (Серый блок полностью удален)
    # =========================================================================
    l1 = ax1.plot(epochs, train_loss, color=COLORS["train"], lw=STRONG_LW, linestyle="-", label="Train Loss")
    l2 = ax1.plot(epochs, val_loss, color=COLORS["val"], lw=STRONG_LW, linestyle="-", label="Val Loss")
    ax1.axvline(best_ep, color=BLACK, lw=THIN_LW, linestyle=":")
    ax1.scatter([best_ep], [val_loss[best_ep - 1]], color=COLORS["val"], s=26, edgecolor=BLACK, linewidth=0.75, zorder=5)

    ax1.set_xlabel("Training epoch", color=BLACK)
    ax1.set_ylabel("Physics-guided loss", color=BLACK)
    ax1.set_xlim(0, 205)
    ax1.set_ylim(0.30, 0.95)

    # Легенда в правом верхнем углу относительно графика
    ax_leg1.legend(handles=l1 + l2, loc="upper left", bbox_to_anchor=(0.04, 1.0), frameon=False)

    # =========================================================================
    # Panel b: Atomic Accuracy (R^2_w & Weighted MAE)
    # =========================================================================
    line_b1 = ax2.plot(epochs, weighted_r2, color=COLORS["r2"], lw=STRONG_LW, linestyle="-", label=r"Val $R^2_w$")
    ax2.set_xlabel("Training epoch", color=BLACK)
    ax2.set_ylabel(r"Atomic $R^2_w$", color=BLACK)
    ax2.set_xlim(0, 205)
    ax2.set_ylim(0.80, 0.93)

    ax2_r = ax2.twinx()
    apply_nature_twinx(ax2_r)
    line_b2 = ax2_r.plot(epochs, weighted_mae, color=COLORS["mae"], lw=STRONG_LW, linestyle="--", label="Weighted MAE")
    ax2_r.set_ylabel(r"Weighted MAE ($\mathrm{\AA}^2$)", color=BLACK, labelpad=6)
    ax2_r.set_ylim(0.10, 0.16)

    ax_leg2.legend(handles=line_b1 + line_b2, loc="upper left", bbox_to_anchor=(0.04, 1.0), frameon=False)

    # =========================================================================
    # Panel c: Polar Region MAE & Molecular MAE (Лаконичные подписи)
    # =========================================================================
    line_c1 = ax3.plot(epochs, polar_mae, color=COLORS["polar"], lw=STRONG_LW, linestyle="-", label="Polar MAE")
    ax3.set_xlabel("Training epoch", color=BLACK)
    ax3.set_ylabel(r"Polar MAE ($\mathrm{\AA}^2$)", color=BLACK)
    ax3.set_xlim(0, 205)
    ax3.set_ylim(0.012, 0.0175)

    ax3_r = ax3.twinx()
    apply_nature_twinx(ax3_r)
    line_c2 = ax3_r.plot(epochs, mol_mae, color=COLORS["mol_mae"], lw=STRONG_LW, linestyle="--", label="Mol. MAE")
    ax3_r.set_ylabel(r"Molecular MAE ($\mathrm{\AA}^2$)", color=BLACK, labelpad=6)
    ax3_r.set_ylim(0.45, 0.85)

    ax_leg3.legend(handles=line_c1 + line_c2, loc="upper left", bbox_to_anchor=(0.04, 1.0), frameon=False)

    # =========================================================================
    # Panel d: Wasserstein EMD & Molecular Cosine
    # =========================================================================
    line_d1 = ax4.plot(epochs, emd, color=COLORS["emd"], lw=STRONG_LW, linestyle="-", label=r"EMD ($\times 10^{-4}$)")
    ax4.set_xlabel("Training epoch", color=BLACK)
    ax4.set_ylabel(r"EMD ($\times 10^{-4}\ e\ \mathrm{\AA}^{-2}$)", color=BLACK)
    ax4.set_xlim(0, 205)
    ax4.set_ylim(5.0, 11.0)

    ax4_r = ax4.twinx()
    apply_nature_twinx(ax4_r)
    line_d2 = ax4_r.plot(epochs, mol_cos, color=COLORS["cosine"], lw=STRONG_LW, linestyle="--", label="Mol. Cosine")
    ax4_r.set_ylabel("Molecular Cosine", color=BLACK, labelpad=6)
    ax4_r.set_ylim(0.980, 0.996)

    ax_leg4.legend(handles=line_d1 + line_d2, loc="upper left", bbox_to_anchor=(0.04, 1.0), frameon=False)

    out_pdf = SCRIPT_DIR / "Figure_Delta_Training_Dynamics.pdf"
    out_png = SCRIPT_DIR / "Figure_Delta_Training_Dynamics.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f" [DONE] Successfully generated clean wide figure:\n  -> {out_pdf}\n  -> {out_png}")

if __name__ == "__main__":
    main()
