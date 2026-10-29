# -*- coding: utf-8 -*-
"""
plot_delta_learning.py
Publication Figure: Training Dynamics & Convergence for DimeNet++ Delta
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# STYLE SETTINGS (Nature / Double-Column 183 mm)
# =============================================================================
MM_TO_INCH = 1.0 / 25.4
FIG_W = 183 * MM_TO_INCH
BLACK = "#111111"
AXIS_LW = 0.75
STRONG_LW = 1.1
THIN_LW = 0.55

COLORS = {
    "purple_dark": "#4A2E76",
    "purple_mid": "#7858A6",
    "purple_light": "#B5A0D9",
    "purple_fill": "#F0ECF8",
    "blue_dark": "#1B4F72",
    "blue_deep": "#2874A6",
    "blue_mid": "#3E8BC2",
    "accent_coral": "#C0392B",
    "accent_green": "#27AE60",
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.5,
    "axes.linewidth": AXIS_LW,
    "axes.edgecolor": BLACK,
    "xtick.major.width": AXIS_LW,
    "ytick.major.width": AXIS_LW,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": STRONG_LW,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

def apply_axes(ax, square=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.grid(False)
    if square:
        ax.set_box_aspect(1)

def add_label(ax, text):
    ax.text(-0.16, 1.06, text, transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=BLACK)

# =============================================================================
# LOAD JSON DATA
# =============================================================================
json_path = "results/physics_v2/metrics/dimenet_pp_delta_seed0_physics.json"
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
emd = np.array([h["emd_raw"] * 1e4 for h in history])  # scale to x10^-4
mol_cos = np.array([h["molecular_cosine"] for h in history])

# =============================================================================
# FIGURE PLOTTING
# =============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH), layout="constrained")

for ax in (ax1, ax2, ax3, ax4):
    apply_axes(ax, square=True)

add_label(ax1, "a")
add_label(ax2, "b")
add_label(ax3, "c")
add_label(ax4, "d")

# -----------------------------------------------------------------------------
# Panel a: Loss Dynamics
# -----------------------------------------------------------------------------
ax1.plot(epochs, train_loss, color=COLORS["blue_dark"], lw=1.2, label="Train Loss")
ax1.plot(epochs, val_loss, color=COLORS["accent_coral"], lw=1.2, label="Val Loss")
ax1.axvline(best_ep, color=BLACK, lw=THIN_LW, linestyle="--")
ax1.scatter([best_ep], [val_loss[best_ep - 1]], color=COLORS["accent_coral"], s=25, edgecolor=BLACK, zorder=5)

ax1.set_xlabel("Training epoch")
ax1.set_ylabel("Physics-guided loss")
ax1.set_xlim(0, 205)
ax1.set_ylim(0.30, 0.95)
ax1.legend(loc="upper right", frameon=False)
ax1.text(0.05, 0.20, f"Best checkpoint:\nEpoch {best_ep}\nVal Loss = {val_loss[best_ep-1]:.4f}",
         transform=ax1.transAxes, fontsize=5.3,
         bbox=dict(boxstyle="round,pad=0.25", facecolor=COLORS["purple_fill"], edgecolor=COLORS["purple_light"], lw=THIN_LW))

# -----------------------------------------------------------------------------
# Panel b: Atomic Accuracy (R^2_w & Weighted MAE)
# -----------------------------------------------------------------------------
ax2.plot(epochs, weighted_r2, color=COLORS["purple_dark"], lw=1.2, label=r"Val $R^2_w$")
ax2.set_xlabel("Training epoch")
ax2.set_ylabel(r"Atomic $R^2_w$", color=COLORS["purple_dark"])
ax2.set_xlim(0, 205)
ax2.set_ylim(0.80, 0.93)
ax2.tick_params(axis="y", labelcolor=COLORS["purple_dark"])

ax2_r = ax2.twinx()
ax2_r.plot(epochs, weighted_mae, color=COLORS["blue_mid"], lw=1.1, linestyle="--", label="Weighted MAE")
ax2_r.set_ylabel(r"Weighted MAE ($\mathrm{\AA}^2$)", color=COLORS["blue_mid"])
ax2_r.set_ylim(0.10, 0.16)
ax2_r.spines["top"].set_visible(False)
ax2_r.spines["right"].set_color(COLORS["blue_mid"])
ax2_r.tick_params(axis="y", labelcolor=COLORS["blue_mid"])

lines2_a, labels2_a = ax2.get_legend_handles_labels()
lines2_b, labels2_b = ax2_r.get_legend_handles_labels()
ax2.legend(lines2_a + lines2_b, labels2_a + labels2_b, loc="center right", frameon=False)

# -----------------------------------------------------------------------------
# Panel c: Polar Region MAE & Molecular MAE
# -----------------------------------------------------------------------------
ax3.plot(epochs, polar_mae, color=COLORS["blue_deep"], lw=1.2, label="Polar MAE")
ax3.set_xlabel("Training epoch")
ax3.set_ylabel(r"Polar MAE ($|\sigma| \ge 0.01$) ($\mathrm{\AA}^2$)", color=COLORS["blue_deep"])
ax3.set_xlim(0, 205)
ax3.set_ylim(0.012, 0.0175)
ax3.tick_params(axis="y", labelcolor=COLORS["blue_deep"])

ax3_r = ax3.twinx()
ax3_r.plot(epochs, mol_mae, color=COLORS["purple_mid"], lw=1.1, linestyle="--", label="Mol. MAE")
ax3_r.set_ylabel(r"Molecular MAE ($\mathrm{\AA}^2$)", color=COLORS["purple_mid"])
ax3_r.set_ylim(0.45, 0.85)
ax3_r.spines["top"].set_visible(False)
ax3_r.spines["right"].set_color(COLORS["purple_mid"])
ax3_r.tick_params(axis="y", labelcolor=COLORS["purple_mid"])

lines3_a, labels3_a = ax3.get_legend_handles_labels()
lines3_b, labels3_b = ax3_r.get_legend_handles_labels()
ax3.legend(lines3_a + lines3_b, labels3_a + labels3_b, loc="upper right", frameon=False)

# -----------------------------------------------------------------------------
# Panel d: Wasserstein EMD & Molecular Cosine
# -----------------------------------------------------------------------------
ax4.plot(epochs, emd, color=COLORS["purple_dark"], lw=1.2, label=r"EMD ($\times 10^{-4}$)")
ax4.set_xlabel("Training epoch")
ax4.set_ylabel(r"Earth Mover's Dist. ($\times 10^{-4}\ e\ \mathrm{\AA}^{-2}$)", color=COLORS["purple_dark"])
ax4.set_xlim(0, 205)
ax4.set_ylim(5.0, 11.0)
ax4.tick_params(axis="y", labelcolor=COLORS["purple_dark"])

ax4_r = ax4.twinx()
ax4_r.plot(epochs, mol_cos, color=COLORS["accent_green"], lw=1.1, linestyle="--", label="Mol. Cosine")
ax4_r.set_ylabel("Molecular Cosine Similarity", color=COLORS["accent_green"])
ax4_r.set_ylim(0.980, 0.996)
ax4_r.spines["top"].set_visible(False)
ax4_r.spines["right"].set_color(COLORS["accent_green"])
ax4_r.tick_params(axis="y", labelcolor=COLORS["accent_green"])

lines4_a, labels4_a = ax4.get_legend_handles_labels()
lines4_b, labels4_b = ax4_r.get_legend_handles_labels()
ax4.legend(lines4_a + lines4_b, labels4_a + labels4_b, loc="center right", frameon=False)

# Сохранение в векторном и растровом виде
fig.savefig("Figure_Delta_Training_Dynamics.pdf", format="pdf", bbox_inches="tight")
fig.savefig("Figure_Delta_Training_Dynamics.png", format="png", dpi=600, bbox_inches="tight")
plt.close(fig)

print(" Successfully saved Figure_Delta_Training_Dynamics.pdf and .png!")
