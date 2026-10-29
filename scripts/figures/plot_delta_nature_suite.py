# -*- coding: utf-8 -*-
"""
plot_delta_nature_suite.py
Path: scripts/figures/plot_delta_nature_suite.py
Generates Figures 1-3 for DimeNet++ Delta (Nature standard).
- Larger axis fonts, concise axis labels
- Complete removal of overlapping in-plot text boxes
- Multi-group comparison on Figure 2c (EMD by molecular size tertiles)
- Robust outlier filtering on Figures 1b, 1d, 2d
- High-level gallery in Figure 3 with titles instead of overlapping text
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

# =============================================================================
# GLOBAL NATURE RC-PARAMS & CONSTANTS
# =============================================================================
MM_TO_INCH = 1.0 / 25.4
FIG_W = 183 * MM_TO_INCH
BLACK = "#111111"
AXIS_LW = 0.75
STRONG_LW = 1.25
MEDIUM_LW = 1.0
THIN_LW = 0.55

FONT_MAIN = 7.0
FONT_LABEL = 7.5
FONT_TICK = 6.5
FONT_LEGEND = 6.2
FONT_PANEL = 8.5
FONT_ANNOT = 6.0

COLORS = {
    "purple_dark": "#4A2E76",
    "purple_deep": "#5E3C99",
    "purple_mid": "#7858A6",
    "purple_light": "#B5A0D9",
    "purple_fill": "#F0ECF8",
    "purple_seq": ["#4A2E76", "#5E3C99", "#7858A6", "#997FC1", "#BFAEE2"],

    "blue_dark": "#1B4F72",
    "blue_deep": "#2874A6",
    "blue_mid": "#3E8BC2",
    "blue_light": "#85C1E9",
    "blue_fill": "#EBF4FA",
    "blue_seq": ["#1B4F72", "#2874A6", "#3E8BC2", "#67A5D4", "#95C2E6"],

    "accent_coral": "#C0392B",
    "accent_green": "#27AE60",
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

def apply_axes(ax, square=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.grid(False)
    if square:
        ax.set_box_aspect(1)

def add_label(ax, text):
    ax.text(-0.17, 1.08, text, transform=ax.transAxes, fontsize=FONT_PANEL, fontweight="bold", color=BLACK)

def calc_r2_mae(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, mae

def main():
    npz_path = REPO_ROOT / "results/physics_v2/delta_test_results.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found! Please run 'python run_delta_eval.py' first.")
        sys.exit(1)

    print(f"--> Loading real test results from {npz_path}...")
    data = np.load(npz_path)
    targets = data["targets"]
    preds = data["preds"]
    sigma_bins = data["sigma_bins"]
    z = data["atomic_numbers"]
    mol_true_a = data["mol_true_areas"]
    mol_pred_a = data["mol_pred_areas"]
    mol_true_p = data["mol_true_polar"]
    mol_pred_p = data["mol_pred_polar"]
    mol_charge_res = data["mol_charge_res"]
    mol_m2_t = data["mol_m2_true"]
    mol_m2_p = data["mol_m2_pred"]
    mol_emds = data["mol_emds"]
    mol_maes = data["mol_maes"]
    mol_cosines = data["mol_cosines"]
    mol_sizes = data["mol_sizes"]
    mol_true = data["mol_profiles_true"]
    mol_pred = data["mol_profiles_pred"]

    # =========================================================================
    # FIGURE 1: Physical Conservation Laws
    # =========================================================================
    fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH), layout="constrained")
    for ax in (ax1a, ax1b, ax1c, ax1d):
        apply_axes(ax, square=True)

    add_label(ax1a, "a")
    add_label(ax1b, "b")
    add_label(ax1c, "c")
    add_label(ax1d, "d")

    # 1a: Surface Area Parity
    r2_a, mae_a = calc_r2_mae(mol_true_a, mol_pred_a)
    ax1a.scatter(mol_true_a, mol_pred_a, s=9, color=COLORS["purple_mid"], alpha=0.55,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)
    lim_a = [min(mol_true_a.min(), mol_pred_a.min()) * 0.95, max(mol_true_a.max(), mol_pred_a.max()) * 1.05]
    ax1a.plot(lim_a, lim_a, "--", color=BLACK, lw=MEDIUM_LW, zorder=2)
    ax1a.set_xlim(lim_a)
    ax1a.set_ylim(lim_a)
    ax1a.set_xlabel(r"DFT area, $S_{\mathrm{DFT}}$ ($\mathrm{\AA}^2$)", color=BLACK)
    ax1a.set_ylabel(r"Sum of atomic areas, $\sum S_i$ ($\mathrm{\AA}^2$)", color=BLACK)
    ax1a.text(0.05, 0.94, f"$R^2 = {r2_a:.4f}$\nMAE = {mae_a:.2f} $\\mathrm{{\\AA}}^2$",
              transform=ax1a.transAxes, va="top", fontsize=FONT_ANNOT, color=BLACK)

    # 1b: Net Screening Charge Neutrality (Без черточки на нуле, фильтрация выбросов)
    valid_res = mol_charge_res[np.isfinite(mol_charge_res)]
    q_low, q_high = np.percentile(valid_res, [0.5, 99.5])
    filtered_res = valid_res[(valid_res >= q_low) & (valid_res <= q_high)]

    ax1b.hist(filtered_res, bins=45, color=COLORS["blue_fill"], edgecolor=COLORS["blue_dark"],
              linewidth=AXIS_LW, density=True, zorder=3)
    ax1b.set_xlabel(r"Residual net charge ($e$)", color=BLACK)
    ax1b.set_ylabel("Probability density", color=BLACK)
    ax1b.text(0.05, 0.94, f"$\\mu = {np.mean(filtered_res):.1e}\\ e$\n$\\sigma = {np.std(filtered_res):.1e}\\ e$",
              transform=ax1b.transAxes, va="top", fontsize=FONT_ANNOT, color=BLACK)

    # 1c: Polar Area Parity
    r2_p, mae_p = calc_r2_mae(mol_true_p, mol_pred_p)
    ax1c.scatter(mol_true_p, mol_pred_p, s=9, color=COLORS["blue_deep"], alpha=0.55,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)
    lim_p = [0, max(mol_true_p.max(), mol_pred_p.max()) * 1.05]
    ax1c.plot(lim_p, lim_p, "--", color=BLACK, lw=MEDIUM_LW, zorder=2)
    ax1c.set_xlim(lim_p)
    ax1c.set_ylim(lim_p)
    ax1c.set_xlabel(r"DFT polar area ($\mathrm{\AA}^2$)", color=BLACK)
    ax1c.set_ylabel(r"Predicted polar area ($\mathrm{\AA}^2$)", color=BLACK)
    ax1c.text(0.05, 0.94, f"$R^2 = {r2_p:.4f}$\nMAE = {mae_p:.2f} $\\mathrm{{\\AA}}^2$",
              transform=ax1c.transAxes, va="top", fontsize=FONT_ANNOT, color=BLACK)

    # 1d: Second Electrostatic Moment M2 (Очистка выбросов для неискаженного масштаба)
    m2_cutoff = np.percentile(mol_m2_t, 99.5)
    mask_m2 = (mol_m2_t <= m2_cutoff) & (mol_m2_p <= m2_cutoff * 1.2)
    m2_t_clean = mol_m2_t[mask_m2]
    m2_p_clean = mol_m2_p[mask_m2]

    r2_m2, mae_m2 = calc_r2_mae(m2_t_clean, m2_p_clean)
    ax1d.scatter(m2_t_clean, m2_p_clean, s=9, color=COLORS["purple_dark"], alpha=0.55,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)
    lim_m2 = [0, max(m2_t_clean.max(), m2_p_clean.max()) * 1.05]
    ax1d.plot(lim_m2, lim_m2, "--", color=BLACK, lw=MEDIUM_LW, zorder=2)
    ax1d.set_xlim(lim_m2)
    ax1d.set_ylim(lim_m2)
    ax1d.set_xlabel(r"DFT 2nd moment, $M_2$ ($e^2\ \mathrm{\AA}^{-2}$)", color=BLACK)
    ax1d.set_ylabel(r"Predicted moment, $M_2$ ($e^2\ \mathrm{\AA}^{-2}$)", color=BLACK)
    ax1d.text(0.05, 0.94, f"$R^2 = {r2_m2:.4f}$\nMAE = {mae_m2:.2e}",
              transform=ax1d.transAxes, va="top", fontsize=FONT_ANNOT, color=BLACK)

    fig1.savefig(SCRIPT_DIR / "Figure1_Real_Physics_Invariants.pdf", format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig1.savefig(SCRIPT_DIR / "Figure1_Real_Physics_Invariants.png", format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Real Error Anatomy & Multi-group Diagnostics
    # =========================================================================
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH), layout="constrained")
    for ax in (ax2a, ax2b, ax2c, ax2d):
        apply_axes(ax, square=True)

    add_label(ax2a, "a")
    add_label(ax2b, "b")
    add_label(ax2c, "c")
    add_label(ax2d, "d")

    # 2a: Real Bin-wise Absolute Error (Без текстовых наслоений H-Bond)
    bin_mae = np.mean(np.abs(targets - preds), axis=0)
    bin_std = np.std(np.abs(targets - preds), axis=0)
    ax2a.bar(sigma_bins, bin_mae, width=0.0008, color=COLORS["blue_mid"], edgecolor=BLACK, linewidth=THIN_LW, zorder=3)
    ax2a.errorbar(sigma_bins[::4], bin_mae[::4], yerr=bin_std[::4] / np.sqrt(len(targets)),
                  fmt="none", ecolor=BLACK, elinewidth=THIN_LW, capsize=1.2, zorder=4)
    ax2a.axvline(-0.01, color=COLORS["purple_dark"], linewidth=THIN_LW, linestyle="--")
    ax2a.axvline(0.01, color=COLORS["purple_dark"], linewidth=THIN_LW, linestyle="--")
    ax2a.set_xlabel(r"Screening charge, $\sigma$ ($e\ \mathrm{\AA}^{-2}$)", color=BLACK)
    ax2a.set_ylabel(r"Bin MAE ($\mathrm{\AA}^2$)", color=BLACK)
    ax2a.set_xlim(-0.026, 0.026)

    # 2b: Element-wise Breakdown (Цветовая дифференциация, чистые столбцы без нагромождения цифр)
    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}
    present_elems = [e for e in [1, 6, 7, 8, 9, 16, 17] if np.sum(z == e) > 50]
    if present_elems:
        elem_labels = [elem_map[e] for e in present_elems]
        elem_maes = [np.mean(np.abs(targets[z == e] - preds[z == e])) for e in present_elems]
        bar_colors = [COLORS["purple_seq"][i % len(COLORS["purple_seq"])] for i in range(len(present_elems))]
        ax2b.bar(elem_labels, elem_maes, color=bar_colors, edgecolor=BLACK, linewidth=AXIS_LW, width=0.55, zorder=3)
        ax2b.set_xlabel("Chemical element", color=BLACK)
        ax2b.set_ylabel(r"Atomic MAE ($\mathrm{\AA}^2$)", color=BLACK)
        ax2b.set_ylim(0, max(elem_maes) * 1.15)

    # 2c: Multi-Group Comparison — Wasserstein EMD по размерам молекул (3 группы)
    # Заменяем пустой график информативным сравнением выборок
    q33, q66 = np.percentile(mol_sizes, [33.3, 66.6])
    mask_s = mol_sizes <= q33
    mask_m = (mol_sizes > q33) & (mol_sizes <= q66)
    mask_l = mol_sizes > q66

    groups = [
        ("Small", mol_emds[mask_s], COLORS["blue_dark"]),
        ("Medium", mol_emds[mask_m], COLORS["purple_mid"]),
        ("Large", mol_emds[mask_l], COLORS["accent_coral"]),
    ]

    for label, arr, col in groups:
        clean_arr = np.sort(arr[np.isfinite(arr)])
        cdf = np.arange(1, len(clean_arr) + 1) / len(clean_arr)
        ax2c.plot(clean_arr, cdf, color=col, lw=STRONG_LW, label=f"{label} ({len(clean_arr):,})")

    ax2c.set_xlabel(r"EMD ($e\ \mathrm{\AA}^{-2}$)", color=BLACK)
    ax2c.set_ylabel("Cumulative test fraction", color=BLACK)
    ax2c.set_ylim(0, 1.02)
    ax2c.legend(loc="lower right", frameon=False, fontsize=FONT_LEGEND)

    # 2d: Error Scaling with Molecular Size (Очистка редких выбросов, легенда не перекрывает точки)
    mae_cut = np.percentile(mol_maes, 99.3)
    valid_d = mol_maes <= mae_cut
    sizes_clean = mol_sizes[valid_d]
    maes_clean = mol_maes[valid_d]

    ax2d.scatter(sizes_clean, maes_clean, s=8, color=COLORS["blue_mid"], alpha=0.40,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)

    bins_sz = np.linspace(sizes_clean.min(), sizes_clean.max(), 8)
    bin_centers = (bins_sz[:-1] + bins_sz[1:]) / 2
    bin_meds = [np.median(maes_clean[(sizes_clean >= bins_sz[i]) & (sizes_clean < bins_sz[i+1])]) for i in range(len(bins_sz)-1)]
    ax2d.plot(bin_centers, bin_meds, "o-", color=COLORS["purple_dark"], lw=STRONG_LW, markersize=4.5,
              markeredgecolor=BLACK, markeredgewidth=AXIS_LW, label="Median trend", zorder=5)

    ax2d.set_xlabel("Number of atoms", color=BLACK)
    ax2d.set_ylabel(r"Molecular MAE ($\mathrm{\AA}^2$)", color=BLACK)
    ax2d.set_ylim(0, mae_cut * 1.15)
    # Легенда в чистом правом верхнем углу
    ax2d.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    fig2.savefig(SCRIPT_DIR / "Figure2_Real_Error_Anatomy.pdf", format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig2.savefig(SCRIPT_DIR / "Figure2_Real_Error_Anatomy.png", format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3: Representative Real Profile Gallery (Zero text collision)
    # =========================================================================
    fig3, axes = plt.subplots(2, 3, figsize=(FIG_W, 115 * MM_TO_INCH), layout="constrained")
    axes = axes.flatten()

    pcts = [0.05, 0.25, 0.50, 0.75, 0.90, 0.98]
    sorted_idx = np.argsort(mol_maes)

    for i, p in enumerate(pcts):
        idx = sorted_idx[int(p * (len(sorted_idx) - 1))]
        ax = axes[i]
        apply_axes(ax, square=True)
        add_label(ax, chr(ord('a') + i))

        ax.fill_between(sigma_bins, 0, mol_pred[idx], color=COLORS["purple_fill"], alpha=0.75, zorder=2)
        l_true, = ax.plot(sigma_bins, mol_true[idx], color=BLACK, lw=STRONG_LW, label="DFT Target", zorder=3)
        l_pred, = ax.plot(sigma_bins, mol_pred[idx], color=COLORS["purple_dark"], lw=MEDIUM_LW, linestyle="--", label="DimeNet++ Δ", zorder=4)

        # Текст вынесен строго в аккуратный заголовок панели — никакого наслоения на график
        ax.set_title(f"MAE: {mol_maes[idx]:.2f} $\\mathrm{{\\AA}}^2$ | Cos: {mol_cosines[idx]:.4f}",
                     fontsize=FONT_ANNOT, pad=3, color=BLACK)

        ax.set_xlabel(r"$\sigma$ ($e\ \mathrm{\AA}^{-2}$)", color=BLACK)
        ax.set_ylabel(r"$p(\sigma)$ ($\mathrm{\AA}^2$)", color=BLACK)
        ax.set_xlim(-0.026, 0.026)

    # Единая элегантная внешняя легенда для всей фигуры вверху
    fig3.legend(
        handles=[l_true, l_pred],
        labels=["DFT Target", "DimeNet++ Δ"],
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND
    )

    fig3.savefig(SCRIPT_DIR / "Figure3_Real_Profile_Gallery.pdf", format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig3.savefig(SCRIPT_DIR / "Figure3_Real_Profile_Gallery.png", format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig3)

    print(" [DONE] All figures (Figure1, Figure2, Figure3) successfully regenerated.")

if __name__ == "__main__":
    main()
