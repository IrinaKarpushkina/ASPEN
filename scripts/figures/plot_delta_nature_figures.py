# -*- coding: utf-8 -*-
"""
plot_delta_nature_figures.py
Generates Figures 1-3 for DimeNet++ Delta (Nature / npj style)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# Загрузка реальных данных
data = np.load("results/physics_v2/delta_full_diagnostics.npz")
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
mol_true = data["mol_profiles_true"]
mol_pred = data["mol_profiles_pred"]

# Константы стиля из вашего шаблона
MM_TO_INCH = 1.0 / 25.4
FIG_W = 183 * MM_TO_INCH
BLACK = "#111111"
COLORS = {
    "purple_dark": "#4A2E76",
    "purple_mid": "#7858A6",
    "purple_light": "#B5A0D9",
    "purple_fill": "#F0ECF8",
    "blue_dark": "#1B4F72",
    "blue_deep": "#2874A6",
    "blue_mid": "#3E8BC2",
    "blue_light": "#85C1E9",
    "blue_fill": "#EBF4FA",
    "accent_coral": "#C0392B",
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
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.75,
    "axes.edgecolor": BLACK,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

def apply_axes(ax, square=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.grid(False)
    if square:
        ax.set_box_aspect(1)

def add_label(ax, text):
    ax.text(-0.16, 1.06, text, transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=BLACK)

def calc_r2_mae(y_true, y_pred):
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12)
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, mae

# =============================================================================
# FIGURE 1: Physical Invariants of DimeNet++ Delta
# =============================================================================
fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH), layout="constrained")
for ax in (ax1a, ax1b, ax1c, ax1d):
    apply_axes(ax, square=True)

add_label(ax1a, "a")
add_label(ax1b, "b")
add_label(ax1c, "c")
add_label(ax1d, "d")

# 1a. Total Area Parity
r2_a, mae_a = calc_r2_mae(mol_true_a, mol_pred_a)
ax1a.scatter(mol_true_a, mol_pred_a, s=9, color=COLORS["purple_mid"], alpha=0.6, edgecolors=BLACK, linewidths=0.4, rasterized=True)
lims_a = [min(mol_true_a.min(), mol_pred_a.min()) * 0.95, max(mol_true_a.max(), mol_pred_a.max()) * 1.05]
ax1a.plot(lims_a, lims_a, "--", color=BLACK, lw=0.75)
ax1a.set_xlim(lims_a); ax1a.set_ylim(lims_a)
ax1a.set_xlabel(r"DFT cavity area, $S_{\mathrm{DFT}}$ ($\mathrm{\AA}^2$)")
ax1a.set_ylabel(r"Predicted area, $\sum S_i$ ($\mathrm{\AA}^2$)")
ax1a.text(0.05, 0.92, f"$R^2 = {r2_a:.4f}$\n$\\mathrm{{MAE}} = {mae_a:.2f}\\ \\mathrm{{\\AA}}^2$", transform=ax1a.transAxes, va="top", fontsize=5.8)

# 1b. Charge Neutrality Residual
ax1b.hist(mol_charge_res, bins=45, color=COLORS["blue_fill"], edgecolor=COLORS["blue_dark"], linewidth=0.75, density=True)
ax1b.axvline(0, color=BLACK, lw=1.0, linestyle="--")
ax1b.set_xlabel(r"Residual net charge, $\int \sigma p(\sigma) d\sigma$ ($e$)")
ax1b.set_ylabel("Probability density")
ax1b.text(0.05, 0.92, f"$\\mu = {np.mean(mol_charge_res):.2e}\\ e$\n$\\mathrm{{std}} = {np.std(mol_charge_res):.2e}\\ e$", transform=ax1b.transAxes, va="top", fontsize=5.8)

# 1c. Polar Area Parity
r2_p, mae_p = calc_r2_mae(mol_true_p, mol_pred_p)
ax1c.scatter(mol_true_p, mol_pred_p, s=9, color=COLORS["blue_deep"], alpha=0.6, edgecolors=BLACK, linewidths=0.4, rasterized=True)
lims_p = [0, max(mol_true_p.max(), mol_pred_p.max()) * 1.05]
ax1c.plot(lims_p, lims_p, "--", color=BLACK, lw=0.75)
ax1c.set_xlim(lims_p); ax1c.set_ylim(lims_p)
ax1c.set_xlabel(r"DFT polar area ($|\sigma| \geq 0.01$) ($\mathrm{\AA}^2$)")
ax1c.set_ylabel(r"Predicted polar area ($\mathrm{\AA}^2$)")
ax1c.text(0.05, 0.92, f"$R^2 = {r2_p:.4f}$\n$\\mathrm{{MAE}} = {mae_p:.2f}\\ \\mathrm{{\\AA}}^2$", transform=ax1c.transAxes, va="top", fontsize=5.8)

# 1d. Second Moment (Electrostatic Spread)
r2_m2, mae_m2 = calc_r2_mae(mol_m2_t, mol_m2_p)
ax1d.scatter(mol_m2_t, mol_m2_p, s=9, color=COLORS["purple_dark"], alpha=0.6, edgecolors=BLACK, linewidths=0.4, rasterized=True)
lims_m2 = [0, max(mol_m2_t.max(), mol_m2_p.max()) * 1.05]
ax1d.plot(lims_m2, lims_m2, "--", color=BLACK, lw=0.75)
ax1d.set_xlim(lims_m2); ax1d.set_ylim(lims_m2)
ax1d.set_xlabel(r"DFT electrostatic moment $M_2$ ($e^2\ \mathrm{\AA}^{-2}$)")
ax1d.set_ylabel(r"Predicted moment $M_2$ ($e^2\ \mathrm{\AA}^{-2}$)")
ax1d.text(0.05, 0.92, f"$R^2 = {r2_m2:.4f}$\n$\\mathrm{{MAE}} = {mae_m2:.2e}$", transform=ax1d.transAxes, va="top", fontsize=5.8)

fig1.savefig("Figure1_Delta_Physics_Invariants.pdf", format="pdf", bbox_inches="tight")
fig1.savefig("Figure1_Delta_Physics_Invariants.png", format="png", dpi=600, bbox_inches="tight")
plt.close(fig1)

# =============================================================================
# FIGURE 2: Error Anatomy along Sigma, Elements and EMD
# =============================================================================
fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH), layout="constrained")
for ax in (ax2a, ax2b, ax2c, ax2d):
    apply_axes(ax, square=True)

add_label(ax2a, "a")
add_label(ax2b, "b")
add_label(ax2c, "c")
add_label(ax2d, "d")

# 2a. Bin-wise MAE
bin_mae = np.mean(np.abs(targets - preds), axis=0)
bin_std = np.std(np.abs(targets - preds), axis=0)
ax2a.bar(sigma_bins, bin_mae, width=0.0008, color=COLORS["blue_mid"], edgecolor=BLACK, linewidth=0.5, zorder=3)
ax2a.errorbar(sigma_bins[::4], bin_mae[::4], yerr=bin_std[::4] / np.sqrt(len(targets)), fmt="none", ecolor=BLACK, elinewidth=0.5, capsize=1.2, zorder=4)
ax2a.axvline(-0.01, color=COLORS["purple_dark"], linewidth=0.6, linestyle="--")
ax2a.axvline(0.01, color=COLORS["purple_dark"], linewidth=0.6, linestyle="--")
ax2a.set_xlabel(r"Screening charge density $\sigma$ ($e\ \mathrm{\AA}^{-2}$)")
ax2a.set_ylabel(r"Atomic bin MAE ($\mathrm{\AA}^2$)")
ax2a.text(-0.018, bin_mae.max() * 0.9, "Acceptor", fontsize=5.5, ha="center")
ax2a.text(0.018, bin_mae.max() * 0.9, "Donor", fontsize=5.5, ha="center")

# 2b. Element-wise Breakdown (if atomic numbers present)
elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}
present_elems = [e for e in [1, 6, 7, 8, 9, 16, 17] if np.sum(z == e) > 50]
if present_elems:
    elem_labels = [elem_map[e] for e in present_elems]
    elem_maes = [np.mean(np.abs(targets[z == e] - preds[z == e])) for e in present_elems]
    bars = ax2b.bar(elem_labels, elem_maes, color=COLORS["purple_mid"], edgecolor=BLACK, linewidth=0.75, width=0.55)
    for bar in bars:
        ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=5.1)
    ax2b.set_xlabel("Chemical element")
    ax2b.set_ylabel(r"Atomic MAE ($\mathrm{\AA}^2$)")
    ax2b.set_ylim(0, max(elem_maes) * 1.25)

# 2c. EMD / Wasserstein Cumulative Distribution Function (CDF)
sorted_emd = np.sort(mol_emds[~np.isnan(mol_emds)])
cdf = np.arange(1, len(sorted_emd) + 1) / len(sorted_emd)
ax2c.plot(sorted_emd, cdf, color=COLORS["purple_dark"], lw=1.2)
ax2c.fill_between(sorted_emd, 0, cdf, color=COLORS["purple_fill"], alpha=0.8)
median_emd = np.median(sorted_emd)
p90_emd = np.percentile(sorted_emd, 90)
ax2c.axvline(median_emd, color=COLORS["accent_coral"], linestyle=":", lw=0.8)
ax2c.text(median_emd * 1.1, 0.5, f"Median EMD\n{median_emd:.4f}", fontsize=5.3, color=COLORS["accent_coral"])
ax2c.set_xlabel(r"Earth Mover's Distance ($e\ \mathrm{\AA}^{-2}$)")
ax2c.set_ylabel("Cumulative fraction")
ax2c.set_ylim(0, 1.05)

# 2d. Scaling with Molecule Size
mol_sizes = data["mol_sizes"]
ax2d.scatter(mol_sizes, mol_maes, s=9, color=COLORS["blue_mid"], alpha=0.5, edgecolors=BLACK, linewidths=0.3, rasterized=True)
# Трендовая линия медианы
bins_sz = np.linspace(mol_sizes.min(), mol_sizes.max(), 8)
bin_centers = (bins_sz[:-1] + bins_sz[1:]) / 2
bin_meds = [np.median(mol_maes[(mol_sizes >= bins_sz[i]) & (mol_sizes < bins_sz[i+1])]) for i in range(len(bins_sz)-1)]
ax2d.plot(bin_centers, bin_meds, "o-", color=COLORS["purple_dark"], lw=1.0, markersize=4.0, label="Bin median")
ax2d.set_xlabel("Number of atoms in molecule")
ax2d.set_ylabel(r"Molecular MAE ($\mathrm{\AA}^2$)")
ax2d.legend(loc="upper left", frameon=False, fontsize=5.2)

fig2.savefig("Figure2_Delta_Error_Anatomy.pdf", format="pdf", bbox_inches="tight")
fig2.savefig("Figure2_Delta_Error_Anatomy.png", format="png", dpi=600, bbox_inches="tight")
plt.close(fig2)

# =============================================================================
# FIGURE 3: Profile Gallery (Representative molecules)
# =============================================================================
fig3, axes = plt.subplots(2, 3, figsize=(FIG_W, 110 * MM_TO_INCH), layout="constrained")
axes = axes.flatten()

# Подбираем 6 интересных примеров по квантилям ошибки (от лучшего к 95%-перцентилю)
pcts = [0.01, 0.20, 0.40, 0.60, 0.80, 0.95]
sorted_idx = np.argsort(mol_maes)

for i, p in enumerate(pcts):
    idx = sorted_idx[int(p * (len(sorted_idx) - 1))]
    ax = axes[i]
    apply_axes(ax, square=True)
    add_label(ax, chr(ord('a') + i))

    ax.fill_between(sigma_bins, 0, mol_pred[idx], color=COLORS["purple_fill"], alpha=0.7)
    ax.plot(sigma_bins, mol_true[idx], color=BLACK, lw=1.1, label="DFT Total")
    ax.plot(sigma_bins, mol_pred[idx], color=COLORS["purple_dark"], lw=0.9, linestyle="--", label="DimeNet++ Δ")

    ax.text(0.04, 0.90, f"MAE = {mol_maes[idx]:.2f} $\\mathrm{{\\AA}}^2$\nEMD = {mol_emds[idx]:.4f}", transform=ax.transAxes, fontsize=5.2)
    ax.set_xlabel(r"$\sigma$ ($e\ \mathrm{\AA}^{-2}$)")
    ax.set_ylabel(r"$p(\sigma)$ ($\mathrm{\AA}^2$)")
    if i == 0:
        ax.legend(loc="upper right", frameon=False, fontsize=5.0)

fig3.savefig("Figure3_Delta_Profile_Gallery.pdf", format="pdf", bbox_inches="tight")
fig3.savefig("Figure3_Delta_Profile_Gallery.png", format="png", dpi=600, bbox_inches="tight")
plt.close(fig3)

print(" [DONE] Successfully generated Figures 1, 2, and 3 for DimeNet++ Delta!")
