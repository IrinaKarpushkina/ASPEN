# -*- coding: utf-8 -*-
"""
plot_delta_deep_diagnostics.py
Path: scripts/figures/plot_delta_deep_diagnostics.py
Deep Physical Diagnostics for DimeNet++ Delta (Nature standard):
- Hexbin atomic parity with density
- Region-wise error breakdown (Donor / Acceptor / Nonpolar)
- Error cancellation dynamics (Atomic vs Molecular)
- Element-wise cosine similarity distribution
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

MM_TO_INCH = 1.0 / 25.4
FIG_W = 183 * MM_TO_INCH
BLACK = "#111111"
AXIS_LW = 0.75
STRONG_LW = 1.15
THIN_LW = 0.55

COLORS = {
    "purple_dark": "#4A2E76",
    "purple_mid": "#7858A6",
    "purple_light": "#B5A0D9",
    "purple_fill": "#F0ECF8",
    "blue_dark": "#1B4F72",
    "blue_deep": "#2874A6",
    "blue_mid": "#3E8BC2",
    "blue_fill": "#EBF4FA",
    "accent_coral": "#C0392B",
    "accent_green": "#1E8449",
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.fontsize": 5.5,
    "axes.linewidth": AXIS_LW,
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
    ax.text(-0.16, 1.06, text, transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=BLACK)

def main():
    npz_path = REPO_ROOT / "results/physics_v2/delta_test_results.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found!")
        sys.exit(1)

    print(f"--> Loading real test results from {npz_path}...")
    data = np.load(npz_path)
    targets = data["targets"]
    preds = data["preds"]
    sigma_bins = data["sigma_bins"]
    z = data["atomic_numbers"]
    mol_sizes = data["mol_sizes"]
    mol_maes = data["mol_maes"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(FIG_W, 140 * MM_TO_INCH))
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, wspace=0.38, hspace=0.38)

    for ax in (ax1, ax2, ax3, ax4):
        apply_axes(ax, square=True)

    add_label(ax1, "a")
    add_label(ax2, "b")
    add_label(ax3, "c")
    add_label(ax4, "d")

    # =========================================================================
    # Panel a: 2D Hexbin Atomic Profile Parity (221k points sampled)
    # =========================================================================
    np.random.seed(42)
    sample_idx = np.random.choice(len(targets), size=min(100000, len(targets)), replace=False)
    t_sample = targets[sample_idx].flatten()
    p_sample = preds[sample_idx].flatten()

    # Фильтруем нулевые области для наглядного логарифмического hexbin
    mask = (t_sample > 1e-4) | (p_sample > 1e-4)
    t_val = t_sample[mask]
    p_val = p_sample[mask]

    hb = ax1.hexbin(t_val, p_val, gridsize=65, cmap="Purples", norm=LogNorm(),
                    mincnt=1, linewidths=0.1, edgecolors="none", rasterized=True)
    max_val = min(float(np.percentile(t_val, 99.9)), float(np.percentile(p_val, 99.9)))
    ax1.plot([0, max_val], [0, max_val], "--", color=BLACK, lw=0.8, zorder=5)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_xlabel(r"DFT atomic profile, $p_i(\sigma)$ ($\mathrm{\AA}^2$)")
    ax1.set_ylabel(r"DimeNet++ $\Delta$ predicted, $\hat{p}_i(\sigma)$ ($\mathrm{\AA}^2$)")

    cb_ax = fig.add_axes([0.435, 0.58, 0.012, 0.32])
    cb = fig.colorbar(hb, cax=cb_ax)
    cb.set_label("Point density", color=BLACK, fontsize=5.5)
    cb.ax.tick_params(labelsize=5.0, colors=BLACK)
    cb.outline.set_linewidth(AXIS_LW)

    # =========================================================================
    # Panel b: Region-wise Error Breakdown (Nonpolar vs Donor vs Acceptor)
    # =========================================================================
    acc_mask = sigma_bins <= -0.010
    don_mask = sigma_bins >= 0.010
    non_mask = (sigma_bins > -0.008) & (sigma_bins < 0.008)

    mae_acc = np.mean(np.abs(targets[:, acc_mask] - preds[:, acc_mask]), axis=1)
    mae_non = np.mean(np.abs(targets[:, non_mask] - preds[:, non_mask]), axis=1)
    mae_don = np.mean(np.abs(targets[:, don_mask] - preds[:, don_mask]), axis=1)

    region_data = [mae_non, mae_acc, mae_don]
    labels_b = ["Non-polar\nbulk", "H-Bond\nacceptor", "H-Bond\ndonor"]

    bp = ax2.boxplot(
        region_data, tick_labels=labels_b, patch_artist=True, widths=0.52,
        medianprops=dict(color=BLACK, linewidth=STRONG_LW),
        flierprops=dict(marker="o", markersize=1.2, markerfacecolor="#7F8C8D", markeredgecolor="none", alpha=0.25),
        whiskerprops=dict(color=BLACK, linewidth=THIN_LW),
        capprops=dict(color=BLACK, linewidth=THIN_LW),
    )
    palette_b = [COLORS["blue_mid"], COLORS["purple_mid"], COLORS["accent_coral"]]
    for patch, col in zip(bp["boxes"], palette_b):
        patch.set_facecolor(col)
        patch.set_edgecolor(BLACK)
        patch.set_linewidth(AXIS_LW)
        patch.set_alpha(0.85)

    ax2.set_ylabel(r"Atomic regional MAE ($\mathrm{\AA}^2$)")
    ax2.set_ylim(0, float(np.percentile(mae_don, 98.5)) * 1.6)

    # =========================================================================
    # Panel c: Atomic-to-Molecular Error Cancellation
    # =========================================================================
    atom_maes_per_mol = []
    idx = 0
    for sz in mol_sizes:
        atom_maes_per_mol.append(float(np.mean(np.abs(targets[idx: idx + sz] - preds[idx: idx + sz]))))
        idx += sz
    atom_maes_per_mol = np.array(atom_maes_per_mol)

    ax3.scatter(atom_maes_per_mol, mol_maes, s=7, color=COLORS["purple_dark"],
                alpha=0.45, edgecolors=BLACK, linewidths=0.25, rasterized=True)
    r2_c = 1.0 - np.sum((mol_maes - atom_maes_per_mol)**2) / np.sum((mol_maes - np.mean(mol_maes))**2)
    ax3.set_xlabel(r"Mean atomic MAE in mol ($\mathrm{\AA}^2$)")
    ax3.set_ylabel(r"Aggregated molecular MAE ($\mathrm{\AA}^2$)")
    ax3.text(0.05, 0.92, f"Correlation\nPearson $r = {np.corrcoef(atom_maes_per_mol, mol_maes)[0, 1]:.3f}$",
             transform=ax3.transAxes, fontsize=5.3)

    # =========================================================================
    # Panel d: Element-wise Cosine Similarity Distribution
    # =========================================================================
    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
    present = [e for e in [1, 6, 7, 8, 9, 16, 17] if np.sum(z == e) > 100]

    elem_cos_list = []
    elem_names = []
    eps = 1e-12
    for e in present:
        t_e = targets[z == e]
        p_e = preds[z == e]
        # Косинус для каждого атома
        num = np.sum(t_e * p_e, axis=1)
        den = np.linalg.norm(t_e, axis=1) * np.linalg.norm(p_e, axis=1) + eps
        cos_e = num / den
        elem_cos_list.append(cos_e[~np.isnan(cos_e)])
        elem_names.append(elem_map[e])

    bp_d = ax4.boxplot(
        elem_cos_list, tick_labels=elem_names, patch_artist=True, widths=0.55,
        medianprops=dict(color=BLACK, linewidth=STRONG_LW),
        flierprops=dict(marker="o", markersize=1.2, markerfacecolor="#7F8C8D", markeredgecolor="none", alpha=0.25),
        whiskerprops=dict(color=BLACK, linewidth=THIN_LW),
        capprops=dict(color=BLACK, linewidth=THIN_LW),
    )
    for patch in bp_d["boxes"]:
        patch.set_facecolor(COLORS["blue_mid"])
        patch.set_edgecolor(BLACK)
        patch.set_linewidth(AXIS_LW)
        patch.set_alpha(0.75)

    ax4.set_xlabel("Chemical element")
    ax4.set_ylabel("Atomic cosine similarity")
    ax4.set_ylim(0.75, 1.01)

    out_pdf = SCRIPT_DIR / "Figure4_Real_Deep_Diagnostics.pdf"
    out_png = SCRIPT_DIR / "Figure4_Real_Deep_Diagnostics.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f" [DONE] Saved deep diagnostics:\n  -> {out_pdf}\n  -> {out_png}")

if __name__ == "__main__":
    main()
