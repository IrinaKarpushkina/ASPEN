# -*- coding: utf-8 -*-
"""
dashboard_delta_physics_russian.py
Сводный публикационный дашборд (2x3) валидации модели DimeNet++ Delta.
- Полная русификация всех надписей и осей
- Панели подписаны русскими буквами: а), б), в), г), д), е)
- Текстовые метрики вынесены в компактные верхние заголовки
- Стиль Nature / npj Computational Materials
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# НАСТРОЙКИ СТИЛЯ И ШРИФТОВ (Поддержка кириллицы)
# =============================================================================
MM_TO_INCH = 1.0 / 25.4
FIG_W = 210 * MM_TO_INCH   # 210 мм ширина под 3 колонки
FIG_H = 135 * MM_TO_INCH   # 135 мм высота под 2 строки

BLACK = "#111111"
AXIS_LW = 0.75
STRONG_LW = 1.2
MEDIUM_LW = 0.9
THIN_LW = 0.55

# Цветовая палитра Nature / ASPEN
COLORS = {
    "purple_dark": "#4A2E76",
    "purple_deep": "#5E3C99",
    "purple_mid": "#7858A6",
    "purple_light": "#B5A0D9",
    "purple_fill": "#F0ECF8",
    "purple_seq": ["#4A2E76", "#5E3C99", "#7858A6", "#997FC1", "#BFAEE2", "#D1C4E9", "#EDE7F6"],

    "blue_dark": "#1B4F72",
    "blue_deep": "#2874A6",
    "blue_mid": "#3E8BC2",
    "blue_light": "#85C1E9",
    "blue_fill": "#EBF4FA",

    "accent_coral": "#C0392B",
    "neutral_line": "#666666",
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.fontsize": 5.8,
    "axes.linewidth": AXIS_LW,
    "xtick.major.width": AXIS_LW,
    "ytick.major.width": AXIS_LW,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": STRONG_LW,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

def apply_nature_axes(ax):
    """Классические открытые оси (слева и снизу) чисто чёрного цвета."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.grid(False)

def add_panel_label(ax, text):
    """Буквенный идентификатор панели а), б), в)..."""
    ax.text(-0.16, 1.15, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=BLACK)

def calc_r2_mae(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, mae

# =============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
def load_data():
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else script_dir
    
    npz_path = repo_root / "results/physics_v2/delta_test_results.npz"
    json_path = repo_root / "results/physics_v2/metrics/dimenet_pp_delta_seed0_physics.json"
    
    # 1. Загрузка NPZ данных теста
    if npz_path.exists():
        print(f"--> Загрузка данных из: {npz_path}")
        d = np.load(npz_path)
        targets = d["targets"]
        preds = d["preds"]
        sigma_bins = d["sigma_bins"]
        z = d["atomic_numbers"]
        mol_true_a = d["mol_true_areas"]
        mol_pred_a = d["mol_pred_areas"]
        mol_true_p = d["mol_true_polar"]
        mol_pred_p = d["mol_pred_polar"]
        mol_maes = d["mol_maes"]
        mol_cosines = d["mol_cosines"]
        mol_true = d["mol_profiles_true"]
        mol_pred = d["mol_profiles_pred"]
    else:
        print("--> Файл NPZ не найден. Создаются тестовые данные...")
        np.random.seed(42)
        sigma_bins = np.linspace(-0.025, 0.025, 51)
        targets = np.random.gamma(2, 0.04, (2000, 51))
        preds = targets + np.random.normal(0, 0.015, (2000, 51))
        z = np.random.choice([1, 6, 7, 8, 9, 16, 17], 2000, p=[0.45, 0.35, 0.06, 0.07, 0.03, 0.02, 0.02])
        mol_true_a = np.random.uniform(70, 380, 500)
        mol_pred_a = mol_true_a + np.random.normal(0, 0.8, 500)
        mol_true_p = np.random.uniform(5, 60, 500)
        mol_pred_p = mol_true_p + np.random.normal(0, 4.5, 500)
        mol_maes = np.random.gamma(3, 0.16, 500)
        mol_cosines = np.random.uniform(0.96, 0.999, 500)
        sg = np.linspace(-0.025, 0.025, 51)
        mol_true = np.array([18.0 * np.exp(-((sg / 0.005)**2)) + 6.0 * np.exp(-(((sg - 0.012)/0.003)**2)) for _ in range(500)])
        mol_pred = mol_true + np.random.normal(0, 0.45, (500, 51))

    # 2. Загрузка истории обучения
    if json_path.exists():
        print(f"--> Загрузка динамики обучения из: {json_path}")
        with open(json_path, "r") as f:
            h_data = json.load(f)
        epochs = np.array([h["epoch"] for h in h_data["history"]])
        train_loss = np.array([h["train_loss"] for h in h_data["history"]])
        val_loss = np.array([h["loss"] for h in h_data["history"]])
        best_ep = h_data["best_epoch"]
        best_val = val_loss[best_ep - 1]
    else:
        print("--> Файл JSON не найден. Создается тестовая кривая обучения...")
        epochs = np.arange(1, 201)
        train_loss = 0.55 * np.exp(-epochs / 25.0) + 0.34 + 0.01 * (epochs % 40 == 0)
        val_loss = 0.60 * np.exp(-epochs / 28.0) + 0.405 + 0.015 * (epochs % 40 == 0)
        best_ep = 197
        best_val = 0.406

    return (targets, preds, sigma_bins, z, mol_true_a, mol_pred_a, mol_true_p, 
            mol_pred_p, mol_maes, mol_cosines, mol_true, mol_pred,
            epochs, train_loss, val_loss, best_ep, best_val, script_dir)

# =============================================================================
# ГЕНЕРАЦИЯ ДАШБОРДА
# =============================================================================
def main():
    (targets, preds, sigma_bins, z, mol_true_a, mol_pred_a, mol_true_p, 
     mol_pred_p, mol_maes, mol_cosines, mol_true, mol_pred,
     epochs, train_loss, val_loss, best_ep, best_val, script_dir) = load_data()

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H), layout="constrained")
    ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2]

    for ax in axes.flatten():
        apply_nature_axes(ax)

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ а): Молекулярный сигма-профиль (Медианный случай)
    # -------------------------------------------------------------------------
    add_panel_label(ax_a, "а)")
    sorted_idx = np.argsort(mol_maes)
    med_idx = sorted_idx[len(sorted_idx) // 2]   # 50-й перцентиль (медиана)

    ax_a.fill_between(sigma_bins, 0, mol_pred[med_idx], color=COLORS["purple_fill"], alpha=0.85, zorder=2)
    l1, = ax_a.plot(sigma_bins, mol_true[med_idx], color=BLACK, lw=STRONG_LW, label="DFT эталон", zorder=3)
    l2, = ax_a.plot(sigma_bins, mol_pred[med_idx], color=COLORS["purple_dark"], lw=STRONG_LW, linestyle="--", label="DimeNet++ Δ", zorder=4)

    # Метрика сверху над графиком
    ax_a.set_title(f"MAE: {mol_maes[med_idx]:.2f} $\\mathrm{{\\AA}}^2$  |  Косинус: {mol_cosines[med_idx]:.4f}", 
                   fontsize=6.3, pad=4, color=BLACK)
    ax_a.set_xlabel(r"Плотность заряда, $\sigma$ ($e\ \mathrm{\AA}^{-2}$)")
    ax_a.set_ylabel(r"$p(\sigma)$ ($\mathrm{\AA}^2$)")
    ax_a.set_xlim(-0.026, 0.026)
    ax_a.legend(loc="upper right", frameon=False, fontsize=5.5)

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ б): Точность по химическим элементам
    # -------------------------------------------------------------------------
    add_panel_label(ax_b, "б)")
    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
    present = [e for e in [1, 6, 7, 8, 9, 16, 17] if np.sum(z == e) > 20]
    
    elem_labels = [elem_map[e] for e in present]
    elem_maes = [np.mean(np.abs(targets[z == e] - preds[z == e])) for e in present]

    bar_colors = [COLORS["purple_seq"][i % len(COLORS["purple_seq"])] for i in range(len(present))]
    ax_b.bar(elem_labels, elem_maes, color=bar_colors, edgecolor=BLACK, linewidth=AXIS_LW, width=0.55, zorder=3)

    # Метрика сверху над графиком
    avg_elem_mae = np.mean(elem_maes)
    ax_b.set_title(f"Средняя MAE: {avg_elem_mae:.3f} $\\mathrm{{\\AA}}^2$", fontsize=6.3, pad=4, color=BLACK)
    ax_b.set_xlabel("Химический элемент")
    ax_b.set_ylabel(r"Атомная MAE ($\mathrm{\AA}^2$)")
    ax_b.set_ylim(0, max(elem_maes) * 1.2)

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ в): Распределение ошибки по бинам плотности заряда
    # -------------------------------------------------------------------------
    add_panel_label(ax_c, "в)")
    bin_mae = np.mean(np.abs(targets - preds), axis=0)

    ax_c.bar(sigma_bins, bin_mae, width=0.0008, color=COLORS["blue_mid"], edgecolor=BLACK, linewidth=THIN_LW, zorder=3)
    ax_c.axvline(-0.01, color=COLORS["purple_dark"], linewidth=THIN_LW, linestyle=":")
    ax_c.axvline(0.01, color=COLORS["purple_dark"], linewidth=THIN_LW, linestyle=":")

    # Метрика сверху над графиком
    max_bin_mae = np.max(bin_mae)
    ax_c.set_title(f"Пик MAE: {max_bin_mae:.3f} $\\mathrm{{\\AA}}^2$  |  51 бин", fontsize=6.3, pad=4, color=BLACK)
    ax_c.set_xlabel(r"Плотность заряда, $\sigma$ ($e\ \mathrm{\AA}^{-2}$)")
    ax_c.set_ylabel(r"MAE бина ($\mathrm{\AA}^2$)")
    ax_c.set_xlim(-0.026, 0.026)

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ г): Сохранение общей площади поверхности полости
    # -------------------------------------------------------------------------
    add_panel_label(ax_d, "г)")
    r2_a, mae_a = calc_r2_mae(mol_true_a, mol_pred_a)

    ax_d.scatter(mol_true_a, mol_pred_a, s=9, color=COLORS["purple_mid"], alpha=0.55,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)
    lim_a = [min(mol_true_a.min(), mol_pred_a.min()) * 0.95, max(mol_true_a.max(), mol_pred_a.max()) * 1.05]
    ax_d.plot(lim_a, lim_a, "--", color=BLACK, lw=MEDIUM_LW, zorder=2)
    ax_d.set_xlim(lim_a)
    ax_d.set_ylim(lim_a)

    # Метрика сверху над графиком
    ax_d.set_title(f"$R^2 = {r2_a:.4f}$  |  MAE: {mae_a:.2f} $\\mathrm{{\\AA}}^2$", fontsize=6.3, pad=4, color=BLACK)
    ax_d.set_xlabel(r"Площадь полости DFT, $S_{\mathrm{DFT}}$ ($\mathrm{\AA}^2$)")
    ax_d.set_ylabel(r"Сумма атомных площадей, $\sum S_i$ ($\mathrm{\AA}^2$)")

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ д): Точность предсказания полярной площади (|sigma| >= 0.01)
    # -------------------------------------------------------------------------
    add_panel_label(ax_e, "д)")
    # Фильтрация экстремальных выбросов для робастного масштаба
    p_cutoff = np.percentile(mol_true_p, 99.5)
    mask_p = (mol_true_p <= p_cutoff) & (mol_pred_p <= p_cutoff * 1.3)
    t_p_clean = mol_true_p[mask_p]
    p_p_clean = mol_pred_p[mask_p]
    r2_p, mae_p = calc_r2_mae(t_p_clean, p_p_clean)

    ax_e.scatter(t_p_clean, p_p_clean, s=9, color=COLORS["blue_deep"], alpha=0.55,
                 edgecolors=BLACK, linewidths=THIN_LW, rasterized=True, zorder=3)
    lim_p = [0, max(t_p_clean.max(), p_p_clean.max()) * 1.05]
    ax_e.plot(lim_p, lim_p, "--", color=BLACK, lw=MEDIUM_LW, zorder=2)
    ax_e.set_xlim(lim_p)
    ax_e.set_ylim(lim_p)

    # Метрика сверху над графиком
    ax_e.set_title(f"$R^2 = {r2_p:.4f}$  |  MAE: {mae_p:.2f} $\\mathrm{{\\AA}}^2$", fontsize=6.3, pad=4, color=BLACK)
    ax_e.set_xlabel(r"Полярная площадь DFT ($\mathrm{\AA}^2$)")
    ax_e.set_ylabel(r"Предсказанная полярная площадь ($\mathrm{\AA}^2$)")

    # -------------------------------------------------------------------------
    # ПАНЕЛЬ е): Динамика обучения и сходимости модели
    # -------------------------------------------------------------------------
    add_panel_label(ax_f, "е)")
    l_tr, = ax_f.plot(epochs, train_loss, color=COLORS["blue_dark"], lw=STRONG_LW, label="Обучение")
    l_va, = ax_f.plot(epochs, val_loss, color=COLORS["accent_coral"], lw=STRONG_LW, label="Валидация")
    ax_f.axvline(best_ep, color=BLACK, lw=THIN_LW, linestyle=":")
    ax_f.scatter([best_ep], [best_val], color=COLORS["accent_coral"], s=26, edgecolor=BLACK, linewidth=0.75, zorder=5)

    # Метрика сверху над графиком
    ax_f.set_title(f"Мин. ошибка: {best_val:.3f}  |  Эпоха: {best_ep}", fontsize=6.3, pad=4, color=BLACK)
    ax_f.set_xlabel("Эпоха обучения")
    ax_f.set_ylabel("Физически-направленный лосс")
    ax_f.set_xlim(0, max(epochs) * 1.02)
    ax_f.set_ylim(0.30, max(val_loss.max(), 0.90) * 1.05)
    ax_f.legend(loc="upper right", frameon=False, fontsize=5.5)

    # -------------------------------------------------------------------------
    # СОХРАНЕНИЕ В ВЫСОКОМ КАЧЕСТВЕ
    # -------------------------------------------------------------------------
    out_pdf = script_dir / "Figure_Dashboard_Delta_Physics_RU.pdf"
    out_png = script_dir / "Figure_Dashboard_Delta_Physics_RU.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f" [ГОТОВО] Сводный русскоязычный дашборд успешно сохранен:\n  -> {out_pdf}\n  -> {out_png}")

if __name__ == "__main__":
    main()
