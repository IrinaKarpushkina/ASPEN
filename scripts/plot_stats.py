"""Графики для встречи: результаты ablation + статистические тесты (stats_compare.py).

Рисунки (PNG 200 dpi + SVG, светлый фон, крупные шрифты под проектор):
  1_forest_<metric>.png     - ГЛАВНЫЙ: для каждой группы вопросов - насколько каждая модель лучше/хуже эталона, 95% ДИ,
                              серая полоса = ±SESOI (разница, которой можно пренебречь). Справа вердикт словами.
  2_seeds_<metric>.png      - все модели: точки = отдельные сиды, чёрта = среднее; видно, где разброс больше разницы.
  3_tradeoff.png            - R² против polar_mae: MSE-головы и physics-головы лежат на одном фронте компромисса.
  4_scaling.png             - R² от числа параметров (кольцо = обучение остановилось рано, размер смешан с длиной обучения).

Запуск:
  python -m scripts.stats_compare --dirs <metrics...> --groups configs/stats/groups_ablation.yaml \
         --metrics weighted_r2 polar_mae --out-csv results/stats_seeds.csv        # (+ --mol-dir ... для уровня molecules)
  python -m scripts.plot_stats --dirs <metrics...> --stats-csv results/stats_seeds.csv --out-dir results/figs \
         --metric weighted_r2 --highlight dimenet_pp_enhanced_xtb sigma_dn_global
Зависимости: matplotlib, numpy, pandas. torch не нужен.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._runs import HIGHER_IS_BETTER, by_label, load_runs  # noqa: E402

# --- палитра (референс dataviz, светлая поверхность)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#9a9994"

METRIC_RU = {"weighted_r2": "weighted R²", "weighted_mae": "weighted MAE", "polar_mae": "polar MAE", "emd_raw": "EMD (raw)"}
SHORT_VERDICT = [("ЛУЧШЕ", "лучше"), ("ХУЖЕ", "хуже"), ("значимо, но пренебрежимо", "значимо, но мало"),
                 ("≈ эквивалентны", "≈ эквивалентны"), ("лучше, но", "лучше, эффект не доказан"),
                 ("хуже, но", "хуже, эффект не доказан"), ("не различимы", "не различимы"), ("нет теста", "нет теста (мало сидов)")]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.join(out_dir, name + ".png"))


def short_verdict(v):
    for key, ru in SHORT_VERDICT:
        if str(v).startswith(key):
            return ru
    return str(v)


def verdict_color(v):
    s = short_verdict(v)
    if s.startswith("лучше"):
        return BLUE
    if s.startswith("хуже"):
        return ORANGE
    return NEUTRAL


# ------------------------------------------------------------------ 1. forest
def forest(df, metric, level, out_dir, min_p_rows=1):
    d = df[(df.metric == metric) & (df.level == level)]
    if d.empty:
        print(f"[forest] нет строк для metric={metric}, level={level}")
        return
    groups = list(dict.fromkeys(d.group))
    n_rows = len(d) + len(groups) * 1.6
    fig, ax = plt.subplots(figsize=(13.5, max(3.5, 0.42 * n_rows + 1.8)))
    lim = np.nanmax(np.abs(np.r_[d.ci_lo.fillna(0), d.ci_hi.fillna(0), d.improvement])) * 1.12 or 1e-3
    y, yticks, ylabels = 0.0, [], []
    for g in groups:
        gg = d[d.group == g]
        sesoi = gg.sesoi.iloc[0]
        top = y
        ax.text(-lim * 0.995, y - 0.2, "\n".join(textwrap.wrap(g, 95)), fontsize=12.5, fontweight="bold",
                color=INK, va="bottom", ha="left", clip_on=False)
        y += 1.1
        start = y
        for _, r in gg.iterrows():
            c = verdict_color(r.verdict)
            has_ci = np.isfinite(r.ci_lo)
            if has_ci:
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
            ax.scatter([r.improvement], [y], s=95, color=c if has_ci else SURF, edgecolor=c, linewidth=2,
                       zorder=4)
            ax.text(lim * 1.03, y, short_verdict(r.verdict), va="center", ha="left", fontsize=11.5, color=INK2)
            n_txt = f"{int(r.n_a)}v{int(r.n_b)}"
            yticks.append(y)
            ylabels.append(f"{r.model}  ({n_txt})")
            y += 1.0
        ax.axhspan(start - 0.5, y - 0.5, xmin=0, xmax=1, color="none")
        ax.fill_betweenx([start - 0.5, y - 0.5], -sesoi, sesoi, color=GRID, alpha=0.9, zorder=1, lw=0)
        y += 0.9
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(y - 0.5, -1.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11.5)
    ax.grid(axis="y", visible=False)
    sign_txt = "вправо = лучше эталона"
    ax.set_xlabel(f"Δ {METRIC_RU.get(metric, metric)} относительно эталона группы ({sign_txt}); точка = среднее, линия = 95% ДИ, "
                  f"серая полоса = ±SESOI", fontsize=12)
    lvl = "по сидам (Welch)" if level == "seeds" else "по тестовым молекулам (бутстрап)"
    ax.set_title(f"{METRIC_RU.get(metric, metric)}: сравнение с эталоном, уровень — {lvl}", loc="left", pad=16)
    fig.text(0.01, 0.0, "(nA v nB) — число сидов модели и эталона; пустой кружок — тест невозможен (мало сидов); "
             "p-значения скорректированы по Холму внутри группы.", fontsize=10.5, color=INK2, ha="left", va="top")
    save(fig, out_dir, f"1_forest_{metric}_{level}")


# ------------------------------------------------------------------ 2. seeds
def seeds_strip(G, metric, out_dir, ref=None, sesoi=None, min_epochs=30):
    labs = [l for l in G if min(r["epochs"] for r in G[l]) >= min_epochs and any(metric in r["metrics"] for r in G[l])]
    hib = HIGHER_IS_BETTER.get(metric, True)
    means = {l: np.mean([r["metrics"][metric] for r in G[l]]) for l in labs}
    labs.sort(key=lambda l: means[l], reverse=hib)  # лучший сверху (ось y инвертирована)
    fig, ax = plt.subplots(figsize=(12, 0.36 * len(labs) + 2.2))
    if ref in G and sesoi:
        m = means[ref]
        ax.axvspan(m - sesoi, m + sesoi, color=GRID, zorder=1, lw=0)
        ax.axvline(m, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.plot([], [], color=INK2, lw=1, ls=(0, (4, 3)), label=f"эталон: {ref} (серая полоса = ±SESOI)")
    for i, l in enumerate(labs):
        v = np.array([r["metrics"][metric] for r in G[l]])
        c = BLUE if G[l][0]["loss"] == "mse" else ORANGE
        single = len(v) == 1
        ax.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)) * (len(v) > 1), s=46, color=SURF if single else c,
                   edgecolor=c, linewidth=1.6, zorder=3, alpha=0.95)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2.2, zorder=4)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"{l}  ({len(G[l])})" for l in labs], fontsize=11.5)
    ax.set_ylim(len(labs) - 0.4, -0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{METRIC_RU.get(metric, metric)}  (test)   — точка = сид, чёрта = среднее, пустой кружок = 1 сид")
    ax.set_title(f"{METRIC_RU.get(metric, metric)} по сидам — где разброс сопоставим с разницей между моделями", loc="left", pad=14)
    ax.scatter([], [], color=BLUE, label="MSE loss")
    ax.scatter([], [], color=ORANGE, label="physics / v2 loss")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    save(fig, out_dir, f"2_seeds_{metric}")


# ------------------------------------------------------------------ 3. tradeoff
def pareto(points):
    """points: [(x=polar (min), y=r2 (max), label)] -> метки на фронте."""
    front = []
    for i, (x, y, _) in enumerate(points):
        if not any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for j, (x2, y2, _) in enumerate(points) if j != i):
            front.append(i)
    return front


def tradeoff(G, out_dir, highlight, min_epochs=30):
    labs = [l for l in G if min(r["epochs"] for r in G[l]) >= min_epochs]
    pts = [(np.mean([r["metrics"]["polar_mae"] for r in G[l]]), np.mean([r["metrics"]["weighted_r2"] for r in G[l]]), l) for l in labs]
    front = pareto(pts)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for i, (x, y, l) in enumerate(pts):
        c = BLUE if G[l][0]["loss"] == "mse" else ORANGE
        single = len(G[l]) == 1
        ax.scatter(x, y, s=120 if i in front or l in highlight else 70, color=SURF if single else c, edgecolor=c, linewidth=2,
                   zorder=3 + (i in front))
    fr = sorted(front, key=lambda i: pts[i][0])
    ax.step([pts[i][0] for i in fr], [pts[i][1] for i in fr], where="post", color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    lab_set = set(fr) | {i for i, p in enumerate(pts) if p[2] in highlight}
    xmid = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
    slots = [18, -26, 46, -54, 74, -82]          # разные высоты подписей, чтобы не налезали друг на друга
    for k, i in enumerate(sorted(lab_set, key=lambda i: -pts[i][1])):
        x, y, l = pts[i]
        right_side = x < xmid                    # ось x инвертирована: малый polar MAE = правая половина
        ax.annotate(l, (x, y), xytext=(-12 if right_side else 12, slots[k % len(slots)]), textcoords="offset points",
                    fontsize=11, color=INK, ha="right" if right_side else "left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.invert_xaxis()
    ax.set_xlabel("polar MAE (ниже = лучше)  →  лучше вправо")
    ax.set_ylabel("weighted R² (выше = лучше)")
    ax.set_title("weighted R² против polar MAE: пунктир — фронт компромисса (недоминируемые модели)", loc="left", pad=14)
    ax.scatter([], [], color=BLUE, s=70, label="MSE loss")
    ax.scatter([], [], color=ORANGE, s=70, label="physics / v2 loss")
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    save(fig, out_dir, "3_tradeoff")


# ------------------------------------------------------------------ 4. scaling
def scaling(runs, out_dir, min_epochs=30):
    import re
    fam = {}
    for r in runs:
        if r["epochs"] < min_epochs:
            continue
        fam.setdefault(re.sub(r"_(\d+(?:\.\d+)?x)$", "", r["prefix"]), {}).setdefault(r["n_params"], []).append(r)
    fam = {k: v for k, v in fam.items() if len(v) >= 3 and "xtb" not in k}
    cols = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for (name, d), c in zip(sorted(fam.items()), cols):
        ns = sorted(d)
        m = np.array([np.mean([r["metrics"]["weighted_r2"] for r in d[n]]) for n in ns])
        s = np.array([np.std([r["metrics"]["weighted_r2"] for r in d[n]]) for n in ns])
        ax.plot(ns, m, color=c, lw=2.4, zorder=3)
        ax.errorbar(ns, m, yerr=s, fmt="none", ecolor=c, elinewidth=1.6, capsize=4, zorder=3)
        for n, mm in zip(ns, m):
            early = np.mean([r["epochs"] for r in d[n]]) < 150
            ax.scatter(n, mm, s=95, color=SURF if early else c, edgecolor=c, linewidth=2.2, zorder=4)
        ax.annotate(name, (ns[-1], m[-1]), xytext=(10, 0), textcoords="offset points", fontsize=12, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    ax.set_xlabel("число параметров (лог. шкала)")
    ax.set_ylabel("weighted R² (test), среднее ± std по сидам")
    ax.set_title("weighted R² от числа параметров; пустой кружок — обучение остановилось раньше 150 эпох", loc="left", pad=14)
    save(fig, out_dir, "4_scaling")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="папки metrics (как для stats_compare)")
    ap.add_argument("--stats-csv", default=None, help="CSV из stats_compare.py --out-csv (для forest)")
    ap.add_argument("--out-dir", default="results/figs")
    ap.add_argument("--metric", default="weighted_r2")
    ap.add_argument("--forest-metrics", nargs="*", default=None, help="по умолчанию = --metric")
    ap.add_argument("--levels", nargs="+", default=["seeds"], help="seeds и/или molecules")
    ap.add_argument("--only-groups", nargs="*", default=None, help="подстроки названий групп (например Q1 Q4)")
    ap.add_argument("--ref", default="dimenet_pp_enhanced", help="эталон для полосы SESOI на рис. 2")
    ap.add_argument("--sesoi", type=float, default=1e-3)
    ap.add_argument("--highlight", nargs="*", default=[])
    ap.add_argument("--min-epochs", type=int, default=30)
    a = ap.parse_args()

    setup()
    runs = load_runs(a.dirs)
    G = by_label(runs)
    print(f"прогонов: {len(runs)}, моделей: {len(G)}")
    short = [l for l in G if min(r["epochs"] for r in G[l]) < a.min_epochs]
    if short:
        print(f"[note] исключены из рисунков 2-4 как короткие (<{a.min_epochs} эпох): {short}")
    if a.stats_csv:
        df = pd.read_csv(a.stats_csv)
        if a.only_groups:
            df = df[df.group.apply(lambda g: any(s in g for s in a.only_groups))]
        for m in (a.forest_metrics or [a.metric]):
            for lv in a.levels:
                forest(df, m, lv, a.out_dir)
    seeds_strip(G, a.metric, a.out_dir, ref=a.ref, sesoi=a.sesoi, min_epochs=a.min_epochs)
    tradeoff(G, a.out_dir, set(a.highlight) | {a.ref}, a.min_epochs)
    scaling(runs, a.out_dir, a.min_epochs)


if __name__ == "__main__":
    main()
