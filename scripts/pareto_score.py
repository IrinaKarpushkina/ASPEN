"""Фронт Парето + скор в единицах SESOI для моделей из metrics-json.

    python -m scripts.pareto_score \
        --dirs results/all_pred_ablation_experiments_results/metrics results/xtb/metrics \
        --metrics weighted_r2 polar_mae emd_raw --ref dimenet_pp_enhanced --out-dir results/pareto \
        --highlight dimenet_pp_enhanced_xtb sigma_dn_geom dimenet_pp_delta

Что считается
  1. Фронт Парето по выбранным метрикам (по умолчанию R², polar MAE, EMD), с допуском на шум: модель A доминирует B,
     если A не хуже B ни по одной метрике более чем на SESOI и лучше хотя бы по одной более чем на SESOI.
     Слои: слой 1 = недоминируемые, затем убираем их и ищем слой 2 и т. д. (--strict: допуск 0, обычное доминирование).
  2. Скор в единицах SESOI относительно эталона --ref: для каждой метрики (лучше эталона = плюс)
        вклад_m = (качество_модели - качество_эталона) / SESOI_m ,   скор = взвешенное среднее вкладов.
     1.0 означает «в среднем на один порог практической значимости лучше эталона». Смена эталона сдвигает все скоры на
     одну константу и порядок моделей НЕ меняет.
Модели с < --min-epochs эпох или < --min-seeds сидов не участвуют.
Выход: pareto_table.csv/.md, pareto_<m1>_vs_<m2>.png/.svg, score_sesoi.png/.svg
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._runs import HIGHER_IS_BETTER, by_label, load_runs  # noqa: E402
from scripts import plot_stats as P  # noqa: E402

plt = P.plt
DEFAULT_SESOI = {"weighted_r2": 1e-3, "weighted_mae": 7e-4, "polar_mae": 3e-4, "emd_raw": 3e-5,
                 "cosine_similarity": 1e-3, "molecular_mae": 5e-3}
METRIC_COLORS = [P.BLUE, P.ORANGE, P.AQUA, "#8a5cf6", "#c9a227"]


def pareto_layers(G, tol):
    """G: (n_models, n_metrics) качество (больше = лучше); tol: (n_metrics,) допуск. Возвращает слой каждой модели (1 = фронт)."""
    n = len(G)
    layer = np.zeros(n, dtype=int)
    left, k = set(range(n)), 1
    while left:
        front = []
        for i in left:
            dominated = any(np.all(G[j] >= G[i] - tol) and np.any(G[j] > G[i] + tol) for j in left if j != i)
            if not dominated:
                front.append(i)
        if not front:                      # защита от циклов при нетранзитивном допуске
            front = list(left)
        for i in front:
            layer[i] = k
        left -= set(front)
        k += 1
    return layer


def scatter_pair(df, mx, my, ref, sesoi, highlight, out_dir):
    hx, hy = HIGHER_IS_BETTER.get(mx, True), HIGHER_IS_BETTER.get(my, True)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    if ref in df.index:                    # серый прямоугольник = шум вокруг эталона (точки внутри неотличимы от него)
        rx, ry = df.loc[ref, mx], df.loc[ref, my]
        ax.add_patch(plt.Rectangle((rx - sesoi[mx], ry - sesoi[my]), 2 * sesoi[mx], 2 * sesoi[my],
                                   color=P.GRID, alpha=0.9, zorder=1, lw=0))
    for lab, r in df.iterrows():
        c = P.BLUE if r["loss"] == "mse" else P.ORANGE
        top = r["layer"] == 1
        ax.scatter(r[mx], r[my], s=130 if top else 60, color=c, alpha=1 if top else 0.35, edgecolor=c, linewidth=1.5,
                   zorder=4 if top else 3)
    # строгий 2D-фронт этой пары (ступенька)
    gx = df[mx] * (1 if hx else -1)
    gy = df[my] * (1 if hy else -1)
    idx = [i for i in df.index if not any((gx[j] >= gx[i] and gy[j] >= gy[i]) and (gx[j] > gx[i] or gy[j] > gy[i])
                                          for j in df.index if j != i)]
    idx = sorted(idx, key=lambda i: -gx[i])
    xs, ys = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        xs += [df.loc[a, mx], df.loc[b, mx]]
        ys += [df.loc[a, my], df.loc[a, my]]
        xs.append(df.loc[b, mx]); ys.append(df.loc[b, my])
    if xs:
        ax.plot([df.loc[idx[0], mx]] + xs, [df.loc[idx[0], my]] + ys, color=P.INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    lab_set = [l for l in df.index if df.loc[l, "layer"] == 1 or l in highlight or l == ref]
    xmid = (df[mx].min() + df[mx].max()) / 2
    slots = [18, -26, 46, -54, 74, -82, 102, -110]
    for k, l in enumerate(sorted(lab_set, key=lambda l: -(df.loc[l, my] if hy else -df.loc[l, my]))):
        x, y = df.loc[l, mx], df.loc[l, my]
        right = (x < xmid) == hx               # подписываем в сторону, где больше места
        ax.annotate(l, (x, y), xytext=(-12 if right else 12, slots[k % len(slots)]), textcoords="offset points",
                    fontsize=10.5, color=P.INK, ha="right" if right else "left",
                    arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.8))
    if not hx:
        ax.invert_xaxis()
    if not hy:
        ax.invert_yaxis()
    ru = P.METRIC_RU
    ax.set_xlabel(f"{ru.get(mx, mx)} ({'выше' if hx else 'ниже'} = лучше)  →  лучше вправо")
    ax.set_ylabel(f"{ru.get(my, my)} ({'выше' if hy else 'ниже'} = лучше)  ↑  лучше вверх")
    ax.set_title(f"{ru.get(my, my)} против {ru.get(mx, mx)}", loc="left", pad=26)
    ax.text(0, 1.015, "яркие точки — слой 1 Парето по всем метрикам; пунктир — фронт этой пары; серый квадрат — ±SESOI вокруг эталона",
            transform=ax.transAxes, fontsize=10.5, color=P.INK2)
    ax.scatter([], [], color=P.BLUE, s=70, label="MSE loss")
    ax.scatter([], [], color=P.ORANGE, s=70, label="physics / v2 loss")
    ax.legend(loc="best", frameon=False, fontsize=11)
    P.save(fig, out_dir, f"pareto_{mx}_vs_{my}")


def score_bars(df, metrics, w, out_dir, ref):
    d = df.sort_values("score", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(d) + 2.2))
    y = np.arange(len(d))
    pos0, neg0 = np.zeros(len(d)), np.zeros(len(d))
    for k, m in enumerate(metrics):
        v = (w[m] * d[f"d_{m}"] / sum(w.values())).values
        left = np.where(v >= 0, pos0, neg0)
        ax.barh(y, v, left=left, color=METRIC_COLORS[k % len(METRIC_COLORS)], height=0.7, label=P.METRIC_RU.get(m, m))
        pos0 = pos0 + np.where(v >= 0, v, 0)
        neg0 = neg0 + np.where(v < 0, v, 0)
    ax.scatter(d["score"], y, color=P.INK, marker="|", s=260, zorder=5, label="итоговый скор")
    ax.axvline(0, color=P.INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([("★ " if l1 == 1 else "") + l for l, l1 in zip(d.index, d["layer"])])
    ax.set_xlabel(f"скор в единицах SESOI относительно «{ref}» (вправо = лучше); ★ = слой 1 Парето; черта = итог")
    ax.set_title("Скор в единицах SESOI: сколько порогов практической значимости модель выигрывает у эталона",
                 loc="left", pad=12, fontsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    P.save(fig, out_dir, "score_sesoi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", default=["weighted_r2", "polar_mae", "emd_raw"])
    ap.add_argument("--sesoi", nargs="*", default=[], help="метрика=значение, переопределяет пороги (напр. polar_mae=0.0003)")
    ap.add_argument("--weights", nargs="*", default=[], help="метрика=вес для скора (по умолчанию все 1)")
    ap.add_argument("--ref", default="dimenet_pp_enhanced", help="эталон для скора (метка модели, как в stats_compare --list)")
    ap.add_argument("--strict", action="store_true", help="Парето без допуска на SESOI")
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--highlight", nargs="*", default=[])
    ap.add_argument("--out-dir", default="results/pareto")
    a = ap.parse_args()

    sesoi = dict(DEFAULT_SESOI)
    sesoi.update({k: float(v) for k, v in (s.split("=") for s in a.sesoi)})
    w = {m: 1.0 for m in a.metrics}
    w.update({k: float(v) for k, v in (s.split("=") for s in a.weights)})
    for m in a.metrics:
        if m not in sesoi:
            sys.exit(f"нет порога SESOI для {m}: задайте --sesoi {m}=...")

    G = by_label(load_runs(a.dirs))
    rows = []
    for lab, rr in G.items():
        if min(r["epochs"] for r in rr) < a.min_epochs or len(rr) < a.min_seeds:
            print(f"[skip] {lab}: эпох {min(r['epochs'] for r in rr)}, сидов {len(rr)}")
            continue
        if not all(m in r["metrics"] for r in rr for m in a.metrics):
            print(f"[skip] {lab}: нет всех метрик"); continue
        row = dict(label=lab, loss=rr[0]["loss"], n_params=rr[0]["n_params"], n_seeds=len(rr))
        for m in a.metrics:
            row[m] = float(np.mean([r["metrics"][m] for r in rr]))
            row[m + "_std"] = float(np.std([r["metrics"][m] for r in rr]))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("label")
    if a.ref not in df.index:
        sys.exit(f"эталон '{a.ref}' не найден среди: {', '.join(df.index)}")

    sign = np.array([1.0 if HIGHER_IS_BETTER.get(m, True) else -1.0 for m in a.metrics])
    Q = df[a.metrics].values * sign                          # качество: больше = лучше
    tol = np.zeros(len(a.metrics)) if a.strict else np.array([sesoi[m] for m in a.metrics])
    df["layer"] = pareto_layers(Q, tol)

    qref = df.loc[a.ref, a.metrics].values * sign
    for k, m in enumerate(a.metrics):
        df[f"d_{m}"] = (Q[:, k] - qref[k]) / sesoi[m]
    df["score"] = sum(w[m] * df[f"d_{m}"] for m in a.metrics) / sum(w.values())
    df = df.sort_values(["layer", "score"], ascending=[True, False])

    cols = ["loss", "n_params", "n_seeds", "layer", "score"] + a.metrics + [f"d_{m}" for m in a.metrics]
    os.makedirs(a.out_dir, exist_ok=True)
    df[cols].to_csv(os.path.join(a.out_dir, "pareto_table.csv"))
    with open(os.path.join(a.out_dir, "pareto_table.md"), "w") as f:
        hdr = ["модель", "слой", "скор"] + [f"{m} (Δ/SESOI)" for m in a.metrics] + ["параметры", "сидов"]
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for lab, r in df.iterrows():
            f.write(f"| {lab} | {r['layer']} | {r['score']:+.2f} | "
                    + " | ".join(f"{r[m]:.5f} ({r['d_' + m]:+.2f})" for m in a.metrics)
                    + f" | {int(r['n_params']):,} | {r['n_seeds']} |\n")

    print(f"\nэталон: {a.ref}; допуск Парето: {'нет (strict)' if a.strict else 'SESOI'}; метрики: {', '.join(a.metrics)}\n")
    print(f"{'модель':34s} слой  скор   " + "  ".join(f"{m[:12]:>12s}" for m in a.metrics))
    for lab, r in df.iterrows():
        print(f"{lab:34s} {r['layer']:3d} {r['score']:+6.2f}   " + "  ".join(f"{r['d_' + m]:+12.2f}" for m in a.metrics))

    P.setup()
    for mx, my in itertools.combinations(a.metrics, 2):
        scatter_pair(df, mx, my, a.ref, sesoi, a.highlight, a.out_dir)
    score_bars(df, a.metrics, w, a.out_dir, a.ref)
    print(f"\nфайлы в {a.out_dir}")


if __name__ == "__main__":
    main()
