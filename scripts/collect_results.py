"""
collect_results.py -- сводная таблица (mean +/- std по seed) из ЛЮБЫХ папок и ЛЮБЫХ loss.

loss -- обычная колонка таблицы (как n_params), а не условие отбора файлов.

    python -m scripts.collect_results \
        --metrics-dir results/metrics results/physics results/physics_v2 \
        --csv results/all_models.csv

    # только некоторые модели / loss, сортировка по метрике:
    python -m scripts.collect_results --metrics-dir results/physics_v2 \
        --loss physics mse --sort polar_mae --csv results/x.csv
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from scripts.result_loader import add_source_args, apply_filters, load_runs

ALL_METRICS = ["weighted_r2", "weighted_mae", "polar_mae", "cosine_similarity", "emd_raw",
               "emd_normalized", "molecular_mae", "molecular_emd", "molecular_cosine",
               "molecular_polar_mae"]
HIGHER = {"weighted_r2", "cosine_similarity", "molecular_cosine"}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ALL_METRICS if m in df.columns]
    rows = []
    for (model, loss), g in df.groupby(["model", "loss"]):
        row = dict(model=model, loss=loss, mode=",".join(sorted({x for x in g["mode"] if x})),
                   n_params=int(g["n_params"].dropna().iloc[0]) if g["n_params"].notna().any() else None,
                   n_seeds=len(g), seeds=sorted(g["seed"].tolist()),
                   best_epoch_mean=g["best_epoch"].mean(),
                   source=",".join(sorted(set(g["source"]))))
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--csv", default=None, help="Куда сохранить агрегат (рядом пишется .tex)")
    ap.add_argument("--per-run-csv", default=None, help="Опционально: таблица по каждому (model, loss, seed)")
    ap.add_argument("--sort", default=None, choices=ALL_METRICS)
    ap.add_argument("--min-seeds", type=int, default=1, help="Скрыть группы с меньшим числом seed")
    args = ap.parse_args()

    df = apply_filters(load_runs(args.metrics_dir, args.on_duplicate), args.loss, args.models)
    if args.per_run_csv:
        df.drop(columns=["label"]).to_csv(args.per_run_csv, index=False)
        print(f"Per-run CSV saved: {args.per_run_csv}")
    agg, metrics = aggregate(df)
    agg = agg[agg["n_seeds"] >= args.min_seeds]
    if args.sort:
        agg = agg.sort_values(f"{args.sort}_mean", ascending=args.sort not in HIGHER)
    else:
        agg = agg.sort_values(["model", "loss"])
    agg = agg.reset_index(drop=True)

    best = {m: (agg[f"{m}_mean"].max() if m in HIGHER else agg[f"{m}_mean"].min()) for m in metrics}
    print("=" * 150)
    print(f"  {len(agg)} model/loss groups from {len(args.metrics_dir)} location(s)   * = best mean for this metric")
    print("=" * 150)
    print(f"{'model':26s} {'loss':12s} {'n_params':>10s} {'seeds':>9s}  " + "  ".join(f"{m:>22s}" for m in metrics))
    for _, r in agg.iterrows():
        cells = []
        for m in metrics:
            star = "*" if np.isclose(r[f"{m}_mean"], best[m], rtol=0, atol=1e-12) else " "
            cells.append(f"{r[f'{m}_mean']:.5f}+/-{r[f'{m}_std']:.5f}{star}")
        n_par = f"{r['n_params']:,}" if pd.notna(r["n_params"]) else "-"
        print(f"{r['model']:26s} {r['loss']:12s} {n_par:>10s} {str(r['seeds']):>9s}  " + "  ".join(f"{c:>22s}" for c in cells))

    cnt = Counter()
    for m in metrics:
        w = agg.loc[agg[f"{m}_mean"].idxmax() if m in HIGHER else agg[f"{m}_mean"].idxmin()]
        cnt[f"{w['model']}@{w['loss']}"] += 1
    print("\nBest-metric count:")
    for k, v in cnt.most_common():
        print(f"  {k:40s} {v:2d}/{len(metrics)}  {'#' * v}")

    if args.csv:
        agg.assign(seeds=agg["seeds"].astype(str)).to_csv(args.csv, index=False)
        print(f"\nCSV saved: {args.csv}")
        tex = args.csv.replace(".csv", ".tex")
        with open(tex, "w") as f:
            f.write("\\begin{tabular}{llr" + "c" * len(metrics) + "}\n\\toprule\nModel & Loss & $N$ & "
                    + " & ".join(m.replace("_", r"\_") for m in metrics) + " \\\\\n\\midrule\n")
            for _, r in agg.iterrows():
                cells = []
                for m in metrics:
                    c = f"{r[f'{m}_mean']:.4f} $\\pm$ {r[f'{m}_std']:.4f}"
                    if np.isclose(r[f"{m}_mean"], best[m], rtol=0, atol=1e-12):
                        c = "\\textbf{" + c + "}"
                    cells.append(c)
                n_par = f"{int(r['n_params']):,}" if pd.notna(r["n_params"]) else "-"
                f.write(f"{r['model'].replace('_', chr(92) + '_')} & {r['loss']} & {n_par} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"LaTeX saved: {tex}")


if __name__ == "__main__":
    main()
