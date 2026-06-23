"""
aggregate_results.py — собирает results/metrics/*.json в итоговую таблицу
(mean ± std по seed-ам) для статьи / Supplementary.

Запуск:
    python -m scripts.aggregate_results
    python -m scripts.aggregate_results --loss mse
    python -m scripts.aggregate_results --csv results/comparison_table.csv
    python -m scripts.aggregate_results --sort emd_raw   # сортировка по метрике
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Все метрики из test_metrics (в нужном порядке для таблицы)
ALL_METRICS = [
    "weighted_r2",
    "weighted_mae",
    "polar_mae",
    "cosine_similarity",
    "emd_raw",
    "emd_normalized",
    "molecular_mae",
    "molecular_emd",
    "molecular_cosine",
    "molecular_polar_mae",
]

# Для каждой метрики: True = выше лучше, False = ниже лучше
HIGHER_IS_BETTER = {
    "weighted_r2":         True,
    "weighted_mae":        False,
    "polar_mae":           False,
    "cosine_similarity":   True,
    "emd_raw":             False,
    "emd_normalized":      False,
    "molecular_mae":       False,
    "molecular_emd":       False,
    "molecular_cosine":    True,
    "molecular_polar_mae": False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--sort", default=None,
        help="Метрика для сортировки строк (напр. emd_raw, weighted_mae)",
    )
    args = parser.parse_args()

    # ── Загрузка json-ов ─────────────────────────────────────────────────────
    by_model = defaultdict(list)
    for path in sorted(glob.glob(
            os.path.join(args.metrics_dir, f"*_{args.loss}.json"))):
        with open(path) as f:
            r = json.load(f)
        by_model[r["model"]].append(r)

    if not by_model:
        print(f"No results found for loss='{args.loss}' in {args.metrics_dir}")
        return

    # ── Агрегация ─────────────────────────────────────────────────────────────
    rows = []
    for model, runs in sorted(by_model.items()):
        seeds = sorted(r["seed"] for r in runs)
        row = {
            "model":            model,
            "n_seeds":          len(runs),
            "seeds":            str(seeds),
            "n_params":         runs[0]["n_params"],
            "best_epoch_mean":  float(np.mean([r["best_epoch"] for r in runs])),
        }
        for metric in ALL_METRICS:
            vals = [r["test_metrics"][metric] for r in runs]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"]  = float(np.std(vals))
        rows.append(row)

    if not rows:
        print("No rows to display.")
        return

    # ── Сортировка ────────────────────────────────────────────────────────────
    if args.sort and args.sort in ALL_METRICS:
        reverse = HIGHER_IS_BETTER[args.sort]
        rows.sort(key=lambda r: r[f"{args.sort}_mean"], reverse=reverse)

    # Лучшее значение по каждой метрике (для маркировки *)
    best_per_metric = {}
    for metric in ALL_METRICS:
        all_means = [r[f"{metric}_mean"] for r in rows]
        best_per_metric[metric] = (
            max(all_means) if HIGHER_IS_BETTER[metric] else min(all_means)
        )

    # ── Вывод в терминал ─────────────────────────────────────────────────────
    sep = "-" * 130
    print(sep)
    print(f"  Benchmark results  |  loss={args.loss}  |  dir: {args.metrics_dir}")
    print(f"  * = best mean for this metric")
    print(sep)

    col_w = 22
    header = f"{'model':10s} {'n_params':>9s} {'seeds':>8s}"
    for m in ALL_METRICS:
        header += f"  {m:>{col_w}s}"
    print(header)
    print(sep)

    for row in rows:
        line = (f"{row['model']:10s} {row['n_params']:>9,} "
                f"{str(row['seeds']):>8s}")
        for metric in ALL_METRICS:
            mean = row[f"{metric}_mean"]
            std  = row[f"{metric}_std"]
            mark = "*" if abs(mean - best_per_metric[metric]) < 1e-9 else " "
            cell = f"{mean:.5f}±{std:.5f}{mark}"
            line += f"  {cell:>{col_w}s}"
        print(line)

    print(sep)

    # Сводка: сколько раз каждая модель лучшая
    print("\nBest-metric count (* = best mean across models):")
    counts = defaultdict(int)
    for metric in ALL_METRICS:
        for row in rows:
            if abs(row[f"{metric}_mean"] - best_per_metric[metric]) < 1e-9:
                counts[row["model"]] += 1
    for model, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"  {model:10s}: {cnt:2d}/{len(ALL_METRICS)}  {bar}")

    print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fieldnames = (
            ["model", "loss", "n_params", "n_seeds", "seeds", "best_epoch_mean"]
            + [f"{m}_mean" for m in ALL_METRICS]
            + [f"{m}_std"  for m in ALL_METRICS]
        )
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row["loss"] = args.loss
                writer.writerow(row)
        print(f"CSV saved:   {args.csv}")

        # LaTeX-таблица рядом с CSV
        tex_path = args.csv.replace(".csv", ".tex")
        _write_latex(rows, ALL_METRICS, best_per_metric, args.loss, tex_path)
        print(f"LaTeX saved: {tex_path}")


def _write_latex(rows, metrics, best_per_metric, loss, path):
    """
    Генерирует booktabs LaTeX-таблицу для вставки в статью.
    Лучшие значения выделены \textbf{}.
    """
    short = {
        "weighted_r2":         r"$R^2_w$",
        "weighted_mae":        r"wMAE",
        "polar_mae":           r"PolarMAE",
        "cosine_similarity":   r"Cosine",
        "emd_raw":             r"EMD",
        "emd_normalized":      r"EMD$_\mathrm{norm}$",
        "molecular_mae":       r"Mol.MAE",
        "molecular_emd":       r"Mol.EMD",
        "molecular_cosine":    r"Mol.Cos",
        "molecular_polar_mae": r"Mol.PolMAE",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Controlled comparison of GNN architectures "
         r"(mean\,$\pm$\,std over 3 seeds, MSE loss, "
         r"$\approx$695k parameters per model). "
         r"\textbf{Bold} = best mean per metric.}"),
        r"\label{tab:benchmark_" + loss + r"}",
        r"\small",
        r"\begin{tabular}{l r " + "c " * len(metrics) + r"}",
        r"\toprule",
        "Model & $N_{\\text{params}}$ & "
        + " & ".join(short[m] for m in metrics) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [row["model"].upper(), f"{row['n_params']:,}"]
        for metric in metrics:
            mean = row[f"{metric}_mean"]
            std  = row[f"{metric}_std"]
            is_best = abs(mean - best_per_metric[metric]) < 1e-9
            cell = f"{mean:.4f}$\\pm${std:.4f}"
            if is_best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
