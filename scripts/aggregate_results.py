"""
aggregate_results.py — собирает results/metrics/*.json в итоговую таблицу
(mean +/- std по seed-ам) для статьи / Supplementary.

Запуск:
    python -m scripts.aggregate_results --metrics-dir results/metrics --csv results/comparison_table_2d.csv
    python -m scripts.aggregate_results --metrics-dir results/metrics --sort emd_raw

ВАЖНО (см. PROVENANCE.md, "Как был найден баг #1"): в предыдущей версии
этого репозитория --metrics-dir имел значение по умолчанию, из-за чего
однажды была случайно сгенерирована "2D"-таблица, фактически содержащая
3D-данные — ошибка осталась незамеченной несколько недель, потому что
ничто не сверяло, откуда реально взяты json-файлы.

Чтобы это не могло повториться:
  1. --metrics-dir теперь ОБЯЗАТЕЛЬНЫЙ аргумент (нет дефолта).
  2. Перед агрегацией скрипт проверяет поле "mode" во всех json (train.py
     теперь всегда его записывает) и падает с ошибкой, если в одной папке
     смешаны результаты разных режимов.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

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

HIGHER_IS_BETTER = {
    "weighted_r2": True,
    "weighted_mae": False,
    "polar_mae": False,
    "cosine_similarity": True,
    "emd_raw": False,
    "emd_normalized": False,
    "molecular_mae": False,
    "molecular_emd": False,
    "molecular_cosine": True,
    "molecular_polar_mae": False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir", required=True,
        help="Папка с *_<loss>.json файлами. НЕТ значения по умолчанию "
             "намеренно — см. docstring этого файла.",
    )
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--sort", default=None,
                         help="Метрика для сортировки строк (напр. emd_raw, weighted_mae)")
    parser.add_argument(
        "--allow-mixed-mode", action="store_true",
        help="Разрешить агрегацию, даже если json-файлы содержат разные "
             "значения поля 'mode' (по умолчанию запрещено — защита от "
             "повторения бага #1, см. PROVENANCE.md).",
    )
    args = parser.parse_args()

    by_model = defaultdict(list)
    modes_seen = set()
    for path in sorted(glob.glob(os.path.join(args.metrics_dir, f"*_{args.loss}.json"))):
        with open(path) as f:
            r = json.load(f)
        by_model[r["model"]].append(r)
        modes_seen.add(r.get("mode", "<missing>"))

    if not by_model:
        print(f"No results found for loss='{args.loss}' in {args.metrics_dir}")
        return

    if len(modes_seen) > 1 and not args.allow_mixed_mode:
        raise RuntimeError(
            f"Found result files from DIFFERENT modes in the same directory: "
            f"{modes_seen}. This is exactly the kind of mix-up that caused "
            f"a previous silent bug (see PROVENANCE.md). If this is really "
            f"intentional, pass --allow-mixed-mode."
        )
    if "<missing>" in modes_seen:
        print("WARNING: some result files have no 'mode' field (produced by "
              "an older train.py?). Cannot verify provenance for those files.")

    rows = []
    for model, runs in sorted(by_model.items()):
        seeds = sorted(r["seed"] for r in runs)
        row = {
            "model": model,
            "mode": next(iter(modes_seen)) if len(modes_seen) == 1 else "mixed",
            "n_seeds": len(runs),
            "seeds": str(seeds),
            "n_params": runs[0]["n_params"],
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in runs])),
        }
        for metric in ALL_METRICS:
            vals = [r["test_metrics"][metric] for r in runs]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)

    if args.sort and args.sort in ALL_METRICS:
        reverse = HIGHER_IS_BETTER[args.sort]
        rows.sort(key=lambda r: r[f"{args.sort}_mean"], reverse=reverse)

    best_per_metric = {}
    for metric in ALL_METRICS:
        all_means = [r[f"{metric}_mean"] for r in rows]
        best_per_metric[metric] = max(all_means) if HIGHER_IS_BETTER[metric] else min(all_means)

    sep = "-" * 130
    print(sep)
    print(f"  Benchmark results (2D)  |  loss={args.loss}  |  dir: {args.metrics_dir}  |  mode(s): {modes_seen}")
    print(f"  * = best mean for this metric")
    print(sep)

    col_w = 22
    header = f"{'model':14s} {'n_params':>9s} {'seeds':>8s}"
    for m in ALL_METRICS:
        header += f"  {m:>{col_w}s}"
    print(header)
    print(sep)

    for row in rows:
        line = f"{row['model']:14s} {row['n_params']:>9,} {str(row['seeds']):>8s}"
        for metric in ALL_METRICS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            mark = "*" if abs(mean - best_per_metric[metric]) < 1e-9 else " "
            cell = f"{mean:.5f}+/-{std:.5f}{mark}"
            line += f"  {cell:>{col_w}s}"
        print(line)

    print(sep)

    print("\nBest-metric count (* = best mean across models):")
    counts = defaultdict(int)
    for metric in ALL_METRICS:
        for row in rows:
            if abs(row[f"{metric}_mean"] - best_per_metric[metric]) < 1e-9:
                counts[row["model"]] += 1
    for model, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"  {model:14s}: {cnt:2d}/{len(ALL_METRICS)}  {bar}")

    print()

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fieldnames = (
            ["model", "mode", "loss", "n_params", "n_seeds", "seeds", "best_epoch_mean"]
            + [f"{m}_mean" for m in ALL_METRICS]
            + [f"{m}_std" for m in ALL_METRICS]
        )
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row["loss"] = args.loss
                writer.writerow(row)
        print(f"CSV saved:   {args.csv}")

        tex_path = args.csv.replace(".csv", ".tex")
        _write_latex(rows, ALL_METRICS, best_per_metric, args.loss, tex_path)
        print(f"LaTeX saved: {tex_path}")


def _write_latex(rows, metrics, best_per_metric, loss, path):
    """Генерирует booktabs LaTeX-таблицу для вставки в статью."""
    short = {
        "weighted_r2": r"$R^2_w$",
        "weighted_mae": r"wMAE",
        "polar_mae": r"PolarMAE",
        "cosine_similarity": r"Cosine",
        "emd_raw": r"EMD",
        "emd_normalized": r"EMD$_\mathrm{norm}$",
        "molecular_mae": r"Mol.MAE",
        "molecular_emd": r"Mol.EMD",
        "molecular_cosine": r"Mol.Cos",
        "molecular_polar_mae": r"Mol.PolMAE",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Controlled 2D-only comparison of GNN architectures "
         r"(mean\,$\pm$\,std over seeds, MSE loss, matched parameter budget). "
         r"\textbf{Bold} = best mean per metric.}"),
        r"\label{tab:benchmark_2d_" + loss + r"}",
        r"\small",
        r"\begin{tabular}{l r " + "c " * len(metrics) + r"}",
        r"\toprule",
        "Model & $N_{\\text{params}}$ & " + " & ".join(short[m] for m in metrics) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [row["model"].upper(), f"{row['n_params']:,}"]
        for metric in metrics:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            is_best = abs(mean - best_per_metric[metric]) < 1e-9
            cell = f"{mean:.4f}$\\pm${std:.4f}"
            if is_best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
