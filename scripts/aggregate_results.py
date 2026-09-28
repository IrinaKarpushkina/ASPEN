"""
aggregate_results.py — собирает results/**/metrics/*.json в итоговую таблицу
(mean +/- std по seed-ам) для статьи / Supplementary.

Запуск:
    python -m scripts.aggregate_results --metrics-dir results/metrics --csv results/comparison_table_2d.csv
    python -m scripts.aggregate_results --metrics-dir results/metrics --sort emd_raw

ОГРАНИЧЕНИЯ ПО LOSS УБРАНО НАМЕРЕННО:
    Берутся ВСЕ *.json в --metrics-dir. Loss-метка читается из суффикса имени
    файла (часть после последнего '_', напр. dimenet_pp_physics_seed0_physics.json
    -> loss='physics'). Ключ группировки — (model, loss), поэтому прогоны с
    разными loss живут в разных строках и не смешиваются. Если хочется
    отфильтровать, используйте --loss physics (необязательный фильтр).

Про provenance (см. PROVENANCE.md, "Как был найден баг #1"):
    --metrics-dir обязателен (нет дефолта). Перед агрегацией скрипт проверяет
    поле "mode" во всех json; при смешении режимов падает, если не передан
    --allow-mixed-mode.
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


def loss_from_filename(path: str) -> str:
    """Метка loss из имени файла: часть после последнего '_' в stem.

    dimenet_pp_physics_seed0_physics.json -> 'physics'
    dimenet_pp_enhanced_seed0_mse.json    -> 'mse'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[-1] if "_" in stem else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir", required=True,
        help="Папка с json-файлами результатов. НЕТ значения по умолчанию "
             "намеренно — см. docstring этого файла.",
    )
    parser.add_argument(
        "--loss", default=None,
        help="Опциональный фильтр по loss-метке из имени файла. "
             "Если не задан — агрегируются ВСЕ файлы в папке.",
    )
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

    # Ключ группировки: (model, loss_tag) — прогоны с разными loss-метками
    # не смешиваются даже если model совпадает.
    by_key = defaultdict(list)
    modes_seen = set()
    losses_seen = set()

    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "*.json"))):
        loss_tag = loss_from_filename(path)
        if args.loss and loss_tag != args.loss:
            continue
        with open(path) as f:
            r = json.load(f)
        by_key[(r["model"], loss_tag)].append(r)
        modes_seen.add(r.get("mode", "<missing>"))
        losses_seen.add(loss_tag)

    if not by_key:
        print(f"No results found in {args.metrics_dir}"
              + (f" for loss='{args.loss}'" if args.loss else ""))
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
    for (model, loss_tag), runs in sorted(by_key.items()):
        seeds = sorted(r["seed"] for r in runs)
        row = {
            "model": model,
            "loss": loss_tag,
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

    sep = "-" * 136
    print(sep)
    print(f"  Benchmark results  |  dir: {args.metrics_dir}  |  loss filter: {args.loss or 'ALL'}  |  "
          f"loss tag(s): {sorted(losses_seen)}  |  mode(s): {modes_seen}")
    print(f"  * = best mean for this metric")
    print(sep)

    col_w = 22
    header = f"{'model':14s} {'loss':>10s} {'n_params':>9s} {'seeds':>8s}"
    for m in ALL_METRICS:
        header += f"  {m:>{col_w}s}"
    print(header)
    print(sep)

    for row in rows:
        line = (f"{row['model']:14s} {row['loss']:>10s} "
                f"{row['n_params']:>9,} {str(row['seeds']):>8s}")
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
                counts[(row["model"], row["loss"])] += 1
    for (model, loss_tag), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"  {model}@{loss_tag:14s}: {cnt:2d}/{len(ALL_METRICS)}  {bar}")

    print()

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fieldnames = (
            ["model", "loss", "mode", "n_params", "n_seeds", "seeds", "best_epoch_mean"]
            + [f"{m}_mean" for m in ALL_METRICS]
            + [f"{m}_std" for m in ALL_METRICS]
        )
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved:   {args.csv}")

        tex_path = args.csv.replace(".csv", ".tex")
        _write_latex(rows, ALL_METRICS, best_per_metric, tex_path)
        print(f"LaTeX saved: {tex_path}")


def _write_latex(rows, metrics, best_per_metric, path):
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
        (r"\caption{Controlled comparison of GNN architectures "
         r"(mean\,$\pm$\,std over seeds, matched parameter budget). "
         r"Loss tag is parsed from the filename suffix. "
         r"\textbf{Bold} = best mean per metric.}"),
        r"\label{tab:benchmark}",
        r"\small",
        r"\begin{tabular}{l l r " + "c " * len(metrics) + r"}",
        r"\toprule",
        "Model & Loss & $N_{\\text{params}}$ & " + " & ".join(short[m] for m in metrics) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [row["model"].upper(), row["loss"], f"{row['n_params']:,}"]
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
