"""Единая таблица по ВСЕМ папкам metrics, без смешивания размеров моделей.

    python -m scripts.merge_all_results \
        --dirs results/all_pred_ablation_experiments_results/metrics results/xtb/metrics results/xtb_base_sched/metrics \
        --csv results/comparison_all.csv --md results/comparison_all.md

Чем отличается от scripts/aggregate_results.py:
  * строка = (model, loss, ВАРИАНТ РАЗМЕРА), а не только (model, loss). Вариант берётся из имени файла
    (..._0.5x_seed0_..., ..._2x_..., ..._4x_...; без метки = 1x). У вас конфиги scaling/*.yaml имеют то же поле
    model.name, что и базовые, поэтому aggregate_results.py складывает 0.5x/1x/2x/4x в одну строку и показывает
    n_params первого файла. Если внутри группы всё равно встретятся разные n_params - она автоматически делится по ним.
  * std не печатается при одном сиде (вместо "+/-0.00000"); звёздочка "лучший" ставится только среди строк с >= --min-seeds сидами.
  * помечает прогоны, где эпох меньше --min-epochs (похоже на smoke/обрыв) и дубликаты (model, loss, variant, seed).
  * добавляет колонки ресурсов, если они есть в json: время обучения, пик памяти GPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = ["weighted_r2", "weighted_mae", "polar_mae", "cosine_similarity", "emd_raw", "emd_normalized",
           "molecular_mae", "molecular_emd", "molecular_cosine", "molecular_polar_mae"]
HIGHER = {"weighted_r2": True, "cosine_similarity": True, "molecular_cosine": True}
SHOW = ["weighted_r2", "weighted_mae", "polar_mae", "emd_raw", "molecular_mae"]
VARIANT_RE = re.compile(r"_(\d+(?:\.\d+)?x)_seed\d+_")


def parse(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = VARIANT_RE.search(stem + "_")
    variant = m.group(1) if m else "1x"
    loss = stem.rsplit("_", 1)[-1]
    return variant, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="папки metrics (порядок = приоритет при дубликатах)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--min-epochs", type=int, default=30)
    a = ap.parse_args()

    groups, seen, dup, short = defaultdict(list), {}, [], []
    for d in a.dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(path))
                tm = r["test_metrics"]
            except Exception as e:
                print(f"[skip] {path}: {e}")
                continue
            variant, loss = parse(path)
            key = (r["model"], loss, variant)
            sid = key + (r.get("seed"),)
            if sid in seen:
                dup.append((path, seen[sid]))
                continue
            seen[sid] = path
            r["_path"], r["_variant"] = path, variant
            r["_epochs"] = len(r.get("history", []))
            if r["_epochs"] < a.min_epochs:
                short.append((path, r["_epochs"]))
            groups[key].append(r)

    rows = []
    for (model, loss, variant), runs in groups.items():
        by_np = defaultdict(list)
        for r in runs:
            by_np[r.get("n_params")].append(r)
        for npar, rr in by_np.items():
            label = model + ("" if variant == "1x" else f"@{variant}") + (f"[{npar // 1000}k]" if len(by_np) > 1 else "")
            row = dict(model=label, loss=loss, n_params=npar, n_seeds=len({r.get("seed") for r in rr}), n_runs=len(rr),
                       min_epochs=min(r["_epochs"] for r in rr), seeds=sorted(r.get("seed") for r in rr))
            for k in METRICS:
                v = np.array([r["test_metrics"][k] for r in rr if k in r["test_metrics"]], dtype=float)
                row[k + "_mean"] = float(v.mean()) if len(v) else float("nan")
                row[k + "_std"] = float(v.std()) if len(v) > 1 else float("nan")
            t = [r["resources"]["total_train_time_s"] / 3600 for r in rr if r.get("resources", {}).get("total_train_time_s")]
            g = [r["resources"]["peak_gpu_memory_gb"] for r in rr if r.get("resources", {}).get("peak_gpu_memory_gb")]
            row["train_h"] = float(np.mean(t)) if t else float("nan")
            row["peak_gpu_gb"] = float(np.mean(g)) if g else float("nan")
            rows.append(row)
    rows.sort(key=lambda r: -r["weighted_r2_mean"])

    best = {}
    for k in METRICS:
        cand = [r for r in rows if r["n_seeds"] >= a.min_seeds and not np.isnan(r[k + "_mean"])]
        if cand:
            best[k] = (max if HIGHER.get(k) else min)(cand, key=lambda r: r[k + "_mean"])["model"]

    def cell(r, k):
        m, s = r[k + "_mean"], r[k + "_std"]
        txt = f"{m:.5f}" + (f"±{s:.5f}" if not np.isnan(s) else " (1 сид)")
        return txt + ("*" if best.get(k) == r["model"] and r["n_seeds"] >= a.min_seeds else "")

    hdr = ["model", "loss", "n_params", "сидов", "мин.эпох"] + SHOW + ["train_h", "GPU_GB"]
    lines = []
    for r in rows:
        flag = " ⚠" if r["min_epochs"] < a.min_epochs else ""
        lines.append([r["model"] + flag, r["loss"], f"{r['n_params']:,}", str(r["n_seeds"]), str(r["min_epochs"])]
                     + [cell(r, k) for k in SHOW]
                     + ["—" if np.isnan(r["train_h"]) else f"{r['train_h']:.1f}",
                        "—" if np.isnan(r["peak_gpu_gb"]) else f"{r['peak_gpu_gb']:.1f}"])
    w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for l in lines:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(l)))
    print(f"\n* = лучший среди строк с >= {a.min_seeds} сидами; ⚠ = есть прогон с < {a.min_epochs} эпох; (1 сид) = std не определён")
    if short:
        print("\nКороткие прогоны (smoke/обрыв?):")
        for p, n in short:
            print(f"  {n:3d} эпох  {p}")
    if dup:
        print("\nДубликаты (model, loss, вариант, сид) - использован первый по порядку --dirs, остальные пропущены:")
        for p, q in dup:
            print(f"  пропущен {p}  (есть {q})")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            cols = ["model", "loss", "n_params", "n_seeds", "n_runs", "min_epochs", "seeds", "train_h", "peak_gpu_gb"] + \
                   [k + s for k in METRICS for s in ("_mean", "_std")]
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow({c: r[c] for c in cols})
        print(f"CSV: {a.csv}")
    if a.md:
        with open(a.md, "w") as f:
            f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for l in lines:
                f.write("| " + " | ".join(l) + " |\n")
        print(f"MD:  {a.md}")


if __name__ == "__main__":
    main()
