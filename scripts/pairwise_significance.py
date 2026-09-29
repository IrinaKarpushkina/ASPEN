"""
pairwise_significance.py — parametric + non-parametric paired test between
every pair of models, using the per-seed `test_metrics` already saved in
results/metrics/<model>_seed<k>_<loss>.json (no retraining, no extra
compute — this reuses numbers you already have).

IMPORTANT CAVEAT, printed at the top of every run: with only 3 seeds,
statistical power is very low (a paired t-test with n=3 has 2 degrees of
freedom). Treat this as a QUICK SANITY CHECK, not a final answer — see
`scripts/molecule_level_eval.py` for a much more powerful test-set-level
comparison that doesn't depend on the number of seeds.

Usage:
    python -m scripts.pairwise_significance --metrics-dir results/metrics_3d_only \
        --loss mse --metric weighted_r2

    # compare only two specific models:
    python -m scripts.pairwise_significance --metrics-dir results/metrics_3d_only \
        --loss mse --metric weighted_r2 --models dimenet_pp spherenet
"""
import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HIGHER_IS_BETTER = {
    "weighted_r2": True, "cosine_similarity": True, "molecular_cosine": True,
    "weighted_mae": False, "polar_mae": False, "emd_raw": False,
    "emd_normalized": False, "molecular_mae": False, "molecular_emd": False,
    "molecular_polar_mae": False, "loss": False,
}


def load_seed_values(metrics_dir: str, loss: str, metric: str, group_by_scale: bool = False) -> dict:
    """Returns {group_name: {seed: value}}.

    group_by_scale=True: разные точки шкалы параметров/эпох для одной и той
    же архитектуры считаются РАЗНЫМИ группами (иначе, например, все прогоны
    dimenet_pp_enhanced на 0.5x/1x/2x/4x параметров смешаются в одну кривую
    "модели" и усреднятся между собой). Ключ группы:
    "<model>__n<n_params>__ep<max эпох по факту, len(history)>".
    Без --group-by-scale поведение прежнее (одно имя = одна группа), что
    годится, только если metrics_dir содержит ровно одну точку на шкале.
    """
    by_model = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(metrics_dir, f"*_{loss}.json"))):
        with open(path) as f:
            r = json.load(f)
        if group_by_scale:
            n_params = r.get("n_params", "na")
            n_epochs_run = r.get("resources", {}).get("n_epochs_run", len(r.get("history", [])))
            key = f"{r['model']}__n{n_params}__ep{n_epochs_run}"
        else:
            key = r["model"]
        by_model[key][r["seed"]] = r["test_metrics"][metric]
    return by_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--metric", default="weighted_r2")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Restrict to these models (default: all found)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--group-by-scale", action="store_true",
                        help="Различать точки шкалы параметров/эпох одной и той же "
                             "архитектуры как отдельные группы (см. load_seed_values). "
                             "Обязательно для сравнения результатов экспериментов "
                             "по масштабированию параметров и по числу эпох.")
    args = parser.parse_args()

    higher_better = HIGHER_IS_BETTER.get(args.metric, True)
    by_model = load_seed_values(args.metrics_dir, args.loss, args.metric,
                                 group_by_scale=args.group_by_scale)
    if args.models:
        # при --group-by-scale ключи вида "model__nXXXX__epYYY" — фильтруем
        # по префиксу до "__", чтобы --models принимал обычные имена моделей
        by_model = {m: v for m, v in by_model.items()
                    if m in args.models or m.split("__")[0] in args.models}

    # keep only seeds common to ALL selected models, so the test is properly paired
    common_seeds = set.intersection(*(set(v.keys()) for v in by_model.values()))
    if len(common_seeds) < 3:
        print(f"WARNING: only {len(common_seeds)} seeds are common across all "
              f"selected models — results below will be even less reliable than usual.")
    common_seeds = sorted(common_seeds)

    print(f"Metric: {args.metric} ({'higher' if higher_better else 'lower'} is better)")
    print(f"Common seeds used (paired): {common_seeds}")
    print(f"NOTE: n={len(common_seeds)} paired samples -> LOW statistical power. "
          f"A non-significant p-value here does NOT mean the models are equal — "
          f"it may just mean 3 seeds isn't enough to tell. See "
          f"scripts/molecule_level_eval.py for a stronger test.\n")

    means = {m: np.mean([v[s] for s in common_seeds]) for m, v in by_model.items()}
    ranked = sorted(means, key=lambda m: means[m], reverse=higher_better)

    print(f"{'Model':<15} {'mean':>10}")
    for m in ranked:
        print(f"{m:<15} {means[m]:>10.5f}")
    print()

    header = f"{'Model A':<13} {'Model B':<13} {'mean diff':>10} {'t-test p':>10} {'wilcoxon p':>12} {'sig @'+str(args.alpha):>8}"
    print(header)
    print("-" * len(header))
    for a, b in itertools.combinations(ranked, 2):
        va = np.array([by_model[a][s] for s in common_seeds])
        vb = np.array([by_model[b][s] for s in common_seeds])
        diff = va.mean() - vb.mean()

        t_p = stats.ttest_rel(va, vb).pvalue if not np.allclose(va, vb) else 1.0
        try:
            w_p = stats.wilcoxon(va, vb).pvalue
        except ValueError:
            w_p = float("nan")  # identical arrays / all-zero differences

        sig = "*" if (not np.isnan(t_p) and t_p < args.alpha) else ""
        print(f"{a:<13} {b:<13} {diff:>10.5f} {t_p:>10.4f} {w_p:>12.4f} {sig:>8}")


if __name__ == "__main__":
    main()
