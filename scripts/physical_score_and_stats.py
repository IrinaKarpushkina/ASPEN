"""
physical_score_and_stats.py -- composite physical-fidelity score and
paired statistical comparison for the 3D architecture benchmark.

WHY RANK-BASED AGGREGATION (not raw-value weighted sum):
Metrics live on incomparable scales (R^2 ~0.9, EMD ~0.0007, MAE ~0.1),
so assigning raw numeric weights is methodologically awkward -- a
weight of "50" on EMD vs "2" on R^2 has no natural interpretation.
Ranking models 1..N per metric, then averaging ranks with justified
weights, avoids this: weights instead express "how many times more
important is this criterion", which is directly defensible.

METRIC WEIGHTS (see accompanying chat message for the physical rationale):
  weighted_r2   (atomic)  : 2   -- overall variance-weighted accuracy
  weighted_mae  (atomic)  : 2   -- same information as weighted_r2, kept
                                   for robustness (rank correlation isn't
                                   perfect at this precision)
  polar_mae     (atomic)  : 3   -- thermodynamically decisive region
                                   (H-bond donor/acceptor)
  emd_raw       (atomic)  : 3   -- distributional shape fidelity
  cosine_similarity (atomic): 1 -- saturates near 1.0, least discriminating
  molecular_*   (all four) : 1 total (0.25 each) -- DERIVED from atomic
                                   predictions by summation, not
                                   independent evidence; kept only as a
                                   downstream sanity check, not a primary
                                   criterion

Usage:
    python physical_score_and_stats.py --metrics-dir results/metrics/metrics_3d
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Metric direction (True = higher is better) and weights.
# ---------------------------------------------------------------------------
METRIC_SPEC = {
    "weighted_r2":        dict(higher_is_better=True,  weight=2.0, group="atomic_primary"),
    "weighted_mae":       dict(higher_is_better=False, weight=2.0, group="atomic_primary"),
    "polar_mae":          dict(higher_is_better=False, weight=3.0, group="atomic_primary"),
    "emd_raw":            dict(higher_is_better=False, weight=3.0, group="atomic_primary"),
    "cosine_similarity":  dict(higher_is_better=True,  weight=1.0, group="atomic_secondary"),
    "molecular_mae":         dict(higher_is_better=False, weight=0.25, group="molecular_check"),
    "molecular_emd":         dict(higher_is_better=False, weight=0.25, group="molecular_check"),
    "molecular_cosine":      dict(higher_is_better=True,  weight=0.25, group="molecular_check"),
    "molecular_polar_mae":   dict(higher_is_better=False, weight=0.25, group="molecular_check"),
}

PRIMARY_METRICS_FOR_STATS = ["weighted_r2", "polar_mae", "emd_raw"]


def load_all_metrics(metrics_dir: str) -> pd.DataFrame:
    """Loads every {model}_seed{n}_mse.json in metrics_dir into a long
    DataFrame: columns = model, seed, <metric columns>.

    The metrics of interest live under the top-level "test_metrics" key
    (siblings: "history" per-epoch training log, "val_metrics_at_best",
    "n_params", "config_path", etc.) -- NOT at the top level of the file.
    """
    rows = []
    all_keys_seen = set()
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*_seed*_mse.json"))):
        fname = os.path.basename(path)
        model_part, seed_part = fname.rsplit("_seed", 1)
        seed = int(seed_part.split("_")[0])
        with open(path) as f:
            payload = json.load(f)
        data = payload.get("test_metrics", payload)  # fall back to top-level if absent
        all_keys_seen.update(data.keys())
        row = {"model": model_part, "seed": seed}
        row.update({k: v for k, v in data.items() if k in METRIC_SPEC})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No *_seed*_mse.json files found under {metrics_dir}")

    missing = set(METRIC_SPEC.keys()) - all_keys_seen
    if missing:
        raise KeyError(
            f"Expected metric keys not found under 'test_metrics' in any JSON file: {sorted(missing)}\n"
            f"Actual keys present under 'test_metrics': {sorted(all_keys_seen)}\n"
            f"-> Update METRIC_SPEC's keys in this script to match your real "
            f"JSON field names (see the 'Actual keys present' list above), then rerun."
        )
    return pd.DataFrame(rows)


def composite_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """For each metric, rank models by their SEED-MEAN value (1 = best),
    then compute a weighted average rank as the composite score (lower
    = better, like a golf score)."""
    means = df.groupby("model")[list(METRIC_SPEC.keys())].mean()

    rank_df = pd.DataFrame(index=means.index)
    for metric, spec in METRIC_SPEC.items():
        ascending = not spec["higher_is_better"]  # if higher is better, rank descending
        rank_df[metric + "_rank"] = means[metric].rank(ascending=ascending, method="average")

    total_weight = sum(s["weight"] for s in METRIC_SPEC.values())
    weighted_rank = sum(
        rank_df[m + "_rank"] * METRIC_SPEC[m]["weight"] for m in METRIC_SPEC
    ) / total_weight

    result = means.copy()
    result["composite_rank_score"] = weighted_rank
    result = result.sort_values("composite_rank_score")
    return result, rank_df


def paired_comparison(df: pd.DataFrame, model_a: str, model_b: str, metric: str):
    """Paired comparison across matched seeds (same train/val/test split,
    same seed number) between two models on one metric.

    CAVEAT (state this in the paper): with only 3 seeds, a paired t-test
    or Wilcoxon signed-rank test has very low statistical power -- do not
    over-interpret a non-significant p-value as "no real difference", and
    do not claim strong significance from a small p-value either. Report
    these as a supplementary, descriptive check (direction and magnitude
    of the paired differences across all 3 seeds), not as a definitive
    hypothesis test.
    """
    a = df[df.model == model_a].sort_values("seed")[metric].to_numpy()
    b = df[df.model == model_b].sort_values("seed")[metric].to_numpy()
    if len(a) != len(b) or len(a) < 2:
        return None
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon needs at least one non-zero difference and enough
        # samples; with n=3 it frequently can't compute an exact
        # distribution -- report NaN rather than crash.
        w_stat, w_p = np.nan, np.nan
    return dict(
        metric=metric, model_a=model_a, model_b=model_b,
        mean_diff=diff.mean(), diff_per_seed=diff.tolist(),
        consistent_direction=bool(np.all(diff > 0) or np.all(diff < 0)),
        paired_t_p=t_p, wilcoxon_p=w_p,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--csv", default="results/physical_score_table.csv")
    parser.add_argument("--stats-csv", default="results/pairwise_stats.csv")
    args = parser.parse_args()

    df = load_all_metrics(args.metrics_dir)
    result, rank_df = composite_rank_score(df)

    print("=" * 100)
    print("Composite physical-fidelity score (lower = better; weighted average rank)")
    print("Weights: weighted_r2=2, weighted_mae=2, polar_mae=3, emd_raw=3, "
          "cosine_similarity=1, molecular_*=0.25 each")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        print(result[["composite_rank_score"] + list(METRIC_SPEC.keys())])

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    result.to_csv(args.csv)
    print(f"\nSaved: {args.csv}")

    # Pairwise statistical checks: best model vs every runner-up, on the
    # three primary physical metrics.
    ranked_models = result.index.tolist()
    best = ranked_models[0]
    stats_rows = []
    print("\n" + "=" * 100)
    print(f"Paired comparisons: {best} (top composite score) vs each other model")
    print("CAVEAT: n=3 seeds -> very low power, treat p-values as descriptive only.")
    print("=" * 100)
    for other in ranked_models[1:]:
        for metric in PRIMARY_METRICS_FOR_STATS:
            res = paired_comparison(df, best, other, metric)
            if res is None:
                continue
            stats_rows.append(res)
            direction = "consistent" if res["consistent_direction"] else "MIXED"
            print(f"  {best} vs {other:12s} | {metric:14s} | "
                  f"mean_diff={res['mean_diff']:+.5f} | seed-direction={direction:10s} | "
                  f"paired_t p={res['paired_t_p']:.3f} | wilcoxon p={res['wilcoxon_p']}")

    stats_df = pd.DataFrame(stats_rows)
    os.makedirs(os.path.dirname(args.stats_csv) or ".", exist_ok=True)
    stats_df.to_csv(args.stats_csv, index=False)
    print(f"\nSaved: {args.stats_csv}")

    print("\n" + "=" * 100)
    print("Interpretation guide:")
    print("- 'consistent' direction across all 3 seeds is more informative than the")
    print("  p-value itself at this sample size -- it means the winner won on every")
    print("  seed, not just on average.")
    print("- A small p-value here should be reported as suggestive, not definitive.")
    print("- If composite scores are close AND directions are mixed across seeds,")
    print("  report the models as practically equivalent under this budget, and")
    print("  select among them by secondary criteria (training/inference speed,")
    print("  architectural simplicity) -- this is a valid and honest conclusion.")
    print("=" * 100)


if __name__ == "__main__":
    main()
