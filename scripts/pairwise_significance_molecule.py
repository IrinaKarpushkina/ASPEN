"""
pairwise_significance_molecule.py — paired significance test between TWO
models' predictions on the SAME test-set molecules, using the per-molecule
.npz files produced by molecule_level_eval.py.

With ~7-8k test molecules (typical size for this benchmark's val/test
splits), this has vastly more statistical power than the 3-seed test in
pairwise_significance.py: it's testing "does model A beat model B on this
specific molecule" over thousands of paired observations, not "is model
A's overall test score higher across 3 training runs".

Uses:
  - paired t-test (parametric; assumes per-molecule error differences are
    roughly normal, usually a reasonable approximation at this sample size)
  - Wilcoxon signed-rank test (non-parametric; more robust to the fact
    that MAE distributions across molecules are typically right-skewed)
  - a bootstrap 95% CI on the mean difference, for an effect-size sense
    beyond just "significant or not"

Usage:
    python -m scripts.pairwise_significance_molecule \
        --a results/per_mol/spherenet_seed0.npz \
        --b results/per_mol/dimenet_pp_seed0.npz \
        --metric mae
"""
import argparse

import numpy as np
from scipy import stats

LOWER_IS_BETTER = {"mae", "polar_mae", "emd"}  # cosine is higher-is-better


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Path to model A's .npz (from molecule_level_eval.py)")
    parser.add_argument("--b", required=True, help="Path to model B's .npz")
    parser.add_argument("--metric", default="mae", choices=["mae", "polar_mae", "cosine", "emd"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    da = np.load(args.a, allow_pickle=True)
    db = np.load(args.b, allow_pickle=True)

    ids_a, ids_b = list(da["mol_id"]), list(db["mol_id"])
    common = sorted(set(ids_a) & set(ids_b))
    if len(common) < len(ids_a) or len(common) < len(ids_b):
        print(f"NOTE: {len(ids_a)} molecules in A, {len(ids_b)} in B, "
              f"{len(common)} in common -- using only the common set for a valid paired test.")

    idx_a = {mid: i for i, mid in enumerate(ids_a)}
    idx_b = {mid: i for i, mid in enumerate(ids_b)}
    va = np.array([da[args.metric][idx_a[m]] for m in common])
    vb = np.array([db[args.metric][idx_b[m]] for m in common])

    diff = va - vb  # positive = A worse (for MAE-type) / A better (for cosine)
    lower_is_better = args.metric in LOWER_IS_BETTER

    t_res = stats.ttest_rel(va, vb)
    w_res = stats.wilcoxon(va, vb)
    ci_lo, ci_hi = bootstrap_ci(diff)

    print(f"Metric: {args.metric} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"N paired molecules: {len(common)}")
    print(f"A mean: {va.mean():.6f}   B mean: {vb.mean():.6f}   diff (A-B): {diff.mean():.6f}")
    print(f"95% bootstrap CI on mean diff (A-B): [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Paired t-test:      p = {t_res.pvalue:.6g}")
    print(f"Wilcoxon signed-rank: p = {w_res.pvalue:.6g}")

    better = "A" if (diff.mean() < 0) == lower_is_better else "B"
    if t_res.pvalue < args.alpha and w_res.pvalue < args.alpha:
        print(f"\n=> {better} is significantly better on {args.metric} "
              f"(both tests p < {args.alpha}, CI excludes 0: {not (ci_lo <= 0 <= ci_hi)})")
    else:
        print(f"\n=> Difference is NOT significant at alpha={args.alpha}. "
              f"Do not claim one model beats the other on this metric.")


if __name__ == "__main__":
    main()
