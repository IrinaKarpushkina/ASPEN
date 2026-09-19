"""Build the train-only element-specific sigma-profile prior for experiment F."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is executed directly
# as ``python scripts/build_element_prior.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.constants import ELEMENT_TO_Z, SIGMA_BINS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    sigma_cols = sorted(
        [c for c in df.columns if c.startswith("sigma_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if len(sigma_cols) != 51:
        raise ValueError(f"Expected 51 sigma columns, found {len(sigma_cols)}")

    y = np.nan_to_num(df[sigma_cols].to_numpy(np.float64), nan=0.0)
    z = df["element"].map(ELEMENT_TO_Z).fillna(6).to_numpy(np.int64)

    # The prior is a SHAPE prior, not an area prior: normalize each mean
    # profile to a probability vector before taking log-probabilities.
    global_shape = np.maximum(y.mean(axis=0), 1e-12)
    prior = np.repeat(global_shape[None, :], 87, axis=0)
    counts = np.zeros(87, dtype=np.int64)

    for zi in np.unique(z):
        zi = int(zi)
        if 0 <= zi < 87:
            mask = z == zi
            counts[zi] = int(mask.sum())
            if counts[zi] > 0:
                prior[zi] = np.maximum(y[mask].mean(axis=0), 1e-12)

    prior /= prior.sum(axis=1, keepdims=True).clip(min=1e-12)
    logp = np.log(prior).astype(np.float32)

    np.savez(
        args.output,
        prior_logp=logp,
        sigma_bins=np.asarray(SIGMA_BINS, dtype=np.float32),
        element_counts=counts,
    )
    print(f"saved {args.output}")
    print(f"shape={logp.shape}, covered_Z={(counts > 0).sum()}, total_atoms={counts.sum()}")


if __name__ == "__main__":
    main()
