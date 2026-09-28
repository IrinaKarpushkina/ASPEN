"""Where does the remaining error live: area, shape, or both?  (oracle experiments on the test set)

    python -m scripts.error_decomposition results/physics/checkpoints/dimenet_pp_enhanced_physics_seed0_physics.pt

Prints the benchmark metrics for
  as-is          : model predictions
  oracle area    : predicted SHAPE (P / sum P) rescaled to the TRUE per-atom area (sum y)
  oracle shape   : TRUE shape rescaled to the PREDICTED area
If 'oracle area' is much better than 'as-is', magnitude (geometry: SASA) is the bottleneck and
use_geom_area / fixed-area designs are worth it. If 'oracle shape' is much better, the shape
(distribution) is the bottleneck and area tricks will not help.
"""
import argparse

import numpy as np
import torch

from scripts.ensemble_eval import KEYS, predict
from src.metrics import compute_all_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, T, sizes, bw = predict(a.ckpt, dev, {})
    Ap, At = P.sum(1, keepdims=True), T.sum(1, keepdims=True)
    eps = 1e-12
    variants = {
        "as-is": P,
        "oracle area": P / np.maximum(Ap, eps) * At,
        "oracle shape": T / np.maximum(At, eps) * Ap,
    }
    for k, v in variants.items():
        m = compute_all_metrics(T, v, sizes, bw, 0.0)
        print(f"{k:13s}", {x: round(float(m[x]), 5) for x in KEYS})
    rel = np.abs(Ap - At)[:, 0] / np.maximum(At[:, 0], eps)
    print(f"median relative area error: {np.median(rel):.3f}   p90: {np.percentile(rel, 90):.3f}")
