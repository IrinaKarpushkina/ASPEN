"""Variance test: how much does averaging trained models help?

    python -m scripts.ensemble_eval \
        results/physics/checkpoints/dimenet_pp_enhanced_physics_seed{0,1,2}_physics.pt

If the ensemble of 3 seeds barely beats the single models, the remaining error is bias/noise, not
variance (=> ceiling for this family). If it improves clearly, variance is significant and
ensembling (also across architectures with the same `features`) is a cheap, fully physical gain,
since the mean of positive profiles with exact areas is again a positive profile.
All checkpoints given must use the same `features` (same test set tensors).
"""
import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset_3d import ChaosParquet3DDataset
from src.data.dataset_3d_enhanced import EnhancedChaosParquet3DDataset
from src.metrics import compute_all_metrics
from src.train_3d_physics import _MODELS_NEEDING_TRIPLETS, build_model

KEYS = ["weighted_r2", "weighted_mae", "polar_mae", "emd_raw", "molecular_mae", "molecular_cosine"]


def ds_class(features):
    if features == "sasa":
        from src.data.dataset_3d_sasa import SasaChaosParquet3DDataset
        return SasaChaosParquet3DDataset
    return EnhancedChaosParquet3DDataset if features == "enhanced" else ChaosParquet3DDataset


@torch.no_grad()
def predict(ckpt_path, device, cache):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    name, features = cfg["model"]["name"], cfg.get("features", "base")
    trip = name in _MODELS_NEEDING_TRIPLETS or name.startswith("sigma_dn")
    key = (features, trip)
    dc = cfg["data"]
    if key not in cache:
        kw = dict(cache_dir=dc.get("cache_dir"), compute_triplets=trip, require_provided_coords=True)
        cls = ds_class(features)
        cache[key] = (cls(dc["train_path"], **kw), cls(dc["test_path"], **kw))
    train_ds, test_ds = cache[key]
    model, _ = build_model(cfg)
    if hasattr(model, "calibrate"):
        model.calibrate(train_ds)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    P, T, sizes = [], [], []
    for d in DataLoader(test_ds, batch_size=24, shuffle=False):
        d = d.to(device)
        P.append(model(d).float().cpu().numpy()); T.append(d.y.float().cpu().numpy())
        sizes.extend(torch.bincount(d.batch.cpu()).tolist())
    return np.concatenate(P), np.concatenate(T), sizes, train_ds.bin_weights_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, preds, ref = {}, [], None
    for c in a.ckpts:
        p, t, sizes, bw = predict(c, dev, cache)
        if ref is None:
            ref = (t, sizes, bw)
        elif p.shape != ref[0].shape or not np.allclose(t, ref[0]):
            raise SystemExit("checkpoints do not share the same test set/order")
        preds.append(p)
        m = compute_all_metrics(ref[0], p, ref[1], ref[2], 0.0)
        print(f"{c.split('/')[-1]:60s}", {k: round(m[k], 5) for k in KEYS})
    m = compute_all_metrics(ref[0], np.mean(preds, 0), ref[1], ref[2], 0.0)
    print(f"{'ENSEMBLE (mean)':60s}", {k: round(m[k], 5) for k in KEYS})


if __name__ == "__main__":
    main()
