"""
molecule_level_eval.py — computes PER-MOLECULE test-set metrics for one
trained checkpoint, saved to a .npz keyed by mol_id.

Why: results/metrics/*.json only stores one aggregated number per seed,
so any significance test built on it has n=3 (very low power).
compute_all_metrics/molecular_metrics (src/metrics.py) already compute a
per-molecule value internally for every "molecular_*" metric — they just
average it away before returning. This script reruns that same per-
molecule computation (re-using the exact same functions from
src/metrics.py, NOT reimplementing the math) but keeps the per-molecule
array, keyed by mol_id so results from different models/architectures
(2D or 3D, different atom counts, different loader shuffling) can be
aligned correctly for a PAIRED test — see pairwise_significance_molecule.py.

This does NOT modify src/metrics.py or src/evaluate.py — it only imports
already-existing, already-tested functions from them.

Usage:
    # 2D model:
    python -m scripts.molecule_level_eval \
        --config configs/2d/gps.yaml --checkpoint results/checkpoints/gps_seed0_mse.pt \
        --mode 2d --split test --out results/per_mol/gps_seed0.npz

    # 3D model:
    python -m scripts.molecule_level_eval \
        --config configs/3d/dimenet_pp.yaml --checkpoint results/checkpoints/dimenet_pp_seed0_mse.pt \
        --mode 3d --split test --out results/per_mol/dimenet_pp_seed0.npz
"""
import argparse
import logging
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.config import load_config
from src.metrics import _cosine, POLAR_MASK, SIGMA_BINS, EPS
from src.models import MODEL_REGISTRY
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def per_molecule_metrics(targets: np.ndarray, preds: np.ndarray, mol_sizes: list,
                         mol_ids: list) -> dict:
    """Same math as src/metrics.py::molecular_metrics, but returns the raw
    per-molecule arrays (keyed by mol_id) instead of averaging them away."""
    out = {"mol_id": [], "mae": [], "polar_mae": [], "cosine": [], "emd": []}
    idx = 0
    for size, mid in zip(mol_sizes, mol_ids):
        t_mol = targets[idx: idx + size].sum(axis=0)
        p_mol = preds[idx: idx + size].sum(axis=0)

        out["mol_id"].append(mid)
        out["mae"].append(float(np.mean(np.abs(t_mol - p_mol))))
        out["polar_mae"].append(float(np.mean(np.abs(t_mol[POLAR_MASK] - p_mol[POLAR_MASK]))))
        out["cosine"].append(_cosine(
            t_mol / (np.linalg.norm(t_mol) + EPS),
            p_mol / (np.linalg.norm(p_mol) + EPS),
        ))
        try:
            out["emd"].append(wasserstein_distance(
                SIGMA_BINS, SIGMA_BINS,
                np.maximum(t_mol, 0.0) + EPS, np.maximum(p_mol, 0.0) + EPS,
            ))
        except Exception:
            out["emd"].append(float("nan"))
        idx += size
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["2d", "3d"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    if args.mode == "2d":
        from src.data.dataset import ChaosParquetDataset as DatasetCls
        ds_kwargs = dict(cache_dir=data_cfg.get("cache_dir"))
    else:
        from src.data.dataset_3d import ChaosParquet3DDataset as DatasetCls
        model_name = cfg["model"]["name"]
        needs_triplets = model_name in {"dimenet", "dimenet_pp", "spherenet"}
        ds_kwargs = dict(cache_dir=data_cfg.get("cache_dir"), compute_triplets=needs_triplets)

    path_key = f"{args.split}_path"
    dataset = DatasetCls(data_cfg[path_key], **ds_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model_cfg.pop("n_params_target", None)
    model = MODEL_REGISTRY[model_name](**model_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")

    all_preds, all_targets, mol_sizes, mol_ids = [], [], [], []
    mol_idx = 0
    with torch.no_grad():
        for data in loader:
            n_mols_in_batch = int(data.batch.max().item()) + 1
            batch_mol_ids = dataset.mol_ids[mol_idx: mol_idx + n_mols_in_batch]
            mol_idx += n_mols_in_batch

            data = data.to(device)
            preds = model(data)

            all_preds.append(preds.float().cpu().numpy())
            all_targets.append(data.y.float().cpu().numpy())
            mol_sizes.extend(torch.bincount(data.batch.cpu()).numpy().tolist())
            mol_ids.extend(batch_mol_ids)

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    assert len(mol_sizes) == len(mol_ids) == len(dataset), (
        f"mol_id alignment sanity check failed: {len(mol_sizes)} molecules seen, "
        f"{len(mol_ids)} ids collected, {len(dataset)} in dataset. "
        f"DataLoader(shuffle=False) is required for this alignment to hold — "
        f"do not pass shuffle=True."
    )

    result = per_molecule_metrics(targets_np, preds_np, mol_sizes, mol_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        mol_id=np.array(result["mol_id"], dtype=object),
        mae=np.array(result["mae"]),
        polar_mae=np.array(result["polar_mae"]),
        cosine=np.array(result["cosine"]),
        emd=np.array(result["emd"]),
    )
    logger.info(f"Saved per-molecule metrics for {len(result['mol_id'])} molecules -> {args.out}")


if __name__ == "__main__":
    main()
