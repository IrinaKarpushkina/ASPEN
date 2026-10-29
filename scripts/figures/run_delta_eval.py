# -*- coding: utf-8 -*-
"""
run_delta_eval.py
Path: scripts/figures/run_delta_eval.py
Full test-set inference for DimeNet++ Delta with physics invariant extraction.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance

# Определение корня репозитория (scripts/figures/run_delta_eval.py -> parents[2] это ASPEN)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Переход в корень, чтобы внутренние загрузчики (YAML, кэш) корректно видели структуру
os.chdir(REPO_ROOT)

from src.config import load_config
from src.data.dataset_3d_enhanced import EnhancedChaosParquet3DDataset
from src.data.constants import SIGMA_BINS, POLAR_MASK, EPS
from src.train_3d_physics import build_model, _MODELS_NEEDING_TRIPLETS
from torch_geometric.loader import DataLoader

_trapz = getattr(np, "trapezoid", None) or np.trapz

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> [1/5] Using device: {device}")
    print(f"--> Repository root detected: {REPO_ROOT}")

    config_path = REPO_ROOT / "configs/3d/dimenet_pp_delta.yaml"
    cfg = load_config(str(config_path))

    dc = cfg["data"]
    tc = dict(cfg["training"])
    name = cfg["model"]["name"]

    ds_kwargs = dict(
        cache_dir=dc.get("cache_dir"),
        force_recompute=False,
        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith("sigma_dn")
                          or (name == "sigma_kitchen_sink" and cfg["model"].get("backbone", "dimenet") == "dimenet")
                          or name == "sigma_continuous_dimenet"),
        require_provided_coords=True,
    )

    print("--> [2/5] Loading datasets (Train for calibration, Test for evaluation)...")
    train_ds = EnhancedChaosParquet3DDataset(dc["train_path"], **ds_kwargs)
    test_ds = EnhancedChaosParquet3DDataset(dc["test_path"], **ds_kwargs)

    bs = tc.get("batch_size", 32)
    nw = tc.get("num_workers", 4)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print("--> [3/5] Building DimeNet++ Delta and performing calibration...")
    model, model_name = build_model(cfg)
    if hasattr(model, "calibrate"):
        model.calibrate(train_ds)
    model = model.to(device)
    for _m in model.modules():
        if hasattr(_m, "calibrate_area"):
            _m.calibrate_area(train_ds)

    ckpt_path = REPO_ROOT / "results/physics_v2/checkpoints/dimenet_pp_delta_seed0_physics.pt"
    print(f"--> [4/5] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state_dict)
    model.eval()

    print(f"--> [5/5] Running test inference on {len(test_ds)} molecules...")
    all_preds = []
    all_targets = []
    all_z = []
    mol_sizes = []

    t0 = time.time()
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        all_preds.append(out.float().cpu().numpy())
        all_targets.append(batch.y.float().cpu().numpy())
        if hasattr(batch, "z"):
            all_z.append(batch.z.cpu().numpy())
        mol_sizes.extend(torch.bincount(batch.batch.cpu()).numpy().tolist())

    elapsed = time.time() - t0
    print(f"--> Inference completed in {elapsed:.2f}s")

    preds = np.maximum(np.concatenate(all_preds, axis=0), 0.0)
    targets = np.maximum(np.concatenate(all_targets, axis=0), 0.0)
    atomic_numbers = np.concatenate(all_z, axis=0) if all_z else np.zeros(len(preds), dtype=int)

    # Расчет молекулярных физических инвариантов
    mol_true_areas, mol_pred_areas = [], []
    mol_true_polar, mol_pred_polar = [], []
    mol_charge_res = []
    mol_m2_true, mol_m2_pred = [], []
    mol_emds = []
    mol_maes = []
    mol_cosines = []
    mol_profiles_true, mol_profiles_pred = [], []

    idx = 0
    for sz in mol_sizes:
        t_mol = targets[idx : idx + sz].sum(axis=0)
        p_mol = preds[idx : idx + sz].sum(axis=0)

        # 1. Total Cavity Area: integral p(sigma) dsigma
        mol_true_areas.append(float(_trapz(t_mol, SIGMA_BINS)))
        mol_pred_areas.append(float(_trapz(p_mol, SIGMA_BINS)))

        # 2. Polar Area (|sigma| >= 0.01)
        mol_true_polar.append(float(_trapz(t_mol[POLAR_MASK], SIGMA_BINS[POLAR_MASK])))
        mol_pred_polar.append(float(_trapz(p_mol[POLAR_MASK], SIGMA_BINS[POLAR_MASK])))

        # 3. Residual net screening charge: integral sigma * p(sigma) dsigma
        mol_charge_res.append(float(_trapz(SIGMA_BINS * p_mol, SIGMA_BINS)))

        # 4. Electrostatic 2nd Moment: integral sigma^2 * p(sigma) dsigma
        mol_m2_true.append(float(_trapz((SIGMA_BINS**2) * t_mol, SIGMA_BINS)))
        mol_m2_pred.append(float(_trapz((SIGMA_BINS**2) * p_mol, SIGMA_BINS)))

        # 5. EMD / Wasserstein
        try:
            emd = wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_mol + EPS, p_mol + EPS)
        except Exception:
            emd = np.nan
        mol_emds.append(float(emd))

        # 6. MAE & Cosine
        mol_maes.append(float(np.mean(np.abs(t_mol - p_mol))))
        cos = np.dot(t_mol, p_mol) / (np.linalg.norm(t_mol) * np.linalg.norm(p_mol) + EPS)
        mol_cosines.append(float(cos))

        mol_profiles_true.append(t_mol)
        mol_profiles_pred.append(p_mol)

        idx += sz

    out_file = REPO_ROOT / "results/physics_v2/delta_test_results.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        targets=targets,
        preds=preds,
        atomic_numbers=atomic_numbers,
        mol_sizes=np.array(mol_sizes),
        mol_true_areas=np.array(mol_true_areas),
        mol_pred_areas=np.array(mol_pred_areas),
        mol_true_polar=np.array(mol_true_polar),
        mol_pred_polar=np.array(mol_pred_polar),
        mol_charge_res=np.array(mol_charge_res),
        mol_m2_true=np.array(mol_m2_true),
        mol_m2_pred=np.array(mol_m2_pred),
        mol_emds=np.array(mol_emds),
        mol_maes=np.array(mol_maes),
        mol_cosines=np.array(mol_cosines),
        mol_profiles_true=np.array(mol_profiles_true),
        mol_profiles_pred=np.array(mol_profiles_pred),
        sigma_bins=SIGMA_BINS,
    )
    print(f" [DONE] Successfully dumped real evaluation data to {out_file}")

if __name__ == "__main__":
    main()
