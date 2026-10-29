# -*- coding: utf-8 -*-
"""
extract_delta_diagnostics.py
Runs test inference for dimenet_pp_delta and dumps all physics diagnostics.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance

# Добавляем корень в PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from src.train_3d_physics import build_model, get_dataset  # если функции называются иначе, поправим по выводу grep
from src.data.constants import SIGMA_BINS, POLAR_MASK, EPS

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # Загружаем конфиг
    with open("configs/3d/dimenet_pp_delta.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("configs/3d/base.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    # Мёрджим base
    merged_cfg = {**base_cfg, **cfg}
    merged_cfg["model"] = {**base_cfg.get("model", {}), **cfg.get("model", {})}

    # Датасет
    print("--> Loading test dataset...")
    # Строим тест-датасет по логике вашего train_3d_physics
    # (Используем test_path из base.yaml)
    from torch_geometric.loader import DataLoader
    test_ds = get_dataset(merged_cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Модель
    print("--> Building DimeNet++ Delta...")
    model, _ = build_model(merged_cfg)
    ckpt_path = "results/physics_v2/checkpoints/dimenet_pp_delta_seed0_physics.pt"
    print(f"--> Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_z = []
    mol_sizes = []

    print("--> Running inference...")
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
        if hasattr(batch, "z"):
            all_z.append(batch.z.cpu().numpy())
        mol_sizes.extend(torch.bincount(batch.batch.cpu()).numpy().tolist())

    preds = np.maximum(np.concatenate(all_preds, axis=0), 0.0)
    targets = np.maximum(np.concatenate(all_targets, axis=0), 0.0)
    atomic_numbers = np.concatenate(all_z, axis=0) if all_z else np.zeros(len(preds), dtype=int)

    d_sigma = float(SIGMA_BINS[1] - SIGMA_BINS[0])

    # Считаем молекулярные характеристики
    mol_true_areas, mol_pred_areas = [], []
    mol_true_polar, mol_pred_polar = [], []
    mol_charge_res = []
    mol_emds = []
    mol_maes = []
    mol_m2_true, mol_m2_pred = [], []
    mol_profiles_true, mol_profiles_pred = [], []

    idx = 0
    print("--> Computing physical invariants...")
    for sz in mol_sizes:
        t_mol = targets[idx : idx + sz].sum(axis=0)
        p_mol = preds[idx : idx + sz].sum(axis=0)

        # 1. Total Area
        s_true = float(t_mol.sum() * d_sigma)
        s_pred = float(p_mol.sum() * d_sigma)
        mol_true_areas.append(s_true)
        mol_pred_areas.append(s_pred)

        # 2. Polar Area (|sigma| >= 0.01)
        mol_true_polar.append(float(t_mol[POLAR_MASK].sum() * d_sigma))
        mol_pred_polar.append(float(p_mol[POLAR_MASK].sum() * d_sigma))

        # 3. Residual Screening Charge (net neutral: integral sigma * p(sigma) dsigma)
        q_res = float(np.sum(SIGMA_BINS * p_mol) * d_sigma)
        mol_charge_res.append(q_res)

        # 4. 2nd Moment (Electrostatic spread: integral sigma^2 * p(sigma) dsigma)
        mol_m2_true.append(float(np.sum((SIGMA_BINS ** 2) * t_mol) * d_sigma))
        mol_m2_pred.append(float(np.sum((SIGMA_BINS ** 2) * p_mol) * d_sigma))

        # 5. EMD (Wasserstein distance)
        try:
            emd = wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_mol + EPS, p_mol + EPS)
        except Exception:
            emd = np.nan
        mol_emds.append(emd)

        mol_maes.append(float(np.mean(np.abs(t_mol - p_mol))))
        mol_profiles_true.append(t_mol)
        mol_profiles_pred.append(p_mol)

        idx += sz

    out_file = "results/physics_v2/delta_full_diagnostics.npz"
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
        mol_profiles_true=np.array(mol_profiles_true),
        mol_profiles_pred=np.array(mol_profiles_pred),
        sigma_bins=SIGMA_BINS,
    )
    print(f" [DONE] Saved all diagnostic tensors to {out_file}")

if __name__ == "__main__":
    main()
