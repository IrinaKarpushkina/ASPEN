"""Загрузка xTB-признаков (см. scripts/xtb_features.py) и присоединение к Data-объектам.

К d.x добавляется XTB_NODE_DIM = 7 инвариантных столбцов:
    q, bo_sum/4, bo_max, mol_charge/2, |mu|/5 [Debye], gap/10 [eV], ok
Дополнительно сохраняется d.xtb_q (N,) - заряды Малликена для surface-модели
(имя без слов "index"/"face", чтобы PyG не пытался смещать его при батчинге).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)
XTB_NODE_DIM = 7


def xtb_path_for(parquet_path: str) -> str:
    base = os.environ.get("ASPEN_XTB_DIR") or os.path.join(os.path.dirname(parquet_path), "xtb")
    return os.path.join(base, os.path.basename(parquet_path) + ".xtb.npz")


def attach_xtb(data_list, parquet_path: str) -> None:
    path = xtb_path_for(parquet_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Нет xTB-признаков: {path}\nСначала: python -m scripts.xtb_features --parquet {parquet_path}")
    f = np.load(path, allow_pickle=True)
    n_atoms = f["n_atoms"]
    if len(n_atoms) != len(data_list):
        raise RuntimeError(f"{path}: {len(n_atoms)} молекул, а в датасете {len(data_list)} "
                           f"(max-mols в тесте? другой parquet?)")
    off = np.concatenate([[0], np.cumsum(n_atoms)])
    z_all, pos_all = f["z"], f["pos"]
    q, bo_sum, bo_max = f["q"], f["bo_sum"], f["bo_max"]
    mc, dip, gap, ok = f["mol_charge"], f["mol_dipole"], f["mol_gap"], f["ok"]
    for k, d in enumerate(data_list):
        a, b = off[k], off[k + 1]
        n = b - a
        if n != d.num_nodes or not np.array_equal(z_all[a:b].astype(np.int64), d.z.cpu().numpy().astype(np.int64)):
            raise RuntimeError(f"{path}: порядок атомов не совпал с датасетом в молекуле #{k}")
        if not np.allclose(pos_all[a:b], d.pos.cpu().numpy(), atol=1e-3):
            raise RuntimeError(f"{path}: координаты не совпали с датасетом в молекуле #{k}")
        mol = np.array([mc[k] / 2.0, dip[k] / 5.0, gap[k] / 10.0, float(ok[k])], dtype=np.float32)
        feats = np.concatenate(
            [q[a:b, None], bo_sum[a:b, None] / 4.0, bo_max[a:b, None], np.repeat(mol[None], n, axis=0)], axis=1)
        feats = np.clip(np.nan_to_num(feats.astype(np.float32)), -10.0, 10.0)
        d.x = torch.cat([d.x, torch.from_numpy(feats)], dim=1)
        d.xtb_q = torch.from_numpy(q[a:b].astype(np.float32))
    logger.info("xTB-признаки присоединены (%s): x dim -> %d, молекул без сходимости: %d/%d",
                os.path.basename(path), data_list[0].x.shape[1], int((ok == 0).sum()), len(ok))
