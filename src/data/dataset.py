"""
dataset.py — единый ChaosParquetDataset с поддержкой режимов "3d" и "2d_pure".

mode="3d"      — текущий режим: radius-graph по координатам, RBF edge features,
                 extended node features (15-dim). SchNet и все 3D-модели.

mode="2d_pure" — чисто 2D: граф из химических связей SMILES/RDKit,
                 edge features только из топологии (7-dim, без расстояний),
                 node features без degree из radius-graph (14-dim).
                 Никаких 3D-координат не используется.

Кеши для двух режимов хранятся отдельно (разный cache_key).
"""

from __future__ import annotations
import hashlib
import logging
import os
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from .constants import ELEMENT_TO_Z, CUTOFF, MAX_NUM_NEIGHBORS, N_RBF
from .features import precompute_molecule_tensors
from .features_2d import precompute_molecule_tensors_2d

logger = logging.getLogger(__name__)

_WALK_LENGTH = 16


def _compute_rwpe(
    edge_index: torch.Tensor, num_nodes: int, walk_length: int
) -> torch.Tensor:
    device = edge_index.device
    N = num_nodes
    E = edge_index.size(1)
    if E == 0:
        return torch.zeros(N, walk_length, device=device)

    ones = torch.ones(E, device=device)
    deg  = torch.zeros(N, device=device).scatter_add_(
        0, edge_index[0], ones
    ).clamp(min=1.0)
    vals = ones / deg[edge_index[0]]
    A    = torch.sparse_coo_tensor(edge_index, vals, (N, N)).coalesce()

    pe = torch.zeros(N, walk_length, device=device)
    Ak = torch.eye(N, device=device)
    for k in range(walk_length):
        Ak = torch.sparse.mm(A.t(), Ak.t()).t()
        pe[:, k] = Ak.diagonal()
    return pe


class ChaosParquetDataset(Dataset):
    """
    Args:
        parquet_path:      путь к train/val/test parquet.
        mode:              "3d"      — radius-graph + RBF (текущий режим)
                           "2d_pure" — граф из SMILES, без 3D
        use_extended:      только для mode="3d". True → 15-dim node feat.
        cutoff, max_num_neighbors, n_rbf — только для mode="3d".
        walk_length:       длина RWPE для GPS.
        cache_dir:         папка для кеша.
        force_recompute:   сбросить кеш.
    """

    def __init__(
        self,
        parquet_path:      str,
        mode:              str   = "3d",       # "3d" или "2d_pure"
        use_extended:      bool  = True,       # только для mode="3d"
        cutoff:            float = CUTOFF,
        max_num_neighbors: int   = MAX_NUM_NEIGHBORS,
        n_rbf:             int   = N_RBF,
        walk_length:       int   = _WALK_LENGTH,
        cache_dir:         Optional[str] = None,
        force_recompute:   bool  = False,
    ):
        super().__init__()
        assert mode in ("3d", "2d_pure"), f"Unknown mode: {mode}"
        self.parquet_path      = parquet_path
        self.mode              = mode
        self.use_extended      = use_extended
        self.cutoff            = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.n_rbf             = n_rbf
        self.walk_length       = walk_length

        cache_dir = cache_dir or os.path.join(
            os.path.dirname(parquet_path), "cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_key  = self._cache_key()
        cache_path = os.path.join(
            cache_dir,
            f"{os.path.basename(parquet_path)}.{cache_key}.pt",
        )

        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading cached dataset [{mode}]: {cache_path}")
            payload         = torch.load(cache_path, weights_only=False)
            self._data_list = payload["data_list"]
            self.sigma_cols = payload["sigma_cols"]
            self.mol_ids    = payload["mol_ids"]
            self.stats      = payload["stats"]
        else:
            self._data_list, self.sigma_cols, self.mol_ids, self.stats = (
                self._build(parquet_path)
            )
            logger.info(f"Caching dataset [{mode}]: {cache_path}")
            torch.save(
                dict(data_list=self._data_list, sigma_cols=self.sigma_cols,
                     mol_ids=self.mol_ids, stats=self.stats),
                cache_path,
            )

        logger.info(
            f"{os.path.basename(parquet_path)} [{mode}]: "
            f"{len(self._data_list)} mols, {self.stats['n_atoms']} atoms, "
            f"RDKit valid: {self.stats['mol_valid']}/{self.stats['n_mols']} "
            f"({100*self.stats['mol_valid']/max(self.stats['n_mols'],1):.1f}%)"
        )

    def _cache_key(self) -> str:
        if self.mode == "2d_pure":
            raw = f"mode=2d_pure_v1"
        else:
            raw = (
                f"mode=3d_ext={self.use_extended}_cutoff={self.cutoff}"
                f"_maxnn={self.max_num_neighbors}_nrbf={self.n_rbf}"
                f"_walk={self.walk_length}_v3"
            )
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def _build(self, parquet_path: str):
        logger.info(
            f"Building [{self.mode}] dataset from {parquet_path} (no cache)"
        )
        df = pd.read_parquet(parquet_path)
        mol_ids    = df["mol_id"].unique().tolist()
        sigma_cols = sorted(
            [c for c in df.columns if c.startswith("sigma_")],
            key=lambda x: int(x.split("_")[1]),
        )
        assert len(sigma_cols) == 51

        has_smiles = "smiles" in df.columns
        if not has_smiles and self.mode == "2d_pure":
            raise ValueError("mode='2d_pure' requires 'smiles' column in parquet")

        data_list, n_atoms_total, mol_valid_count = [], 0, 0

        grouped = df.groupby("mol_id", sort=False)
        for mol_id, mol_df in grouped:
            mol_df = mol_df.sort_values("atom_index")
            z_list = [ELEMENT_TO_Z.get(e, 6) for e in mol_df["element"]]
            z      = torch.tensor(z_list, dtype=torch.long)
            pos    = torch.tensor(
                mol_df[["coord_x", "coord_y", "coord_z"]].values,
                dtype=torch.float,
            )
            y = torch.nan_to_num(
                torch.tensor(mol_df[sigma_cols].values, dtype=torch.float),
                nan=0.0,
            )
            smiles = mol_df["smiles"].iloc[0] if has_smiles else ""

            if self.mode == "2d_pure":
                pre = precompute_molecule_tensors_2d(z_list, smiles)
                # pos хранится для совместимости с Data, но не используется
                # ни в признаках, ни в графе
                pe = _compute_rwpe(
                    pre["edge_index"], len(z_list), self.walk_length
                )
                data = Data(
                    z=z, pos=pos, y=y,
                    x=pre["x"],
                    edge_index=pre["edge_index"],
                    edge_attr=pre["edge_attr"],
                    degree=pre["degree"],
                    pe=pe,
                )
            else:  # "3d"
                pre = precompute_molecule_tensors(
                    z_list, pos, smiles,
                    use_extended=self.use_extended,
                    n_rbf=self.n_rbf,
                    cutoff=self.cutoff,
                    max_num_neighbors=self.max_num_neighbors,
                )
                pe = _compute_rwpe(
                    pre["edge_index"], len(z_list), self.walk_length
                )
                data = Data(
                    z=z, pos=pos, y=y,
                    x=pre["x"],
                    edge_index=pre["edge_index"],
                    edge_attr=pre["edge_attr"],
                    degree=pre["degree"],
                    pe=pe,
                )

            data_list.append(data)
            n_atoms_total   += len(z_list)
            mol_valid_count += int(pre["mol_valid"])

        stats = dict(n_mols=len(mol_ids), n_atoms=n_atoms_total,
                     mol_valid=mol_valid_count)
        return data_list, sigma_cols, mol_ids, stats

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def bin_weights_numpy(self):
        import numpy as np
        from .constants import EPS
        all_y = torch.cat([d.y for d in self._data_list], dim=0).numpy()
        var   = np.var(all_y, axis=0)
        var   = np.maximum(var, EPS)
        return (var / var.sum()).astype("float32")
