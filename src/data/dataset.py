"""
dataset.py — ChaosParquetDataset (2D-only).

Expected parquet schema (one row per atom):
    mol_id       — molecule identifier (grouping key)
    atom_index   — per-molecule atom order (0..N-1), used to sort rows
    element      — element symbol ('H', 'C', 'N', ...)
    smiles       — SMILES string for the whole molecule (same value for
                   every row of a given mol_id)
    sigma_0 .. sigma_50 — 51 sigma-profile bins (target), one row = one atom

coord_x / coord_y / coord_z, if present in your parquet, are IGNORED by
this repository on purpose — this is the 2D-only benchmark. If you need
them, they belong to the (not yet implemented) 3D benchmark under
configs/3d/ and src/models/models_3d/ — see PROVENANCE.md.

Caching: precomputed per-molecule graphs are cached to
<cache_dir>/<parquet_basename>.<hash>.pt so repeated runs (different
seeds/architectures on the same split) don't re-run RDKit parsing.
"""

from __future__ import annotations
import hashlib
import logging
import os
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from .constants import ELEMENT_TO_Z, WALK_LENGTH
from .features import precompute_molecule_tensors

logger = logging.getLogger(__name__)


def _compute_rwpe(edge_index: torch.Tensor, num_nodes: int, walk_length: int) -> torch.Tensor:
    """Random-Walk Positional Encoding (Dwivedi et al., 2022 LSPE), used only
    by the GPS model. Purely topological (built from edge_index), so it is
    well defined for a 2D chemical-bond graph."""
    device = edge_index.device
    N = num_nodes
    E = edge_index.size(1)
    if E == 0:
        return torch.zeros(N, walk_length, device=device)

    ones = torch.ones(E, device=device)
    deg = torch.zeros(N, device=device).scatter_add_(0, edge_index[0], ones).clamp(min=1.0)
    vals = ones / deg[edge_index[0]]
    A = torch.sparse_coo_tensor(edge_index, vals, (N, N)).coalesce()

    pe = torch.zeros(N, walk_length, device=device)
    Ak = torch.eye(N, device=device)
    for k in range(walk_length):
        Ak = torch.sparse.mm(A.t(), Ak.t()).t()
        pe[:, k] = Ak.diagonal()
    return pe


class ChaosParquetDataset(Dataset):
    """
    Args:
        parquet_path:    path to a train/val/test parquet file.
        walk_length:     RWPE length (only consumed by the GPS model).
        cache_dir:       directory for the on-disk cache.
        force_recompute: ignore any existing cache and rebuild it.
    """

    def __init__(
        self,
        parquet_path: str,
        walk_length: int = WALK_LENGTH,
        cache_dir: Optional[str] = None,
        force_recompute: bool = False,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.walk_length = walk_length

        cache_dir = cache_dir or os.path.join(os.path.dirname(parquet_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._cache_key()
        cache_path = os.path.join(
            cache_dir, f"{os.path.basename(parquet_path)}.{cache_key}.pt",
        )

        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading cached dataset [2d_pure]: {cache_path}")
            payload = torch.load(cache_path, weights_only=False)
            self._data_list = payload["data_list"]
            self.sigma_cols = payload["sigma_cols"]
            self.mol_ids = payload["mol_ids"]
            self.stats = payload["stats"]
        else:
            self._data_list, self.sigma_cols, self.mol_ids, self.stats = self._build(parquet_path)
            logger.info(f"Caching dataset [2d_pure]: {cache_path}")
            torch.save(
                dict(data_list=self._data_list, sigma_cols=self.sigma_cols,
                     mol_ids=self.mol_ids, stats=self.stats),
                cache_path,
            )

        logger.info(
            f"{os.path.basename(parquet_path)} [2d_pure]: "
            f"{len(self._data_list)} mols, {self.stats['n_atoms']} atoms, "
            f"RDKit valid: {self.stats['mol_valid']}/{self.stats['n_mols']} "
            f"({100 * self.stats['mol_valid'] / max(self.stats['n_mols'], 1):.1f}%)"
        )

    def _cache_key(self) -> str:
        raw = f"mode=2d_pure_walk={self.walk_length}_v1"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def _build(self, parquet_path: str):
        logger.info(f"Building [2d_pure] dataset from {parquet_path} (no cache found)")
        df = pd.read_parquet(parquet_path)
        mol_ids = df["mol_id"].unique().tolist()
        sigma_cols = sorted(
            [c for c in df.columns if c.startswith("sigma_")],
            key=lambda x: int(x.split("_")[1]),
        )
        assert len(sigma_cols) == 51, (
            f"Expected 51 sigma_* columns, found {len(sigma_cols)}. "
            f"Check your parquet schema."
        )

        if "smiles" not in df.columns:
            raise ValueError(
                "This 2D-only benchmark requires a 'smiles' column in the "
                "parquet file (the graph is built from chemical bonds, not "
                "from coordinates)."
            )

        data_list, n_atoms_total, mol_valid_count = [], 0, 0

        grouped = df.groupby("mol_id", sort=False)
        for mol_id, mol_df in grouped:
            mol_df = mol_df.sort_values("atom_index")
            z_list = [ELEMENT_TO_Z.get(e, 6) for e in mol_df["element"]]
            z = torch.tensor(z_list, dtype=torch.long)
            y = torch.nan_to_num(
                torch.tensor(mol_df[sigma_cols].values, dtype=torch.float), nan=0.0,
            )
            smiles = mol_df["smiles"].iloc[0]

            pre = precompute_molecule_tensors(z_list, smiles)
            pe = _compute_rwpe(pre["edge_index"], len(z_list), self.walk_length)

            data = Data(
                z=z, y=y,
                x=pre["x"],
                edge_index=pre["edge_index"],
                edge_attr=pre["edge_attr"],
                degree=pre["degree"],
                pe=pe,
            )

            data_list.append(data)
            n_atoms_total += len(z_list)
            mol_valid_count += int(pre["mol_valid"])

        stats = dict(n_mols=len(mol_ids), n_atoms=n_atoms_total, mol_valid=mol_valid_count)
        return data_list, sigma_cols, mol_ids, stats

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def bin_weights_numpy(self):
        import numpy as np
        from .constants import EPS
        all_y = torch.cat([d.y for d in self._data_list], dim=0).numpy()
        var = np.var(all_y, axis=0)
        var = np.maximum(var, EPS)
        return (var / var.sum()).astype("float32")
