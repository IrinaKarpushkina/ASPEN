"""
dataset_3d.py — ChaosParquet3DDataset.

Same parquet schema as the 2D dataset (`dataset.py`), one row per atom,
PLUS (optionally) `coord_x` / `coord_y` / `coord_z` columns. If present,
they are used as the molecule's 3D geometry; if absent, a conformer is
generated once with RDKit (see `features_3d._generate_conformer`) and
cached — see `features_3d.py` module docstring.

Caching mirrors `dataset.py` exactly (same on-disk cache directory
convention, different cache-key namespace so 2D and 3D caches never
collide even if pointed at the same `cache_dir`).
"""
from __future__ import annotations
import hashlib
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from .constants import ELEMENT_TO_Z
from .constants_3d import CUTOFF, MAX_NUM_NEIGHBORS
from .features_3d import precompute_molecule_tensors_3d, MissingCoordinatesError

logger = logging.getLogger(__name__)

_COORD_COLS = ("coord_x", "coord_y", "coord_z")


class Data3D(Data):
    """`torch_geometric.data.Data` subclass that teaches `Batch.from_data_list`
    how to correctly offset our custom triplet/torsion index tensors when
    collating several molecules into one batch.

    By default, PyG only auto-increments attributes whose *name* contains
    "index" or "face" (see `Data.__inc__`). Our triplet indices
    (`tri_idx_i/j/k`, `tor_idx_l` — node ids; `tri_idx_kj/ji` — edge ids)
    don't match that name heuristic, so without this override, batching
    would silently concatenate PER-MOLECULE-LOCAL node/edge indices without
    shifting them by the cumulative node/edge count of previous molecules
    in the batch — i.e. every 3D geometric model using triplets/torsions
    (DimeNet, DimeNet++, SphereNet) would silently read garbage
    cross-molecule indices for every molecule after the first one in a
    batch. This is exactly the class of silent batching bug
    `tests/test_forward_shapes.py`'s batch-invariance check (mirrored for
    3D in `tests/test_forward_shapes_3d.py`) is designed to catch.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("tri_idx_i", "tri_idx_j", "tri_idx_k", "tor_idx_l"):
            return self.num_nodes
        if key in ("tri_idx_kj", "tri_idx_ji"):
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class ChaosParquet3DDataset(Dataset):
    """
    Args:
        parquet_path:      path to a train/val/test parquet file.
        cutoff:             radius-graph cutoff (Angstrom), shared by all
                            3D models (see constants_3d.py).
        max_num_neighbors:  per-atom neighbour cap for the radius graph.
        cache_dir:          directory for the on-disk cache.
        force_recompute:    ignore any existing cache and rebuild it.
        require_provided_coords: DEFAULT True. If True, `coord_x`/
            `coord_y`/`coord_z` MUST be present and valid for every
            molecule in the parquet file — RDKit conformer generation is
            NEVER used, and the dataset raises `MissingCoordinatesError`
            (fails fast, at build time) if any molecule is missing valid
            coordinates. Set to False ONLY if you deliberately want
            missing/invalid coordinates to be silently filled in with a
            freshly RDKit-generated (ETKDGv3+MMFF94) conformer instead of
            raising — see `src/data/features_3d.py` docstring for details
            on that fallback.
        compute_triplets: DEFAULT True (safe/complete default). Set to
            False if you are ONLY training architectures that don't
            consume triplet/torsion indices — i.e. anything EXCEPT
            DimeNet/DimeNet++/SphereNet (SchNet, PaiNN, EGNN, TorchMD-Net,
            MACE, Uni-Mol). This skips triplet/torsion computation
            entirely and stores empty (0-length) tensors for them, which
            can shrink the on-disk cache by an order of magnitude for
            dense molecules/large datasets — triplet counts scale as
            roughly (avg_degree)^2 per atom, so they are usually the
            single largest contributor to cache size once cutoff/dataset
            size grow. `src/train_3d.py` sets this automatically based on
            which model you're training (see `_MODELS_NEEDING_TRIPLETS`
            there) — you only need to set it by hand if you're
            instantiating this dataset directly, e.g. in a notebook.
            NOTE: switching this flag invalidates any existing cache (it
            is part of the cache key, see `_cache_key`), so the dataset
            will rebuild from scratch the first time you change it.
    """

    def __init__(
        self,
        parquet_path: str,
        cutoff: float = CUTOFF,
        max_num_neighbors: int = MAX_NUM_NEIGHBORS,
        cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        require_provided_coords: bool = True,
        compute_triplets: bool = True,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.require_provided_coords = require_provided_coords
        self.compute_triplets = compute_triplets

        cache_dir = cache_dir or os.path.join(os.path.dirname(parquet_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._cache_key()
        cache_path = os.path.join(
            cache_dir, f"{os.path.basename(parquet_path)}.{cache_key}.pt",
        )

        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading cached dataset [3d_pure]: {cache_path}")
            payload = torch.load(cache_path, weights_only=False)
            self._data_list = payload["data_list"]
            self.sigma_cols = payload["sigma_cols"]
            self.mol_ids = payload["mol_ids"]
            self.stats = payload["stats"]
        else:
            self._data_list, self.sigma_cols, self.mol_ids, self.stats = self._build(parquet_path)
            logger.info(f"Caching dataset [3d_pure]: {cache_path}")
            torch.save(
                dict(data_list=self._data_list, sigma_cols=self.sigma_cols,
                     mol_ids=self.mol_ids, stats=self.stats),
                cache_path,
            )

        logger.info(
            f"{os.path.basename(parquet_path)} [3d_pure]: "
            f"{len(self._data_list)} mols, {self.stats['n_atoms']} atoms, "
            f"valid geometry: {self.stats['mol_valid']}/{self.stats['n_mols']} "
            f"({100 * self.stats['mol_valid'] / max(self.stats['n_mols'], 1):.1f}%), "
            f"generated conformers: {self.stats['n_generated_conformers']} "
            f"(coord_x/y/z {'present' if self.stats['has_coord_cols'] else 'ABSENT'} in parquet)"
        )

    def _cache_key(self) -> str:
        raw = (f"mode=3d_pure_cutoff={self.cutoff}_maxnn={self.max_num_neighbors}_"
               f"require_coords={self.require_provided_coords}_"
               f"triplets={self.compute_triplets}_v2")
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def _build(self, parquet_path: str):
        logger.info(f"Building [3d_pure] dataset from {parquet_path} (no cache found)")
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
                "Even the 3D benchmark requires a 'smiles' column, used as "
                "the fallback conformer-generation source when coord_x/y/z "
                "are absent or invalid for a given molecule."
            )

        has_coord_cols = all(c in df.columns for c in _COORD_COLS)
        if self.require_provided_coords and not has_coord_cols:
            raise MissingCoordinatesError(
                f"require_provided_coords=True (the default) but "
                f"{parquet_path} has no coord_x/coord_y/coord_z columns at "
                f"all. Either add real coordinates to your parquet file, "
                f"or explicitly pass require_provided_coords=False to "
                f"ChaosParquet3DDataset (train_3d.py) if you want RDKit to "
                f"generate conformers instead — see features_3d.py "
                f"docstring."
            )

        data_list, n_atoms_total = [], 0
        mol_valid_count, n_generated = 0, 0

        grouped = df.groupby("mol_id", sort=False)
        for mol_id, mol_df in grouped:
            mol_df = mol_df.sort_values("atom_index")
            z_list = [ELEMENT_TO_Z.get(e, 6) for e in mol_df["element"]]
            y = torch.nan_to_num(
                torch.tensor(mol_df[sigma_cols].values, dtype=torch.float), nan=0.0,
            )
            smiles = mol_df["smiles"].iloc[0]

            coords = None
            if has_coord_cols:
                coords = mol_df[list(_COORD_COLS)].to_numpy(dtype=np.float32)

            pre = precompute_molecule_tensors_3d(
                z_list, smiles, coords=coords,
                cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
                require_provided_coords=self.require_provided_coords,
                mol_id=mol_id,
                compute_triplets=self.compute_triplets,
            )

            trip = pre["triplets"]
            tors = pre["torsions"]
            data = Data3D(
                z=pre["z"], y=y, x=pre["x"], pos=pre["pos"],
                edge_index=pre["edge_index"], edge_weight=pre["edge_weight"],
                tri_idx_i=trip["idx_i"], tri_idx_j=trip["idx_j"], tri_idx_k=trip["idx_k"],
                tri_idx_kj=trip["idx_kj"], tri_idx_ji=trip["idx_ji"],
                tor_idx_l=tors["idx_l"], tor_has_l=tors["has_l"],
                num_nodes=len(z_list),
            )

            data_list.append(data)
            n_atoms_total += len(z_list)
            mol_valid_count += int(pre["mol_valid"])
            n_generated += int(pre["generated_conformer"])

        stats = dict(
            n_mols=len(mol_ids), n_atoms=n_atoms_total, mol_valid=mol_valid_count,
            n_generated_conformers=n_generated, has_coord_cols=has_coord_cols,
        )
        return data_list, sigma_cols, mol_ids, stats

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def bin_weights_numpy(self):
        from .constants import EPS
        all_y = torch.cat([d.y for d in self._data_list], dim=0).numpy()
        var = np.var(all_y, axis=0)
        var = np.maximum(var, EPS)
        return (var / var.sum()).astype("float32")
