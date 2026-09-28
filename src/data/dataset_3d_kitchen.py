"""Kitchen-sink 3D dataset: enhanced (22) geometric features + SASA (3, from
features_3d_sasa.py) + RDKit chemical node features (8 NEW dims: hybridisation
x3, formal charge, aromatic, in-ring, Gasteiger charge, degree) + a separate
chemical BOND graph (edge_index2 / edge_attr2), reusing src/data/features.py's
already-validated 2D pipeline (mol_from_smiles_validated / build_node_features
/ build_edge_index / build_edge_features).

Why 8 new dims and not all 14 of features.py's node vector: electronegativity,
vdW radius, Z, mass, HB-donor, HB-acceptor and polarizability are already in
the enhanced-22 features (same source tables, src/data/constants.py) -
concatenating them again is redundant. Hybridisation / formal charge /
aromaticity / ring membership / Gasteiger charge are NOT derivable from
coordinates alone (they come from SMILES valence/connectivity), so they are
the only genuinely new information this file adds on the node side.

Atom-order risk: `mol_valid` (data.mol_valid, one bool per molecule) says
whether RDKit's atom order matched atom_index for that molecule. When it does
not, features.py already returns an all-zero node vector and an EMPTY bond
graph for that molecule (not garbage) - the model has to be able to just
ignore an empty edge_index (see bond_branch.py). RUN
scripts/check_bond_graph_coverage.py FIRST and read its output before
trusting this feature on your data: if `mol_valid` is false for a large
fraction of molecules, the model will effectively fall back to the SASA-only
architecture for those molecules, silently.
"""
from __future__ import annotations

import logging

import torch

from .dataset_3d_sasa import SasaChaosParquet3DDataset
from .features import N_EDGE_FEAT_2D, precompute_molecule_tensors

logger = logging.getLogger(__name__)
KITCHEN_CACHE_VERSION = "kitchen_v1"
# indices into features.py's 14-dim node vector that are NOT already covered
# by the enhanced-22 features (see module docstring)
_NEW_DIMS = [6, 7, 8, 9, 10, 11, 12]  # hybrid x3, formal_charge, aromatic, ring, gasteiger
N_CHEM_NODE_DIM = len(_NEW_DIMS)


class KitchenChaosParquet3DDataset(SasaChaosParquet3DDataset):
    def __init__(self, parquet_path, *args, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        force = bool(kwargs.get("force_recompute", False))
        super().__init__(parquet_path, *args, **kwargs)

        cache_dir = cache_dir or __import__("os").path.dirname(parquet_path)
        cache_dir = __import__("os").path.join(cache_dir, "cache")
        __import__("os").makedirs(cache_dir, exist_ok=True)
        path = __import__("os").path.join(
            cache_dir, f"{__import__('os').path.basename(parquet_path)}.{KITCHEN_CACHE_VERSION}.pt")

        pre = None
        if __import__("os").path.exists(path) and not force:
            try:
                pre = torch.load(path, weights_only=False)
                if len(pre) != len(self._data_list):
                    logger.warning("bond-graph cache size mismatch, recomputing")
                    pre = None
            except Exception as e:
                logger.warning("could not read bond-graph cache (%s), recomputing", e)
                pre = None

        if pre is None:
            smiles_list = self._get_smiles_list()  # must exist on the base dataset; see note below
            pre = []
            n_valid = 0
            for k, d in enumerate(self._data_list):
                z = d.z.cpu().numpy().tolist()
                out = precompute_molecule_tensors(z, smiles_list[k])
                n_valid += int(out["mol_valid"])
                pre.append(out)
                if (k + 1) % 5000 == 0:
                    logger.info("  bond graph %d/%d (valid so far: %.1f%%)", k + 1, len(self._data_list),
                               100.0 * n_valid / (k + 1))
            logger.info("bond graph coverage: %d/%d molecules (%.1f%%) had a validated RDKit atom-order match",
                       n_valid, len(pre), 100.0 * n_valid / max(len(pre), 1))
            torch.save(pre, path)

        for d, p in zip(self._data_list, pre):
            chem = p["x"][:, _NEW_DIMS] if p["x"].numel() else torch.zeros(d.num_nodes, N_CHEM_NODE_DIM)
            d.x = torch.cat([d.x, chem], dim=1)
            d.bond_edge_index = p["edge_index"]
            d.bond_edge_attr = p["edge_attr"] if p["edge_attr"].numel() else torch.zeros(0, N_EDGE_FEAT_2D)
            d.mol_valid = torch.tensor([p["mol_valid"]])
        logger.info("Kitchen features added: x dim -> %d (+ bond graph)", self._data_list[0].x.shape[1])

    def _get_smiles_list(self):
        """SasaChaosParquet3DDataset/EnhancedChaosParquet3DDataset don't keep smiles on
        Data objects (see scripts/error_analysis_by_molecule.py's docstring for why).
        Re-read them from the parquet, in the SAME group order the base class used
        (groupby('mol_id', sort=False)) - this mirrors dataset_3d.py's own construction
        exactly, so index k here must line up with self._data_list[k]."""
        import pandas as pd
        df = pd.read_parquet(self.parquet_path)  # set by ChaosParquet3DDataset.__init__
        out = []
        for _, g in df.groupby("mol_id", sort=False):
            out.append(g["smiles"].iloc[0])
        return out
