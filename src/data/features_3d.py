"""
features_3d.py — 3D-only node/geometry feature builder.

Mirrors the structure of `src/data/features.py` (the 2D featurizer)
deliberately, so the 2D/3D comparison differs ONLY in what geometric
information each model gets — not in unrelated pipeline details. See
PROVENANCE.md, "3D benchmark" section, and the module docstring of
`features.py` (kept unmodified) for why this separation matters.

What this module builds, per molecule, ONCE (cached to disk by
`dataset_3d.py`, exactly like the 2D featurizer's cache):

  z          — (N,) atomic numbers
  pos        — (N, 3) 3D coordinates (Angstrom)
  x          — (N, 6) purely physical, NON-topological node features
               (electronegativity, vdW radius, Z, mass, H-bond donor,
               H-bond acceptor — see constants_3d.N_NODE_FEAT_3D). No bond
               order, hybridisation, aromaticity or ring membership: those
               are 2D/topological concepts and belong only to features.py.
  edge_index, edge_weight
             — a single shared cutoff radius graph (constants_3d.CUTOFF /
               MAX_NUM_NEIGHBORS), consumed by every 3D architecture in
               this benchmark.
  triplets   — dict of (k -> j -> i) indices for DimeNet/DimeNet++/SphereNet
               (see geometry_3d.build_triplets).
  torsions   — dict extending triplets with a 4th atom l for SphereNet's
               dihedral term (see geometry_3d.build_torsions).

Coordinate source
------------------
If the parquet file has `coord_x`/`coord_y`/`coord_z` columns (the ASPEN
dataset does — see `configs/3d/base.yaml`), those are used directly (one
already-generated conformer per molecule, consistent with how the
sigma-profiles themselves were presumably computed from a single
representative conformer). If they are absent, this module falls back to
generating ONE conformer per molecule with RDKit's ETKDGv3 + MMFF94
optimisation, which is the standard, widely-used, deterministic (fixed
random seed) way to get 3D coordinates from a SMILES string when no
experimental/QM geometry is available. This fallback is logged and counted
in dataset stats (`stats['n_generated_conformers']`) so it is visible, not
silent, when it is used.
"""
from __future__ import annotations
import logging
from typing import Optional, Sequence

import numpy as np
import torch

from .constants import (
    ATOMIC_MASS, ELECTRONEGATIVITY, ENEG_DEFAULT,
    HB_ACCEPTOR_Z, HB_DONOR_Z, MASS_DEFAULT, MAX_Z,
    VDW_DEFAULT, VDW_RADII, ELEMENT_TO_Z,
)
from .constants_3d import CUTOFF, MAX_NUM_NEIGHBORS, N_NODE_FEAT_3D
from .geometry_3d import radius_graph_single, build_triplets, build_torsions

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False
    logger.warning("RDKit not found — this benchmark cannot run without it.")

_CONFORMER_SEED = 0xA59E7  # fixed seed -> deterministic fallback conformers


class MissingCoordinatesError(ValueError):
    """Raised by `precompute_molecule_tensors_3d` when
    `require_provided_coords=True` and no valid coordinates were found for
    a molecule — see that function's docstring."""


# ─────────────────────────────────────────────────────────────────────────────
# Node features (6-dim, no topology — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
def build_node_features_3d(z_list: Sequence[int]) -> torch.Tensor:
    n = len(z_list)
    feats = torch.zeros(n, N_NODE_FEAT_3D)
    for i, z in enumerate(z_list):
        feats[i, 0] = ELECTRONEGATIVITY.get(z, ENEG_DEFAULT) / 4.0
        feats[i, 1] = VDW_RADII.get(z, VDW_DEFAULT) / 2.5
        feats[i, 2] = min(z, MAX_Z) / float(MAX_Z)
        feats[i, 3] = ATOMIC_MASS.get(z, MASS_DEFAULT) / 200.0
        feats[i, 4] = 1.0 if z in HB_DONOR_Z else 0.0
        feats[i, 5] = 1.0 if z in HB_ACCEPTOR_Z else 0.0
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate fallback: RDKit ETKDGv3 + MMFF94 conformer generation
# ─────────────────────────────────────────────────────────────────────────────
def _generate_conformer(smiles: str, z_list: Sequence[int]) -> Optional[np.ndarray]:
    """Returns an (N, 3) numpy array of coordinates matching z_list's atom
    order, or None if embedding/optimisation fails or atom order can't be
    validated. Atom-order validation follows the same element-symbol check
    as `features.validate_atom_order` (2D), to avoid silently misassigning
    coordinates to the wrong atom."""
    if not _RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() != len(z_list):
            return None

        z_to_symbol = {v: k for k, v in ELEMENT_TO_Z.items()}
        for i, z in enumerate(z_list):
            if mol.GetAtomWithIdx(i).GetSymbol() != z_to_symbol.get(z, "C"):
                return None

        params = AllChem.ETKDGv3()
        params.randomSeed = _CONFORMER_SEED
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            # retry once with random coords as a fallback embedding strategy
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
            if cid < 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            except Exception:
                pass  # keep the embedded (unoptimised) geometry

        conf = mol.GetConformer(cid)
        return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                         dtype=np.float32)
    except Exception as e:
        logger.debug(f"_generate_conformer failed for '{smiles}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Full per-molecule precomputation — single entry point used by dataset_3d.py
# ─────────────────────────────────────────────────────────────────────────────
def precompute_molecule_tensors_3d(
    z_list: Sequence[int],
    smiles: str,
    coords: Optional[np.ndarray] = None,
    cutoff: float = CUTOFF,
    max_num_neighbors: int = MAX_NUM_NEIGHBORS,
    require_provided_coords: bool = False,
    mol_id: Optional[str] = None,
    compute_triplets: bool = True,
) -> dict:
    """
    Args:
        z_list: atomic numbers, one per atom, in the dataset's atom order.
        smiles: SMILES for the whole molecule (used only for the RDKit
            conformer-generation fallback when `coords` is None/invalid,
            AND ONLY IF `require_provided_coords` is False).
        coords: (N, 3) array of coordinates already in the parquet file
            (coord_x/coord_y/coord_z columns), or None to trigger the
            fallback conformer generator (unless disabled, see below).
        require_provided_coords: if True, NEVER fall back to RDKit
            conformer generation. If `coords` is missing/invalid for a
            molecule, raise `MissingCoordinatesError` immediately instead
            of silently generating a geometry. Use this whenever you
            specifically want every molecule trained on the SAME
            coordinates that are already in your parquet file (e.g.
            DFT/MD-derived conformers), not a freshly RDKit-generated one
            — see `ChaosParquet3DDataset(..., require_provided_coords=True)`.
        mol_id: optional molecule identifier, used only to make the error
            message actionable when `require_provided_coords=True`.
        compute_triplets: ONLY DimeNet, DimeNet++ and SphereNet consume
            triplet/torsion indices (see `src/models/models_3d/dimenet.py`,
            `spherenet.py`) — every other architecture (SchNet, PaiNN,
            EGNN, TorchMD-Net, MACE, Uni-Mol) never reads them. For dense
            molecules, triplet counts can be tens to ~100x the atom count
            (each is 5 index tensors), so computing/caching them
            unconditionally for architectures that don't need them wastes
            both build time and disk space (this was found to bloat the
            on-disk cache to tens of GB for large datasets — see
            `ChaosParquet3DDataset`'s own `compute_triplets` parameter,
            which is what actually controls this in practice; the default
            here is True only so this function is safe/complete to call
            standalone). When False, `triplets`/`torsions` in the
            returned dict are empty (0-length) tensors.

    Returns a dict with keys:
        x, z, pos, edge_index, edge_weight, triplets (dict), torsions (dict),
        mol_valid (bool), generated_conformer (bool)
    """
    n = len(z_list)
    generated = False

    pos_np = None
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape == (n, 3) and np.isfinite(coords).all():
            pos_np = coords

    if pos_np is None and require_provided_coords:
        raise MissingCoordinatesError(
            f"require_provided_coords=True but no valid coord_x/coord_y/"
            f"coord_z were found for molecule "
            f"{'`' + str(mol_id) + '`' if mol_id is not None else '(unknown mol_id)'} "
            f"(smiles='{smiles}', {n} atoms). Either fix/complete the "
            f"coordinates in your parquet file for this molecule, or "
            f"explicitly allow the RDKit ETKDGv3+MMFF94 fallback by "
            f"passing require_provided_coords=False "
            f"(ChaosParquet3DDataset default)."
        )

    if pos_np is None:
        pos_np = _generate_conformer(smiles, z_list)
        generated = pos_np is not None

    _EMPTY_TRIPLETS = dict(idx_i=torch.zeros(0, dtype=torch.int32),
                           idx_j=torch.zeros(0, dtype=torch.int32),
                           idx_k=torch.zeros(0, dtype=torch.int32),
                           idx_kj=torch.zeros(0, dtype=torch.int32),
                           idx_ji=torch.zeros(0, dtype=torch.int32))
    _EMPTY_TORSIONS = dict(idx_l=torch.zeros(0, dtype=torch.int32),
                          has_l=torch.zeros(0, dtype=torch.bool))

    if pos_np is None:
        return dict(
            x=torch.zeros(n, N_NODE_FEAT_3D),
            z=torch.tensor(z_list, dtype=torch.long),
            pos=torch.zeros(n, 3),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_weight=torch.zeros(0),
            triplets=_EMPTY_TRIPLETS,
            torsions=_EMPTY_TORSIONS,
            mol_valid=False,
            generated_conformer=False,
        )

    pos = torch.tensor(pos_np, dtype=torch.float)
    x = build_node_features_3d(z_list)
    z = torch.tensor(z_list, dtype=torch.long)

    edge_index, edge_weight = radius_graph_single(pos, cutoff, max_num_neighbors)

    if compute_triplets:
        triplets = build_triplets(edge_index, n)
        torsions = build_torsions(edge_index, triplets, n)
        # int32 is plenty (molecules have nowhere near 2^31 atoms/edges) and
        # halves the on-disk/in-memory footprint of the triplet/torsion
        # tensors relative to PyTorch's int64 default — these are usually
        # the single largest contributor to cache size for DimeNet/
        # DimeNet++/SphereNet (see docstring above).
        triplets = {k: v.to(torch.int32) for k, v in triplets.items()}
        torsions = {"idx_l": torsions["idx_l"].to(torch.int32), "has_l": torsions["has_l"]}
    else:
        triplets = _EMPTY_TRIPLETS
        torsions = _EMPTY_TORSIONS

    return dict(
        x=x, z=z, pos=pos,
        edge_index=edge_index, edge_weight=edge_weight,
        triplets=triplets, torsions=torsions,
        mol_valid=True, generated_conformer=generated,
    )


def node_feat_dim_3d() -> int:
    return N_NODE_FEAT_3D


