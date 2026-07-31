"""
features.py — 2D-only node/edge feature builder, from SMILES/RDKit only.

This repository is intentionally 2D-ONLY:
  - The graph is built from RDKit chemical bonds (Chem.MolFromSmiles + AddHs),
    NOT from a radius graph over 3D coordinates.
  - Edge features contain ONLY topological information (bond type,
    aromaticity, ring membership) — no interatomic distances, no RBF.
  - Node features contain no quantity derived from a 3D radius graph
    (in particular: no "degree in the radius graph", since there is no
    radius graph here — degree is the number of chemical bonds instead,
    computed separately per model if a degree feature is desired).
  - 3D coordinates are never read by this module. If your parquet files
    contain coord_x/coord_y/coord_z columns, they are ignored entirely
    (dataset.py does not even look for them).

Why this separation matters (see PROVENANCE.md, section "2D vs 3D"):
a 2D/3D architecture benchmark is only meaningful if the *only* thing that
differs between the two regimes is the presence of geometric information —
not an accidental difference in graph construction, feature completeness,
or model plumbing. Keeping ALL 2D logic in this single file (no forking
into a shared 3D module) makes that boundary explicit and auditable.

Node features (14-dim), in order:
   0  electronegativity  — Pauling scale / 4
   1  van der Waals radius — Angstrom / 2.5
   2  atomic number Z    — / MAX_Z
   3  atomic mass        — amu / 200
   4  H-bond donor       — binary (N, O, F)
   5  H-bond acceptor    — binary (N, O, F, S, Cl)
   6  hybridisation SP   — one-hot (RDKit)
   7  hybridisation SP2  — one-hot (RDKit)
   8  hybridisation SP3  — one-hot (RDKit)
   9  formal charge      — raw int (RDKit)
  10  aromatic           — binary (RDKit)
  11  in ring            — binary (RDKit)
  12  Gasteiger charge   — raw float (RDKit, AllChem.ComputeGasteigerCharges)
  13  polarizability     — Angstrom^3 / 6.0

Edge features (7-dim), in order:
   0-3  bond type one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
   4    aromatic bond
   5    conjugated bond
   6    in ring
"""

from __future__ import annotations
import logging
from typing import Optional, Sequence

import numpy as np
import torch

from .constants import (
    ATOMIC_MASS, ELECTRONEGATIVITY, ENEG_DEFAULT,
    HB_ACCEPTOR_Z, HB_DONOR_Z, MASS_DEFAULT, MAX_Z,
    POLARIZABILITY, POLARIZABILITY_DEFAULT,
    VDW_DEFAULT, VDW_RADII,
    N_NODE_FEAT_2D, N_EDGE_FEAT_2D, _HYBRID_ORDER, _BOND_TYPES,
)

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False
    logger.warning("RDKit not found — this benchmark cannot run without it "
                    "(install rdkit>=2023.9, see requirements.txt).")


# ─────────────────────────────────────────────────────────────────────────────
# RDKit mol builder from SMILES
# ─────────────────────────────────────────────────────────────────────────────
def mol_from_smiles(smiles: str) -> Optional["Chem.Mol"]:
    """Return a sanitized RDKit Mol WITH explicit Hs (so atom count matches
    the per-atom rows in the dataset, which include H), or None on failure."""
    if not _RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol)
        return mol
    except Exception as e:
        logger.debug(f"mol_from_smiles failed for '{smiles}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Atom-order validation — CRITICAL for correctness.
#
# RDKit's atom ordering after Chem.MolFromSmiles(...).AddHs() may NOT match
# the atom_index ordering used elsewhere in the parquet file (e.g. produced
# by a separate conformer-generation pipeline). If the orderings differ,
# per-atom RDKit features (hybridisation, formal charge, bond types, ...)
# would be silently misassigned to the wrong atoms. We validate by comparing
# element symbols at each index; if they don't match exactly, we fall back
# to an empty/zero graph for that molecule and count it in dataset stats.
# ─────────────────────────────────────────────────────────────────────────────
def validate_atom_order(mol: Optional["Chem.Mol"], z_list: Sequence[int]) -> bool:
    if mol is None or not _RDKIT:
        return False
    if mol.GetNumAtoms() != len(z_list):
        return False
    from .constants import ELEMENT_TO_Z
    z_to_symbol = {v: k for k, v in ELEMENT_TO_Z.items()}
    for i, z in enumerate(z_list):
        expected_symbol = z_to_symbol.get(z, "C")
        actual_symbol = mol.GetAtomWithIdx(i).GetSymbol()
        if actual_symbol != expected_symbol:
            return False
    return True


def mol_from_smiles_validated(smiles: str, z_list: Sequence[int]) -> Optional["Chem.Mol"]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    if validate_atom_order(mol, z_list):
        return mol
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NODE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_node_features(z_list: Sequence[int], mol: Optional["Chem.Mol"] = None) -> torch.Tensor:
    """Returns (N, 14) float tensor. See module docstring for feature order."""
    n = len(z_list)
    feats = torch.zeros(n, N_NODE_FEAT_2D)

    for i, z in enumerate(z_list):
        feats[i, 0] = ELECTRONEGATIVITY.get(z, ENEG_DEFAULT) / 4.0
        feats[i, 1] = VDW_RADII.get(z, VDW_DEFAULT) / 2.5
        feats[i, 2] = min(z, MAX_Z) / float(MAX_Z)
        feats[i, 3] = ATOMIC_MASS.get(z, MASS_DEFAULT) / 200.0
        feats[i, 4] = 1.0 if z in HB_DONOR_Z else 0.0
        feats[i, 5] = 1.0 if z in HB_ACCEPTOR_Z else 0.0
        feats[i, 13] = POLARIZABILITY.get(z, POLARIZABILITY_DEFAULT) / 6.0

    if mol is not None and _RDKIT:
        ri = mol.GetRingInfo()
        for i in range(min(n, mol.GetNumAtoms())):
            atom = mol.GetAtomWithIdx(i)
            hyb = str(atom.GetHybridization())
            for k, name in enumerate(_HYBRID_ORDER):
                if hyb == name:
                    feats[i, 6 + k] = 1.0
            feats[i, 9] = float(atom.GetFormalCharge())
            feats[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
            feats[i, 11] = 1.0 if ri.NumAtomRings(i) > 0 else 0.0
            try:
                gc = float(atom.GetProp("_GasteigerCharge"))
                feats[i, 12] = gc if np.isfinite(gc) else 0.0
            except (KeyError, ValueError):
                pass

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# EDGE / GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_index(mol: "Chem.Mol") -> torch.Tensor:
    """Builds edge_index (2, 2*num_bonds) from RDKit chemical bonds.
    Each bond is added in both directions (i->j and j->i), matching the
    convention used by message-passing layers (undirected graph as two
    directed edges) throughout torch_geometric."""
    if mol is None:
        return torch.zeros(2, 0, dtype=torch.long)

    rows, cols = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        rows += [i, j]
        cols += [j, i]

    if not rows:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([rows, cols], dtype=torch.long)


def build_edge_features(mol: "Chem.Mol", edge_index: torch.Tensor) -> torch.Tensor:
    """Returns (E, 7) float tensor. See module docstring for feature order.
    No RBF, no distances — purely bond topology."""
    E = edge_index.size(1)
    feats = torch.zeros(E, N_EDGE_FEAT_2D)

    if mol is None or not _RDKIT or E == 0:
        return feats

    n_atoms = mol.GetNumAtoms()
    row, col = edge_index.cpu().numpy()

    for e in range(E):
        i, j = int(row[e]), int(col[e])
        if i >= n_atoms or j >= n_atoms:
            continue
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        bt = str(bond.GetBondType())
        for k, name in enumerate(_BOND_TYPES):
            if bt == name:
                feats[e, k] = 1.0
        feats[e, 4] = 1.0 if bond.GetIsAromatic() else 0.0
        feats[e, 5] = 1.0 if bond.GetIsConjugated() else 0.0
        feats[e, 6] = 1.0 if bond.IsInRing() else 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Full per-molecule precomputation — single entry point used by dataset.py
# ─────────────────────────────────────────────────────────────────────────────
def precompute_molecule_tensors(z_list: Sequence[int], smiles: str) -> dict:
    """
    Computes everything needed for a single molecule, once, at dataset
    construction time.

    Returns a dict with keys:
      x          — (N, 14) node features
      edge_index — (2, 2*num_bonds) chemical-bond graph
      edge_attr  — (E, 7) edge features (topology only)
      degree     — (N,) float, number of chemical bonds per atom
      mol_valid  — bool, whether RDKit mol matched atom ordering exactly
    """
    mol = mol_from_smiles_validated(smiles, z_list) if smiles else None
    mol_valid = mol is not None

    if not mol_valid:
        n = len(z_list)
        return dict(
            x=torch.zeros(n, N_NODE_FEAT_2D),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, N_EDGE_FEAT_2D),
            degree=torch.zeros(n),
            mol_valid=False,
        )

    x = build_node_features(z_list, mol)
    edge_index = build_edge_index(mol)
    edge_attr = build_edge_features(mol, edge_index)

    degree = torch.zeros(len(z_list))
    if edge_index.size(1) > 0:
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    return dict(x=x, edge_index=edge_index, edge_attr=edge_attr,
                degree=degree, mol_valid=True)


def node_feat_dim() -> int:
    return N_NODE_FEAT_2D


def edge_feat_dim() -> int:
    return N_EDGE_FEAT_2D
