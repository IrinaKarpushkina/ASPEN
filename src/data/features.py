"""
features.py — shared node/edge feature builders for ALL architectures.

Single source of truth. Any performance difference between models reflects
architecture (or loss in ablation), NOT feature inconsistencies.

Node features — two modes:
  BASE (7-dim):      eneg, vdw, Z, mass, degree, HB-donor, HB-acceptor
  EXTENDED (15-dim): BASE + hybridisation SP/SP2/SP3, formal charge,
                     aromatic, in-ring, Gasteiger charge, polarizability

Edge features:
  RBF (N_RBF-dim):   Gaussian RBF of interatomic distance (always available)
  EXTENDED:          RBF + bond-type one-hot (4) + aromatic + conjugated + in-ring
                     (requires RDKit mol — auto-zeroed if bond not in mol)
"""

from __future__ import annotations
import logging
from typing import Optional, Sequence
import numpy as np
import torch

from .constants import (
    ATOMIC_MASS, CUTOFF, ELECTRONEGATIVITY, ENEG_DEFAULT,
    HB_ACCEPTOR_Z, HB_DONOR_Z, MASS_DEFAULT, MAX_Z,
    N_NODE_FEAT_BASE, N_RBF, VDW_DEFAULT, VDW_RADII,
)

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT = True
except ImportError:
    _RDKIT = False
    logger.warning("RDKit not found — extended features will be zeros.")

# Static polarizability table (Angstrom^3)
POLARIZABILITY = {
    1: 0.667, 6: 1.760, 7: 1.100, 8: 0.802, 9: 0.557,
    14: 5.380, 15: 3.630, 16: 2.900, 17: 2.180,
    35: 3.050, 53: 5.350,
}
POLARIZABILITY_DEFAULT = 1.5

_HYBRID_ORDER  = ("SP", "SP2", "SP3")
_BOND_TYPES    = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

N_NODE_FEAT_EXTENDED = N_NODE_FEAT_BASE + 8   # 15
N_EDGE_FEAT_EXTRA    = len(_BOND_TYPES) + 3   # 7  (bond-type 4 + aromatic + conj + in-ring)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch radius graph (no pyg-lib / torch_cluster dependency).
#
# PyG's `radius_graph` requires the compiled `pyg-lib` or `torch_cluster`
# extensions, which can be fragile to install/match against a given CUDA/
# torch build. Since this is a one-time CPU precompute step (cached to disk,
# see dataset.py) and molecules here are small (tens to low hundreds of
# atoms), an O(N^2) cdist-based implementation is fast enough and removes a
# fragile dependency from the whole pipeline.
#
# Semantics match torch_geometric.nn.radius_graph(loop=False):
#   - edge_index[0] (row) = query/source node i
#   - edge_index[1] (col) = neighbour j, with dist(i,j) <= r, j != i
#   - at most `max_num_neighbors` closest neighbours kept per row-node i
#     (this cap is IDENTICAL across all models — see constants.MAX_NUM_NEIGHBORS)
# ─────────────────────────────────────────────────────────────────────────────
def radius_graph_pure(
    pos: torch.Tensor,
    r: float,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> torch.Tensor:
    n = pos.size(0)
    if n == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    dist = torch.cdist(pos, pos)  # (n, n)
    if not loop:
        dist.fill_diagonal_(float("inf"))

    within = dist <= r

    rows, cols = [], []
    for i in range(n):
        idx = torch.nonzero(within[i], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() > max_num_neighbors:
            d = dist[i, idx]
            top = torch.topk(d, max_num_neighbors, largest=False).indices
            idx = idx[top]
        rows.append(torch.full((idx.numel(),), i, dtype=torch.long))
        cols.append(idx)

    if not rows:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# RDKit mol builder from SMILES (cached per molecule)
# ─────────────────────────────────────────────────────────────────────────────
def mol_from_smiles(smiles: str) -> Optional["Chem.Mol"]:
    """Return sanitized RDKit Mol (WITH explicit Hs, since z_list includes H
    atoms) or None on failure."""
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
# NODE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_node_features_base(z_list: Sequence[int], degree: torch.Tensor) -> torch.Tensor:
    """Returns (N, 7) float tensor — BASE feature set."""
    n      = len(z_list)
    feats  = torch.zeros(n, N_NODE_FEAT_BASE)
    deg_np = degree.detach().cpu().numpy()
    for i, z in enumerate(z_list):
        feats[i, 0] = ELECTRONEGATIVITY.get(z, ENEG_DEFAULT) / 4.0
        feats[i, 1] = VDW_RADII.get(z, VDW_DEFAULT)          / 2.5
        feats[i, 2] = min(z, MAX_Z)                           / float(MAX_Z)
        feats[i, 3] = ATOMIC_MASS.get(z, MASS_DEFAULT)        / 200.0
        feats[i, 4] = float(deg_np[i])                        / 10.0
        feats[i, 5] = 1.0 if z in HB_DONOR_Z    else 0.0
        feats[i, 6] = 1.0 if z in HB_ACCEPTOR_Z else 0.0
    return feats


def _extended_block(mol: Optional["Chem.Mol"], n: int) -> torch.Tensor:
    """Returns (N, 8) — RDKit-derived descriptors, zeros if mol is None."""
    feats = torch.zeros(n, 8)
    if mol is None or not _RDKIT:
        return feats
    ri = mol.GetRingInfo()
    for i in range(min(n, mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        hyb  = str(atom.GetHybridization())
        for k, name in enumerate(_HYBRID_ORDER):
            if hyb == name:
                feats[i, k] = 1.0
        feats[i, 3] = float(atom.GetFormalCharge())
        feats[i, 4] = 1.0 if atom.GetIsAromatic() else 0.0
        feats[i, 5] = 1.0 if ri.NumAtomRings(i) > 0 else 0.0
        try:
            gc = float(atom.GetProp("_GasteigerCharge"))
            feats[i, 6] = gc if np.isfinite(gc) else 0.0
        except (KeyError, ValueError):
            pass
        z = atom.GetAtomicNum()
        feats[i, 7] = POLARIZABILITY.get(z, POLARIZABILITY_DEFAULT) / 6.0
    return feats


def build_node_features(
    z_list:       Sequence[int],
    degree:       torch.Tensor,
    mol:          Optional["Chem.Mol"] = None,
    use_extended: bool = True,
) -> torch.Tensor:
    """
    Single entry point for all models.
    Returns (N, 7) or (N, 15) depending on use_extended.
    """
    base = build_node_features_base(z_list, degree)
    if not use_extended:
        return base
    return torch.cat([base, _extended_block(mol, len(z_list))], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_features_rbf(
    pos:        torch.Tensor,
    edge_index: torch.Tensor,
    n_rbf:  int   = N_RBF,
    cutoff: float = CUTOFF,
) -> torch.Tensor:
    """Returns (E, n_rbf) Gaussian RBF features."""
    row, col = edge_index
    dist     = (pos[col] - pos[row]).norm(dim=-1, keepdim=True)
    centres  = torch.linspace(0.0, cutoff, n_rbf, device=pos.device)
    sigma    = cutoff / n_rbf
    return torch.exp(-((dist - centres) ** 2) / (2.0 * sigma ** 2))


def _edge_extended_block(
    mol:        Optional["Chem.Mol"],
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Returns (E, 7) bond-topology features, zeros if mol is None."""
    E     = edge_index.size(1)
    feats = torch.zeros(E, N_EDGE_FEAT_EXTRA)
    if mol is None or not _RDKIT:
        return feats
    n_atoms   = mol.GetNumAtoms()
    row, col  = edge_index.cpu().numpy()
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
        feats[e, 4] = 1.0 if bond.GetIsAromatic()   else 0.0
        feats[e, 5] = 1.0 if bond.GetIsConjugated() else 0.0
        feats[e, 6] = 1.0 if bond.IsInRing()        else 0.0
    return feats


def build_edge_features(
    pos:          torch.Tensor,
    edge_index:   torch.Tensor,
    mol:          Optional["Chem.Mol"] = None,
    n_rbf:        int   = N_RBF,
    cutoff:       float = CUTOFF,
    use_extended: bool  = True,
) -> torch.Tensor:
    """
    Single entry point for all models that consume edge features.
    Returns (E, n_rbf) or (E, n_rbf+7) depending on use_extended.
    """
    rbf = build_edge_features_rbf(pos, edge_index, n_rbf=n_rbf, cutoff=cutoff)
    if not use_extended:
        return rbf
    extra = _edge_extended_block(mol, edge_index).to(rbf.device)
    return torch.cat([rbf, extra], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience getters (for model __init__ size calculations)
# ─────────────────────────────────────────────────────────────────────────────
def node_feat_dim(use_extended: bool, mode: str = "3d") -> int:
    if mode == "2d_pure":
        from .features_2d import N_NODE_FEAT_2D
        return N_NODE_FEAT_2D
    return N_NODE_FEAT_EXTENDED if use_extended else N_NODE_FEAT_BASE
 
def edge_feat_dim(use_extended: bool, n_rbf: int = N_RBF, mode: str = "3d") -> int:
    if mode == "2d_pure":
        from .features_2d import N_EDGE_FEAT_2D
        return N_EDGE_FEAT_2D
    return (n_rbf + N_EDGE_FEAT_EXTRA) if use_extended else n_rbf

# ─────────────────────────────────────────────────────────────────────────────
# Atom-order validation — CRITICAL for correctness of extended features.
#
# RDKit's atom ordering after Chem.MolFromSmiles(...).AddHs() may NOT match
# the atom_index ordering used when the 3D conformer / sigma-profile was
# computed upstream (e.g. by a separate xtb/conformer pipeline). If the
# orderings differ, per-atom RDKit features (hybridisation, formal charge,
# Gasteiger charge, bond types) would be silently misassigned to the wrong
# atoms — a serious correctness bug for a paper.
#
# We validate by comparing element symbols at each index; if they don't
# match exactly, the mol is rejected (treated as None -> zeros) and counted
# in DatasetStats.mol_mismatch for reporting.
# ─────────────────────────────────────────────────────────────────────────────
def validate_atom_order(mol: Optional["Chem.Mol"], z_list: Sequence[int]) -> bool:
    """Returns True iff mol's atom count and element order match z_list exactly."""
    if mol is None or not _RDKIT:
        return False
    if mol.GetNumAtoms() != len(z_list):
        return False
    from .constants import ELEMENT_TO_Z
    z_to_symbol = {v: k for k, v in ELEMENT_TO_Z.items()}
    for i, z in enumerate(z_list):
        expected_symbol = z_to_symbol.get(z, "C")
        actual_symbol   = mol.GetAtomWithIdx(i).GetSymbol()
        if actual_symbol != expected_symbol:
            return False
    return True


def mol_from_smiles_validated(smiles: str, z_list: Sequence[int]) -> Optional["Chem.Mol"]:
    """
    Build an RDKit Mol from SMILES and validate its atom ordering against
    z_list (the atom_index-ordered element list from the dataset).

    Returns the Mol only if the ordering matches exactly; otherwise None
    (caller should fall back to base features and log the mismatch).
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    if validate_atom_order(mol, z_list):
        return mol
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Full per-molecule precomputation — single entry point used by dataset.py
# ─────────────────────────────────────────────────────────────────────────────
def precompute_molecule_tensors(
    z_list:       Sequence[int],
    pos:          torch.Tensor,
    smiles:       str,
    use_extended: bool = True,
    n_rbf:        int  = N_RBF,
    cutoff:       float = CUTOFF,
    max_num_neighbors: int = 32,
):
    """
    Computes everything needed for a single molecule, once, at dataset
    construction time:

      x          — (N, 7) or (N, 15) node features
      edge_index — (2, E) radius-graph edges (loop=False, capped at
                   max_num_neighbors per node — IDENTICAL cap for ALL models)
      edge_attr  — (E, n_rbf) or (E, n_rbf+7) edge features
      degree     — (N,) float, #neighbours per atom in the radius graph
      mol_valid  — bool, whether RDKit mol matched atom ordering (for stats)

    Returns a dict with these five keys.
    """
    n = len(z_list)
    edge_index = radius_graph_pure(
        pos, r=cutoff, loop=False, max_num_neighbors=max_num_neighbors,
    )
    degree = torch.bincount(edge_index[0], minlength=n).float()

    mol, mol_valid = None, False
    if use_extended and smiles:
        mol = mol_from_smiles_validated(smiles, z_list)
        mol_valid = mol is not None

    x = build_node_features(z_list, degree, mol=mol, use_extended=use_extended)
    edge_attr = build_edge_features(
        pos, edge_index, mol=mol, n_rbf=n_rbf, cutoff=cutoff, use_extended=use_extended,
    )

    return {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "degree": degree, "mol_valid": mol_valid,
    }
