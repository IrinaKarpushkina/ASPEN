"""
features_2d.py — построение ЧИСТО 2D признаков из SMILES/RDKit Mol.

Принципиальное отличие от features.py:
  - Граф строится из химических связей RDKit (не radius-graph по координатам)
  - Edge features содержат ТОЛЬКО топологическую информацию (тип связи,
    ароматичность, кольцо) — БЕЗ межатомных расстояний и RBF
  - Node features не содержат degree из radius-graph
  - 3D-координаты НЕ используются вообще

Это позволяет честно сравнить:
  2D-pure:  граф из SMILES, признаки только из топологии
  3D:       radius-graph по координатам, RBF-расстояния в edge_attr
  SchNet:   3D continuous-filter convolutions

Размерности:
  node features: 14-dim (без degree, иначе идентично extended)
  edge features: 7-dim (bond type 4 + aromatic + conjugated + in_ring)
"""

from __future__ import annotations
from typing import Optional, Sequence

import torch

from .constants import (
    ATOMIC_MASS, ELECTRONEGATIVITY, ENEG_DEFAULT,
    HB_ACCEPTOR_Z, HB_DONOR_Z, MASS_DEFAULT, MAX_Z,
    VDW_DEFAULT, VDW_RADII,
)
from .features import (
    POLARIZABILITY, POLARIZABILITY_DEFAULT,
    _HYBRID_ORDER, _BOND_TYPES,
    _RDKIT, mol_from_smiles,
)

# Размерности 2D feature-сетов
N_NODE_FEAT_2D = 14   # без degree (был индекс 4 в 15-dim extended)
N_EDGE_FEAT_2D = 7    # bond type (4) + aromatic + conjugated + in_ring

# Для совместимости с node_feat_dim() / edge_feat_dim() в моделях
# эти константы импортируются в constants.py через патч ниже


def build_node_features_2d(
    z_list: Sequence[int],
    mol: Optional["Chem.Mol"] = None,
) -> torch.Tensor:
    """
    Возвращает (N, 14) float tensor — purely 2D node features.

    Признаки (порядок такой же как в 15-dim extended, но без degree):
      0   electronegativity  — Pauling / 4
      1   vdW radius         — Å / 2.5
      2   atomic number Z    — / MAX_Z
      3   atomic mass        — / 200
      # degree отсутствует (был бы индекс 4 — зависит от 3D radius-graph)
      4   HB donor           — binary
      5   HB acceptor        — binary
      6   hybridisation SP   — one-hot (RDKit)
      7   hybridisation SP2  — one-hot (RDKit)
      8   hybridisation SP3  — one-hot (RDKit)
      9   formal charge      — raw int (RDKit)
      10  aromatic           — binary (RDKit)
      11  in ring            — binary (RDKit)
      12  Gasteiger charge   — raw float (RDKit)
      13  polarizability     — Å³ / 6.0
    """
    import numpy as np
    n     = len(z_list)
    feats = torch.zeros(n, N_NODE_FEAT_2D)

    for i, z in enumerate(z_list):
        feats[i, 0] = ELECTRONEGATIVITY.get(z, ENEG_DEFAULT) / 4.0
        feats[i, 1] = VDW_RADII.get(z, VDW_DEFAULT) / 2.5
        feats[i, 2] = min(z, MAX_Z) / float(MAX_Z)
        feats[i, 3] = ATOMIC_MASS.get(z, MASS_DEFAULT) / 200.0
        feats[i, 4] = 1.0 if z in HB_DONOR_Z    else 0.0
        feats[i, 5] = 1.0 if z in HB_ACCEPTOR_Z else 0.0
        feats[i, 13] = POLARIZABILITY.get(z, POLARIZABILITY_DEFAULT) / 6.0

    # RDKit-признаки (гибридизация, заряд, ароматичность, кольцо, Gasteiger)
    if mol is not None and _RDKIT:
        ri = mol.GetRingInfo()
        for i in range(min(n, mol.GetNumAtoms())):
            atom = mol.GetAtomWithIdx(i)
            hyb  = str(atom.GetHybridization())
            for k, name in enumerate(_HYBRID_ORDER):
                if hyb == name:
                    feats[i, 6 + k] = 1.0
            feats[i, 9]  = float(atom.GetFormalCharge())
            feats[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
            feats[i, 11] = 1.0 if ri.NumAtomRings(i) > 0 else 0.0
            try:
                import numpy as np
                gc = float(atom.GetProp("_GasteigerCharge"))
                feats[i, 12] = gc if np.isfinite(gc) else 0.0
            except (KeyError, ValueError):
                pass

    return feats


def build_edge_index_2d(mol: "Chem.Mol") -> torch.Tensor:
    """
    Строит edge_index (2, 2*num_bonds) из химических связей RDKit.
    Каждая связь добавляется в обоих направлениях (i→j и j→i).
    """
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


def build_edge_features_2d(
    mol: "Chem.Mol",
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Возвращает (E, 7) float tensor — purely 2D edge features.

    Признаки:
      0–3  bond type one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
      4    aromatic bond
      5    conjugated bond
      6    in ring

    Нет RBF расстояний — никакой 3D информации.
    """
    E     = edge_index.size(1)
    feats = torch.zeros(E, N_EDGE_FEAT_2D)

    if mol is None or not _RDKIT or E == 0:
        return feats

    n_atoms  = mol.GetNumAtoms()
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
        feats[e, 4] = 1.0 if bond.GetIsAromatic()   else 0.0
        feats[e, 5] = 1.0 if bond.GetIsConjugated() else 0.0
        feats[e, 6] = 1.0 if bond.IsInRing()        else 0.0

    return feats


def precompute_molecule_tensors_2d(
    z_list: Sequence[int],
    smiles: str,
) -> dict:
    """
    Полный precompute для одной молекулы в 2D-режиме.
    Возвращает dict с ключами: x, edge_index, edge_attr, mol_valid.

    degree и pos не вычисляются — 3D-координаты не используются.
    """
    mol = mol_from_smiles(smiles) if smiles and _RDKIT else None
    mol_valid = mol is not None and mol.GetNumAtoms() == len(z_list)

    if not mol_valid:
        # Fallback: пустой граф, нулевые признаки
        x          = torch.zeros(len(z_list), N_NODE_FEAT_2D)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, N_EDGE_FEAT_2D)
        # degree тоже нули
        degree     = torch.zeros(len(z_list))
        return dict(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    degree=degree, mol_valid=False)

    x          = build_node_features_2d(z_list, mol)
    edge_index = build_edge_index_2d(mol)
    edge_attr  = build_edge_features_2d(mol, edge_index)
    # degree из химических связей (не из radius-graph)
    degree = torch.zeros(len(z_list))
    if edge_index.size(1) > 0:
        degree.scatter_add_(
            0, edge_index[0],
            torch.ones(edge_index.size(1))
        )

    return dict(x=x, edge_index=edge_index, edge_attr=edge_attr,
                degree=degree, mol_valid=True)
