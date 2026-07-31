"""
test_features_2d_only.py — guards the central invariant of this
repository: the graph and its features are built ONLY from SMILES/RDKit
chemical bonds, with no reliance on 3D coordinates.
"""
import inspect

import torch

from src.data import features
from src.data.constants import N_NODE_FEAT_2D, N_EDGE_FEAT_2D


def test_precompute_signature_has_no_coordinate_arguments():
    """The single entry point used by dataset.py must not accept pos/
    coordinates/cutoff/RBF-style arguments — if someone adds one back by
    accident while merging in 3D code later, this test should fail."""
    sig = inspect.signature(features.precompute_molecule_tensors)
    forbidden = {"pos", "coords", "coordinates", "cutoff", "n_rbf", "max_num_neighbors"}
    present = forbidden & set(sig.parameters)
    assert not present, (
        f"precompute_molecule_tensors() gained 3D-related argument(s) {present} "
        f"— this repository is 2D-only; 3D logic belongs in "
        f"src/models/models_3d/ and configs/3d/, not here."
    )


def test_ethanol_features_have_expected_shape_and_no_nans():
    z_list = [6, 6, 8, 1, 1, 1, 1, 1, 1]  # C, C, O, + 6 H (ethanol, CCO, with explicit Hs)
    smiles = "CCO"
    out = features.precompute_molecule_tensors(z_list, smiles)

    assert out["x"].shape == (len(z_list), N_NODE_FEAT_2D)
    assert out["edge_attr"].shape[1] == N_EDGE_FEAT_2D
    assert torch.isfinite(out["x"]).all()
    assert torch.isfinite(out["edge_attr"]).all()
    assert out["mol_valid"] is True
    # Ethanol has 3 heavy-atom bonds (C-C, C-O) + 6 C-H/O-H bonds = 8 bonds,
    # each stored in both directions -> 16 directed edges.
    assert out["edge_index"].shape[1] == 16


def test_atom_order_mismatch_falls_back_to_zero_graph_not_silent_misassignment():
    """If the element list doesn't match the SMILES-derived atom order,
    precompute_molecule_tensors must fall back to a zero graph (mol_valid
    = False) rather than silently assigning RDKit features to the wrong
    atom index."""
    wrong_z_list = [8, 8, 8]  # claims 3 oxygens; ethanol SMILES clearly isn't that
    out = features.precompute_molecule_tensors(wrong_z_list, "CCO")
    assert out["mol_valid"] is False
    assert torch.equal(out["x"], torch.zeros(3, N_NODE_FEAT_2D))
