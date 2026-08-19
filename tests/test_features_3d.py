"""
test_features_3d.py — guards the 3D featurizer's core invariants:
purely physical (non-topological) node features, a valid radius graph,
and correctly-shaped triplet/torsion index tensors — mirroring
`tests/test_features_2d_only.py`'s role for the 2D featurizer.
"""
import numpy as np
import torch

from src.data import features_3d
from src.data.constants_3d import N_NODE_FEAT_3D


def test_ethanol_with_explicit_coords_has_expected_shapes_and_no_nans():
    z_list = [6, 6, 8, 1, 1, 1, 1, 1, 1]  # C, C, O, + 6 H (ethanol, "CCO", explicit Hs)
    # a plausible (non-planar, non-degenerate) coordinate set is all that's
    # required here -- we are testing plumbing/shapes, not chemistry.
    rng = np.random.RandomState(0)
    coords = rng.randn(len(z_list), 3).astype(np.float32) * 1.2

    out = features_3d.precompute_molecule_tensors_3d(z_list, "CCO", coords=coords)

    assert out["mol_valid"] is True
    assert out["generated_conformer"] is False  # explicit coords were used, not the RDKit fallback
    assert out["x"].shape == (len(z_list), N_NODE_FEAT_3D)
    assert out["pos"].shape == (len(z_list), 3)
    assert torch.isfinite(out["x"]).all()
    assert torch.isfinite(out["pos"]).all()
    assert out["edge_index"].shape[0] == 2
    assert out["edge_weight"].shape[0] == out["edge_index"].shape[1]
    # every triplet/torsion tensor must have IDENTICAL length
    trip = out["triplets"]
    lengths = {len(trip[k]) for k in ("idx_i", "idx_j", "idx_k", "idx_kj", "idx_ji")}
    assert len(lengths) == 1
    tors = out["torsions"]
    assert len(tors["idx_l"]) == len(tors["has_l"]) == next(iter(lengths))


def test_missing_coords_falls_back_to_rdkit_conformer_generation():
    z_list = [6, 6, 8, 1, 1, 1, 1, 1, 1]
    out = features_3d.precompute_molecule_tensors_3d(z_list, "CCO", coords=None)

    assert out["mol_valid"] is True
    assert out["generated_conformer"] is True
    assert out["pos"].shape == (len(z_list), 3)
    assert torch.isfinite(out["pos"]).all()
    # a real (generated) conformer should not collapse every atom to the
    # same point -- sanity check that SOME bond length is chemically
    # plausible (roughly 0.9-1.6 Angstrom for a C-H/C-C/C-O single bond).
    dists = torch.cdist(out["pos"], out["pos"])
    dists.fill_diagonal_(float("inf"))
    nearest = dists.min(dim=1).values
    assert (nearest > 0.5).all() and (nearest < 2.5).all()


def test_invalid_coord_shape_triggers_fallback_not_a_crash():
    z_list = [6, 1, 1, 1, 1]  # methane
    bad_coords = np.random.randn(3, 3).astype(np.float32)  # wrong atom count
    out = features_3d.precompute_molecule_tensors_3d(z_list, "C", coords=bad_coords)
    assert out["mol_valid"] is True
    assert out["generated_conformer"] is True  # bad coords rejected -> fallback used


def test_atom_order_mismatch_falls_back_to_empty_graph_not_silent_misassignment():
    wrong_z_list = [8, 8, 8]  # claims 3 oxygens; "CCO" clearly isn't that, and has 9 atoms with Hs
    out = features_3d.precompute_molecule_tensors_3d(wrong_z_list, "CCO", coords=None)
    assert out["mol_valid"] is False
    assert torch.equal(out["x"], torch.zeros(3, N_NODE_FEAT_3D))
    assert out["edge_index"].shape == (2, 0)
