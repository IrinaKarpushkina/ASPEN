"""
test_forward_shapes_3d.py — smoke test: every registered 3D architecture
must run a forward pass on a tiny toy batch and produce a finite
(N_atoms, 51) output, and must be invariant to which OTHER molecule
shares its batch (no cross-molecule leakage — this is exactly the class
of bug `src/data/dataset_3d.py::Data3D.__inc__` exists to prevent, see
that class's docstring).

Mirrors `tests/test_forward_shapes.py` (2D) in spirit and structure.
`num_spherical`/`num_radial` are kept small for DimeNet/DimeNet++/
SphereNet ONLY to keep this test fast — `SphericalBasisLayer`'s
constructor does real sympy symbolic work (building/lambdifying Bessel
and spherical-harmonic basis functions) that takes several seconds per
instantiation regardless of `hidden_channels`; this is a test-speed
concern only; production configs (`configs/3d/*.yaml`) use the papers'
own defaults (`num_spherical=7`, `num_radial=6`).

REVISION NOTE: toy molecules used to sample atomic numbers uniformly
from Z in [1, 10) (`torch.randint(1, 10, ...)`), which can produce
elements (e.g. He, Li) outside this dataset's real element vocabulary
(`models_3d.mace.ELEMENTS_PRESENT`, 25 elements). MACE's
`EquivariantProductBasisBlock` requires atom-type one-hot indices
(`z_to_index`, built from that real vocabulary), so a toy Z not in the
vocabulary raised `RuntimeError: Class values must be non-negative`
(from `F.one_hot` on a -1 "not found" index) -- not a bug in the model,
but a test fixture generating chemically-arbitrary atoms MACE was never
meant to handle. Fixed: toy `z` is now sampled from
`models_3d.mace.ELEMENTS_PRESENT` (mapped to real atomic numbers via
`constants.ELEMENT_TO_Z`), so every toy molecule is restricted to
elements that actually occur in this benchmark's dataset -- meaningful
for every architecture, not just MACE.
"""
import torch
from torch_geometric.data import Data, Batch

from src.models import MODEL_REGISTRY_3D
from src.data.constants import ELEMENT_TO_Z
from src.data.constants_3d import N_NODE_FEAT_3D
from src.models.models_3d.mace import ELEMENTS_PRESENT

_MODEL_KWARGS = {
    "schnet": dict(hidden_channels=32, num_filters=32, num_interactions=2),
    "painn": dict(hidden=32, num_layers=2, num_rbf=8),
    "dimenet": dict(hidden_channels=32, num_blocks=2, num_radial=4, num_spherical=3, pp=False),
    "dimenet_pp": dict(hidden_channels=32, out_emb_channels=32, num_blocks=2,
                       num_radial=4, num_spherical=3, pp=True),
    "spherenet": dict(hidden_channels=32, out_emb_channels=32, num_blocks=2,
                      num_radial=4, num_spherical=3),
    "egnn": dict(hidden=32, num_layers=2),
    "torchmdnet": dict(hidden=32, num_layers=2, num_heads=4, num_rbf=8),
    "unimol": dict(hidden=32, num_layers=2, num_heads=4, num_kernels=8),
    "mace": dict(hidden=16, num_layers=2, num_rbf=8),
}

# Restrict toy atomic numbers to this dataset's real element vocabulary
# (see REVISION NOTE above) -- required by MACE, and meaningful for every
# other architecture too (no model in this benchmark needs to handle
# elements that never actually occur in the data).
_PRESENT_Z = torch.tensor(sorted(ELEMENT_TO_Z[e] for e in ELEMENTS_PRESENT))


def _toy_molecule(n: int, seed: int) -> Data:
    """A small molecule with a plausible (non-degenerate) 3D geometry —
    a slightly perturbed zig-zag chain, so bond angles/torsions are
    well-defined (not collinear)."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.zeros(n, 3)
    for k in range(1, n):
        # alternate zig-zag in the xy-plane, so consecutive bond angles are
        # well-defined (not collinear)
        angle = 0.6 if k % 2 == 0 else -0.6
        step = torch.tensor([1.5 * torch.cos(torch.tensor(angle)),
                             1.5 * torch.sin(torch.tensor(angle)), 0.0])
        pos[k] = pos[k - 1] + step
    pos += 0.05 * torch.randn(n, 3, generator=g)

    z_idx = torch.randint(0, len(_PRESENT_Z), (n,), generator=g)
    z = _PRESENT_Z[z_idx]
    x = torch.rand(n, N_NODE_FEAT_3D, generator=g)

    from src.data.geometry_3d import radius_graph_single, build_triplets, build_torsions
    edge_index, edge_weight = radius_graph_single(pos, cutoff=5.0, max_num_neighbors=16)
    trip = build_triplets(edge_index, n)
    tors = build_torsions(edge_index, trip, n)

    from src.data.dataset_3d import Data3D
    return Data3D(
        x=x, z=z, pos=pos, y=torch.rand(n, 51, generator=g),
        edge_index=edge_index, edge_weight=edge_weight,
        tri_idx_i=trip["idx_i"], tri_idx_j=trip["idx_j"], tri_idx_k=trip["idx_k"],
        tri_idx_kj=trip["idx_kj"], tri_idx_ji=trip["idx_ji"],
        tor_idx_l=tors["idx_l"], tor_has_l=tors["has_l"],
        num_nodes=n,
    )


def _toy_batch() -> Batch:
    return Batch.from_data_list([_toy_molecule(5, seed=0), _toy_molecule(4, seed=1)])


def test_all_3d_models_forward_finite_and_correct_shape():
    batch = _toy_batch()
    n_atoms_total = batch.x.size(0)

    for name, cls in MODEL_REGISTRY_3D.items():
        model = cls(**_MODEL_KWARGS[name])
        model.eval()
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (n_atoms_total, 51), (
            f"{name}: expected output shape ({n_atoms_total}, 51), got {tuple(out.shape)}"
        )
        assert torch.isfinite(out).all(), f"{name}: output contains NaN/Inf"
        assert (out >= 0).all(), f"{name}: output should be non-negative (models end with softplus)"


def test_all_3d_models_are_batch_size_invariant():
    """Running the same molecule alone vs. batched with another molecule
    should give the same per-atom prediction — this is the exact scenario
    `Data3D.__inc__` (src/data/dataset_3d.py) exists to make correct for
    DimeNet/DimeNet++/SphereNet's triplet/torsion indices, and for Uni-Mol's
    dense-padding mask."""
    paired = _toy_batch()
    graphs = paired.to_data_list()
    single = Batch.from_data_list([graphs[0]])
    n0 = single.x.size(0)

    for name, cls in MODEL_REGISTRY_3D.items():
        model = cls(**_MODEL_KWARGS[name])
        model.eval()
        with torch.no_grad():
            out_single = model(single)
            out_paired = model(paired)
        assert torch.allclose(out_single, out_paired[:n0], atol=1e-4), (
            f"{name}: prediction for molecule 0 changed depending on which "
            f"other molecule shares its batch — cross-molecule information "
            f"leakage (see Data3D.__inc__ / to_dense_batch masking)."
        )
