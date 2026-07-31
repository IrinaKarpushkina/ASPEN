"""
test_forward_shapes.py — smoke test: every registered architecture must
run a forward pass on a tiny toy batch and produce a finite (N_atoms, 51)
output. Catches shape/dtype/NaN bugs before a multi-hour training job does.
"""
import torch
from torch_geometric.data import Data, Batch

from src.models import MODEL_REGISTRY
from src.data.constants import WALK_LENGTH, N_NODE_FEAT_2D, N_EDGE_FEAT_2D

_MODEL_KWARGS = {
    "gcn": dict(hidden=64, num_layers=2),
    "gat": dict(hidden=64, num_layers=2, heads=4),
    "gatv2": dict(hidden=64, num_layers=2, heads=4),
    "gine": dict(hidden=64, num_layers=2),
    "dmpnn": dict(hidden=64, num_steps=2),
    "attentive_fp": dict(hidden=64, num_layers=2, num_timesteps=2),
    "gps": dict(hidden=64, num_layers=2, heads=4),
}


def _toy_batch() -> Batch:
    graphs = []
    # A 3-atom chain and a 2-atom molecule, each with bidirectional bond edges.
    for n, edges in [(3, [[0, 1, 1, 2], [1, 0, 2, 1]]), (2, [[0, 1], [1, 0]])]:
        edge_index = torch.tensor(edges, dtype=torch.long)
        g = Data(
            x=torch.rand(n, N_NODE_FEAT_2D),
            edge_index=edge_index,
            edge_attr=torch.rand(edge_index.size(1), N_EDGE_FEAT_2D),
            z=torch.randint(1, 10, (n,)),
            y=torch.rand(n, 51),
            pe=torch.rand(n, WALK_LENGTH),
            degree=torch.zeros(n),
        )
        graphs.append(g)
    return Batch.from_data_list(graphs)


def test_all_models_forward_finite_and_correct_shape():
    batch = _toy_batch()
    n_atoms_total = batch.x.size(0)

    for name, cls in MODEL_REGISTRY.items():
        model = cls(**_MODEL_KWARGS[name])
        model.eval()
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (n_atoms_total, 51), (
            f"{name}: expected output shape ({n_atoms_total}, 51), got {tuple(out.shape)}"
        )
        assert torch.isfinite(out).all(), f"{name}: output contains NaN/Inf"
        assert (out >= 0).all(), (
            f"{name}: output should be non-negative (models end with softplus)"
        )


def test_all_models_are_batch_size_invariant():
    """Running the same molecule alone vs. batched with another molecule
    should give the same per-atom prediction for models with no
    cross-molecule leakage bug (e.g. a wrongly-scoped global pool)."""
    paired = _toy_batch()
    graphs = paired.to_data_list()
    single = Batch.from_data_list([graphs[0]])
    n0 = single.x.size(0)

    for name, cls in MODEL_REGISTRY.items():
        model = cls(**_MODEL_KWARGS[name])
        model.eval()
        with torch.no_grad():
            out_single = model(single)
            out_paired = model(paired)
        # First n0 atoms belong to the same molecule (identical features) in
        # both cases; a correctly-scoped model must predict the same thing
        # for them regardless of which other molecule shares the batch.
        assert torch.allclose(out_single, out_paired[:n0], atol=1e-4), (
            f"{name}: prediction for molecule 0 changed depending on which "
            f"other molecule shares its batch — this would indicate "
            f"cross-molecule information leakage."
        )
