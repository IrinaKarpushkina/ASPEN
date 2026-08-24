"""
test_unimol_pretrained_smoke.py - smoke test for the SECOND, separate
Uni-Mol experiment (src/models/models_3d/unimol_pretrained.py).

Only tests the trainable HEAD (fabricated fake "pretrained" embeddings,
no network access, no real unimol_tools call) -- this is deliberately
fast and network-free, unlike the real pipeline
(dataset_unimol_pretrained.py), which needs unimol_tools installed and a
one-time download from Hugging Face. For an end-to-end check of the real
unimol_tools integration, run `python -m scripts.check_unimol_tools_api`
manually (not part of the automated test suite, since it needs network
access this test environment may not have).
"""
import torch
from torch_geometric.data import Data, Batch

from src.models.models_3d.unimol_pretrained import UniMolPretrainedSigmaModel


def _toy_batch(repr_dim: int = 512):
    def mol(n_atoms, seed):
        g = torch.Generator().manual_seed(seed)
        return Data(
            z=torch.randint(1, 10, (n_atoms,), generator=g),
            y=torch.rand(n_atoms, 51, generator=g),
            unimol_repr=torch.randn(n_atoms, repr_dim, generator=g, dtype=torch.float16),
            num_nodes=n_atoms,
        )
    return Batch.from_data_list([mol(5, 0), mol(4, 1)])


def test_unimol_pretrained_head_forward_shape_and_finite():
    batch = _toy_batch(repr_dim=64)
    model = UniMolPretrainedSigmaModel(repr_dim=64, hidden=32, num_head_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (batch.z.size(0), 51)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()  # softplus output


def test_unimol_pretrained_head_rejects_wrong_repr_dim():
    batch = _toy_batch(repr_dim=64)
    model = UniMolPretrainedSigmaModel(repr_dim=999, hidden=32)  # deliberately wrong
    try:
        model(batch)
        assert False, "expected a ValueError for repr_dim mismatch"
    except ValueError as e:
        assert "repr_dim mismatch" in str(e)
