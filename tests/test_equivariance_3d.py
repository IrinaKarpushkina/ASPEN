"""
test_equivariance_3d.py — every 3D architecture in this benchmark predicts
a per-atom SCALAR sigma-profile, which must be INVARIANT to rotating
and/or translating the input molecule's 3D coordinates — a molecule's
physical/electronic properties do not depend on which way you happened to
orient it in space when you wrote down its coordinates.

This is the central scientific-correctness requirement of the whole 3D
benchmark: a bug that leaks absolute orientation/position into a model's
prediction would make every number in the resulting comparison table
meaningless, and would NOT necessarily show up as a shape or NaN error
(the kind `test_forward_shapes_3d.py` catches) — the model would just
silently learn a spurious, non-physical shortcut. Golden-test-style
regression check, in the same spirit as
`tests/test_gcn_conv_formula.py`/`tests/test_dmpnn_reverse_edge.py`
(2D) — this is exactly the kind of property a unit test, not a training
curve, is suited to catch.

NOTE: models are set to `.eval()` (dropout off) and compared under
`torch.no_grad()` — comparing two forward passes of the SAME stochastic
model in train() mode would (correctly) differ even for a perfectly
invariant architecture, and is not what this test is checking.
"""
import copy

import torch
from torch_geometric.data import Batch

from src.models import MODEL_REGISTRY_3D
from tests.test_forward_shapes_3d import _MODEL_KWARGS, _toy_batch

ATOL = 1e-3  # float32 rounding tolerance across a handful of layers


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1  # ensure a proper rotation (det=+1), not a reflection
    return Q


def test_all_3d_models_are_rotation_and_translation_invariant():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    batch = _toy_batch()
    Q = _random_rotation(gen)
    t = torch.tensor([3.7, -1.2, 2.4])

    rotated = copy.deepcopy(batch)
    rotated.pos = rotated.pos @ Q.T

    translated = copy.deepcopy(batch)
    translated.pos = translated.pos + t

    rotated_and_translated = copy.deepcopy(batch)
    rotated_and_translated.pos = rotated_and_translated.pos @ Q.T + t

    failures = []
    for name, cls in MODEL_REGISTRY_3D.items():
        torch.manual_seed(42)  # identical init across the 4 forward passes below
        model = cls(**_MODEL_KWARGS[name])
        model.eval()

        with torch.no_grad():
            out_ref = model(batch)
            out_rot = model(rotated)
            out_trans = model(translated)
            out_both = model(rotated_and_translated)

        for label, out in [("rotation", out_rot), ("translation", out_trans),
                          ("rotation+translation", out_both)]:
            diff = (out_ref - out).abs().max().item()
            if diff > ATOL:
                failures.append(f"{name} [{label}]: max abs diff {diff:.2e} > {ATOL:.0e}")

    assert not failures, (
        "The following 3D models are NOT rotation/translation invariant "
        "(their prediction changed when the SAME molecule was rotated "
        "and/or translated) — this is a correctness bug, not a training "
        "issue:\n  " + "\n  ".join(failures)
    )
