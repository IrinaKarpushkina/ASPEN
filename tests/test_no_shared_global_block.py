"""
test_no_shared_global_block.py

Regression test for the bug described in PROVENANCE.md: the previous
version of this benchmark gave GCN, GAT, GATv2, GINE, AttentiveFP and
DMPNN an IDENTICAL, hand-written
    global_mean_pool + global_max_pool -> Linear-SiLU-Linear -> LayerNorm
block, which acted as a confound that compressed the measured differences
between architectures.

This test does not try to detect "any pooling" (GPS's built-in attention
and AttentiveFP's own super-node readout are legitimate, paper-specific
mechanisms and should stay). It specifically guards against the concrete
regression: no model in MODEL_REGISTRY should contain a submodule literally
named `global_proj` alongside a `global_norm` LayerNorm of matching shape
to more than one other model — i.e. don't let a shared bolt-on pooling
block silently creep back in.
"""
import torch.nn as nn

from src.models import MODEL_REGISTRY_2D as MODEL_REGISTRY


def _has_named_submodule(model: nn.Module, name: str) -> bool:
    return any(n == name for n, _ in model.named_modules())


def test_no_model_has_the_old_shared_global_block():
    """None of the 7 architectures should define a `global_proj` /
    `global_norm` pair — that was the shared, un-cited pooling add-on
    removed in this rewrite. Each model's own global-context mechanism
    (if any) has a different, architecture-specific name instead."""
    offenders = []
    for name, cls in MODEL_REGISTRY.items():
        model = _instantiate_small(cls, name)
        if _has_named_submodule(model, "global_proj") or _has_named_submodule(model, "global_norm"):
            offenders.append(name)

    assert not offenders, (
        f"These models still define a 'global_proj'/'global_norm' submodule, "
        f"which was the shared pooling confound removed in this rewrite: "
        f"{offenders}. See PROVENANCE.md."
    )


def test_architectures_do_not_share_identical_readout_submodule():
    """Sanity check: the models should not be literally structurally
    identical after the input projection (which would defeat the point of
    comparing them at all)."""
    signatures = {}
    for name, cls in MODEL_REGISTRY.items():
        model = _instantiate_small(cls, name)
        sig = tuple(type(m).__name__ for m in model.modules())
        signatures[name] = sig

    # No two distinct architectures should produce the exact same sequence
    # of module types (that would mean they're the same model under a
    # different name).
    seen = {}
    for name, sig in signatures.items():
        if sig in seen:
            raise AssertionError(
                f"{name} and {seen[sig]} have an identical module structure "
                f"— they are not actually different architectures."
            )
        seen[sig] = name


def _instantiate_small(cls, name):
    kwargs = dict(hidden=64)
    if name in ("gat", "gatv2"):
        kwargs["heads"] = 4
    if name == "gps":
        kwargs["heads"] = 4
    if name == "dmpnn":
        kwargs["num_steps"] = 2
    else:
        kwargs.setdefault("num_layers", 2)
    if name == "attentive_fp":
        kwargs.pop("num_layers", None)
        kwargs["num_layers"] = 2
        kwargs["num_timesteps"] = 2
    return cls(**kwargs)
