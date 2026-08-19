"""
test_param_budget_3d.py — controlled-comparison requirement (Dwivedi et
al., JMLR 2022): every 3D architecture's config must be within +/-5% of
configs/3d/base.yaml's target_params, the SAME budget used by the 2D
benchmark (configs/2d/base.yaml) — see PROVENANCE.md.

NOTE ON RUNTIME: DimeNet/DimeNet++/SphereNet's `SphericalBasisLayer`
constructor does real sympy symbolic work (building/lambdifying Bessel
and spherical-harmonic basis functions) that takes several seconds to
instantiate REGARDLESS of hidden_channels, using the papers' own
defaults (num_spherical=7) in configs/3d/*.yaml — so this specific test
is slow (tens of seconds) compared to every other test in this suite.
That is expected and is not itself a bug.

Mirrors `tests/test_param_budget.py` (2D) exactly in structure.
"""
import glob
import os

from src.config import load_config
from scripts.count_params import count_params

TOLERANCE_PCT = 5.0
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "3d")


def test_all_3d_configs_within_parameter_budget():
    base_cfg = load_config(os.path.join(CONFIGS_DIR, "base.yaml"))
    target = base_cfg["target_params"]

    failures = []
    for path in sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))):
        name = os.path.basename(path)
        if name.startswith("base"):
            continue
        cfg = load_config(path)
        if "model" not in cfg:
            continue
        if cfg["model"]["name"] == "mace":
            try:
                import e3nn  # noqa: F401
            except ImportError:
                continue  # skip: MACE isn't registered without e3nn either (see models_3d/__init__.py)
        n_params = count_params(cfg["model"])
        diff_pct = abs(n_params - target) / target * 100
        if diff_pct > TOLERANCE_PCT:
            failures.append(f"{cfg['model']['name']}: {n_params:,} params ({diff_pct:+.2f}%)")

    assert not failures, (
        f"These configs are outside the +/-{TOLERANCE_PCT}% parameter budget "
        f"(target={target:,}): {failures}. Re-run "
        f"`python -m scripts.count_params --configs-dir configs/3d --auto-tune` "
        f"and update hidden/hidden_channels in the corresponding configs/3d/*.yaml."
    )
