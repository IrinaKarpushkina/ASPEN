"""
test_param_budget.py — controlled-comparison requirement (Dwivedi et al.,
JMLR 2022): every architecture's config must be within +/-5% of
configs/base.yaml's target_params. Run this after any architecture change
(e.g. adding/removing a module) to catch parameter-budget drift before
spending GPU time on a training run.
"""
import glob
import os

from src.config import load_config
from scripts.count_params import count_params

TOLERANCE_PCT = 5.0
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "2d")


def test_all_2d_configs_within_parameter_budget():
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
        n_params = count_params(cfg["model"])
        diff_pct = abs(n_params - target) / target * 100
        if diff_pct > TOLERANCE_PCT:
            failures.append(f"{cfg['model']['name']}: {n_params:,} params ({diff_pct:+.2f}%)")

    assert not failures, (
        f"These configs are outside the +/-{TOLERANCE_PCT}% parameter budget "
        f"(target={target:,}): {failures}. Re-run "
        f"`python -m scripts.count_params --auto-tune` and update hidden= "
        f"in the corresponding configs/*.yaml."
    )
