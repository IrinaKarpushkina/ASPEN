"""Generate loss-weight variants of one architecture and print the run commands.

    python -m scripts.make_loss_sweep --base-config configs/3d/dimenet_pp_enhanced_physics.yaml

Screen every variant with ONE seed, then repeat the best two with seeds 0 1 2.
All variants use the new schedule (40-epoch cycles) so that the schedule artefact does not
mask loss effects. Requires scripts/patch_delta_and_loss_v2.py to have been applied.
Read the results with:  python -m scripts.aggregate_results --metrics-dir results/loss_sweep/<tag>/metrics --loss physics_v2
"""
import argparse
from pathlib import Path

import yaml

BASE = dict(w_profile=1.0, w_area=0.25, w_charge=0.25, w_moments=0.10, w_smooth=0.05,
            w_wasserstein=0.25, w_polar=0.50, profile_loss="mse", moments_mode="raw",
            w_cos=0.0, w_mol=0.0, w_mol_cdf=0.0)

VARIANTS = {
    # control: identical to the current 'physics' loss (sanity check that V2 == V1)
    "L0_control": {},
    # metric alignment: wMAE is L1, molecular_* metrics are on summed profiles
    "L1_l1": dict(profile_loss="l1"),
    "L2_l1_mol": dict(profile_loss="l1", w_mol=0.5, w_mol_cdf=0.25),
    "L3_l1_mol_cos": dict(profile_loss="l1", w_mol=0.5, w_mol_cdf=0.25, w_cos=0.25),
    # make moment / smoothness terms actually matter
    "L4_relmom": dict(profile_loss="l1", moments_mode="relative", w_moments=0.3, w_smooth=0.1,
                      w_mol=0.5, w_mol_cdf=0.25),
    # lighter physics (favours wMAE/R2) and heavier physics (favours distribution metrics)
    "L5_light": dict(w_area=0.05, w_charge=0.05, w_moments=0.02, w_smooth=0.01,
                     w_wasserstein=0.10, w_polar=0.10),
    "L6_heavy_dist": dict(profile_loss="l1", w_wasserstein=0.5, w_polar=1.0, w_mol=0.5,
                          w_mol_cdf=0.5, w_cos=0.25),
}
SCHED = dict(max_epochs=200, patience=45, scheduler_T0=40, scheduler_Tmult=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bp = Path(a.base_config)
    base = yaml.safe_load(open(bp))
    for tag, over in VARIANTS.items():
        cfg = dict(base)
        cfg["training"] = {**base.get("training", {}), **SCHED}
        cfg["loss"] = {"name": "physics_v2", **BASE, **over}
        out = bp.parent / f"ls_{tag}_{bp.stem}.yaml"
        yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
        print(f"python -m src.train_3d_physics --config {out} --seed {a.seed} --output-dir results/loss_sweep/{tag}")
