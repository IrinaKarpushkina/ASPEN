"""
tune_expensive_models.py - narrow hidden-size search for spherenet/mace
ONLY, avoiding scripts.count_params's --auto-tune full sweep (32..1024,
step 1 => ~1000 model instantiations), which is extremely slow for these
two architectures specifically: SphereNet's angle_emb/torsion_emb do
expensive sympy symbolic derivation (real spherical harmonics / Bessel
basis) at construction time, and MACE's EquivariantProductBasisBlock
(correlation=3) builds symmetric-contraction coefficient tables --
BOTH are independent of `hidden` but get needlessly rebuilt on every
single sweep step, turning a >24h job into what should be a
few-minutes search.

This does a coarse-to-fine search with at most ~10-15 model
instantiations total per architecture (not ~1000), printing progress as
it goes so you can see it's actually running (unlike a bare pytest
collection with no output for hours).

Usage:
    python tune_expensive_models.py --model spherenet
    python tune_expensive_models.py --model mace
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from src.config import load_config
from scripts.count_params import count_params


def coarse_to_fine_search(model_cfg: dict, target: int, width_key: str,
                           coarse_candidates: list, refine_radius: int = 8):
    results = {}

    def try_one(h):
        if h in results:
            return results[h]
        cfg = dict(model_cfg)
        cfg[width_key] = h
        t0 = time.time()
        n = count_params(cfg)
        dt = time.time() - t0
        results[h] = n
        print(f"  hidden={h:4d}  params={n:>9,}  diff={((n-target)/target*100):+6.2f}%  ({dt:.1f}s)")
        return n

    print(f"Coarse pass ({len(coarse_candidates)} points):")
    for h in coarse_candidates:
        try_one(h)

    best_h = min(results, key=lambda h: abs(results[h] - target))
    print(f"\nBest so far: hidden={best_h} ({results[best_h]:,} params)")

    print(f"\nRefining around {best_h} (+/-{refine_radius}):")
    for h in range(max(8, best_h - refine_radius), best_h + refine_radius + 1, 2):
        try_one(h)

    best_h = min(results, key=lambda h: abs(results[h] - target))
    print(f"\nFINAL: hidden={best_h}, params={results[best_h]:,}, "
          f"diff={((results[best_h]-target)/target*100):+.2f}%")
    return best_h, results[best_h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["spherenet", "mace"])
    parser.add_argument("--configs-dir", default="configs/3d")
    args = parser.parse_args()

    base_cfg = load_config(f"{args.configs_dir}/base.yaml")
    target = base_cfg["target_params"]

    cfg = load_config(f"{args.configs_dir}/{args.model}.yaml")
    model_cfg = cfg["model"]
    width_key = "hidden_channels" if "hidden_channels" in model_cfg else "hidden"

    print(f"Target: {target:,} params. Current config hidden={model_cfg.get(width_key)}")
    print(f"(Each line below instantiates the model ONCE -- if a single "
          f"line takes a long time, that IS the expected sympy/CG-table "
          f"construction cost for {args.model}, not a hang.)\n")

    # Coarse grid -- adjust the range if your current config's hidden is
    # far outside this, based on the Step-1 (no --auto-tune) count you
    # already have.
    coarse = sorted(set([32, 48, 64, 80, 96, 112, 128, 160, 200,
                        model_cfg.get(width_key, 128)]))
    coarse_to_fine_search(model_cfg, target, width_key, coarse)


if __name__ == "__main__":
    main()
