"""Preflight checks for the additive physics extension.

Run from the repository root. This script never edits benchmark files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make the repository root importable when this file is executed directly
# as ``python scripts/check_physics_setup.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    repo = REPO_ROOT
    forbidden = [
        repo / "src/models/models_3d/dimenet.py",
        repo / "src/models/models_3d/painn.py",
    ]
    print("Repository:", repo)
    print("Baseline files are NOT modified by this extension.")
    print("Existing DimeNet:", forbidden[0], "exists=", forbidden[0].exists())
    print("Existing PaiNN:", forbidden[1], "exists=", forbidden[1].exists())

    try:
        from src.data.dataset_3d import ChaosParquet3DDataset
        from src.data.dataset_3d_enhanced import EnhancedChaosParquet3DDataset
        from src.physics_extension import DimeNetPPPhysics
        from src.physics_extension.physics import PhysicsSigmaHead
    except Exception as exc:
        raise SystemExit(f"Import check failed: {type(exc).__name__}: {exc}")

    ds = ChaosParquet3DDataset(
        args.train,
        cache_dir=args.cache_dir,
        compute_triplets=True,
        require_provided_coords=True,
    )
    d = ds[0]
    if d.x.shape[1] != 6:
        raise SystemExit(f"Expected base x dim 6, got {d.x.shape[1]}")

    eds = EnhancedChaosParquet3DDataset(
        args.train,
        cache_dir=args.cache_dir,
        compute_triplets=True,
        require_provided_coords=True,
    )
    e = eds[0]
    if e.x.shape[1] != 22:
        raise SystemExit(f"Expected enhanced x dim 22, got {e.x.shape[1]}")
    if e.y.shape[-1] != 51:
        raise SystemExit(f"Expected target dim 51, got {e.y.shape[-1]}")

    print("Base features:", tuple(d.x.shape))
    print("Enhanced features:", tuple(e.x.shape))
    print("Target:", tuple(e.y.shape))
    print("Additive physics setup OK.")


if __name__ == "__main__":
    main()
