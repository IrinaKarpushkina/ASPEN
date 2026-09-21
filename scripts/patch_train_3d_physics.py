"""One-shot, idempotent patch for src/train_3d_physics.py.

    python scripts/patch_train_3d_physics.py            # patches src/train_3d_physics.py
    python scripts/patch_train_3d_physics.py --check    # only report

A backup src/train_3d_physics.py.bak_pre_sigma is written first. It:
  1. registers every model whose name starts with ``sigma_`` (sigma_dn_* use
     triplets, sigma_pn_* do not);
  2. adds ``features: sasa`` (SasaChaosParquet3DDataset);
  3. calls model.calibrate(train_ds) before the optimizer is built;
  4. replaces the hard crash on a non-finite loss by "skip batch + log" (crashes
     only after 50 skipped batches) - this is what killed dimenet_pp_delta.
"""
import argparse
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src" / "train_3d_physics.py"

EDITS = [
    ("from .physics_extension.physics import PhysicsSigmaLoss\n",
     "from .physics_extension.physics import PhysicsSigmaLoss\n"
     "from .physics_extension.sigma_global import DimeNetPPSigmaGlobal\n"
     "from .data.dataset_3d_sasa import SasaChaosParquet3DDataset\n"),
    ("    if name not in registry:\n",
     "    if name.startswith(\"sigma_\"):\n"
     "        return DimeNetPPSigmaGlobal(**mc), name\n"
     "    if name not in registry:\n"),
    ('    if features not in {"base", "enhanced"}:\n'
     '        raise ValueError(f"features must be \'base\' or \'enhanced\', got {features}")\n',
     '    if features not in {"base", "enhanced", "sasa"}:\n'
     '        raise ValueError(f"features must be \'base\', \'enhanced\' or \'sasa\', got {features}")\n'),
    ('    ds_cls = EnhancedChaosParquet3DDataset if features == "enhanced" else ChaosParquet3DDataset\n',
     '    ds_cls = {"base": ChaosParquet3DDataset, "enhanced": EnhancedChaosParquet3DDataset,\n'
     '              "sasa": SasaChaosParquet3DDataset}[features]\n'),
    ("        compute_triplets=name in _MODELS_NEEDING_TRIPLETS,\n",
     "        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")),\n"),
    ("    model = model.to(device)\n",
     "    if hasattr(model, \"calibrate\"):\n"
     "        model.calibrate(train_ds)\n"
     "    model = model.to(device)\n"),
    ("    history = []\n",
     "    history = []\n    n_skipped = 0\n"),
    ('            if not torch.isfinite(loss):\n'
     '                raise FloatingPointError(f"Non-finite training loss at epoch={epoch+1}, step={step+1}")\n',
     '            if not torch.isfinite(loss):\n'
     '                n_skipped += 1\n'
     '                logger.warning("Non-finite loss (epoch=%d step=%d, skipped #%d); y.max=%.4g",\n'
     '                               epoch + 1, step + 1, n_skipped, float(data.y.max()))\n'
     '                optimizer.zero_grad(set_to_none=True)\n'
     '                if n_skipped > 50:\n'
     '                    raise FloatingPointError("More than 50 non-finite batches; aborting")\n'
     '                continue\n'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--file", default=str(TARGET))
    args = ap.parse_args()
    path = Path(args.file)
    src = path.read_text()
    if "DimeNetPPSigmaGlobal" in src:
        print("already patched")
        return
    missing = [old[:70] for old, _ in EDITS if src.count(old) != 1]
    if missing:
        raise SystemExit("Cannot patch, anchors not found exactly once:\n  " + "\n  ".join(missing))
    if args.check:
        print("all anchors found; patch would apply cleanly")
        return
    shutil.copy(path, str(path) + ".bak_pre_sigma")
    for old, new in EDITS:
        src = src.replace(old, new)
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
