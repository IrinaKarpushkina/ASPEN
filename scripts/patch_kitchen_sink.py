"""Registers sigma_kitchen_sink and features: kitchen. Apply AFTER
patch_train_3d_physics.py (needs its "sigma_" prefix anchor already in place;
--check will tell you if the order is wrong). Idempotent, makes a backup.

    python scripts/patch_kitchen_sink.py --check
    python scripts/patch_kitchen_sink.py
"""
import argparse
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src" / "train_3d_physics.py"

COMPUTE_TRIPLETS_VARIANTS = [
    ("        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")),\n",
     "        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or (name == \"sigma_kitchen_sink\" and cfg[\"model\"].get(\"backbone\", \"dimenet\") == \"dimenet\")),\n"),
    # variant after patch_continuous.py has already run (order-independent):
    ("        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or name == \"sigma_continuous_dimenet\"),\n",
     "        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or (name == \"sigma_kitchen_sink\" and cfg[\"model\"].get(\"backbone\", \"dimenet\") == \"dimenet\")\n"
     "                          or name == \"sigma_continuous_dimenet\"),\n"),
]

EDITS_FIXED = [
    ("from .physics_extension.sigma_global import DimeNetPPSigmaGlobal\n",
     "from .physics_extension.sigma_global import DimeNetPPSigmaGlobal\n"
     "from .physics_extension.sigma_kitchen_sink import SigmaKitchenSink\n"),
    ("    if name.startswith(\"sigma_\"):\n        return DimeNetPPSigmaGlobal(**mc), name\n",
     "    if name == \"sigma_kitchen_sink\":\n"
     "        return SigmaKitchenSink(**mc), name\n"
     "    if name.startswith(\"sigma_\"):\n        return DimeNetPPSigmaGlobal(**mc), name\n"),
    ('    if features not in {"base", "enhanced", "sasa"}:\n'
     '        raise ValueError(f"features must be \'base\', \'enhanced\' or \'sasa\', got {features}")\n',
     '    if features not in {"base", "enhanced", "sasa", "kitchen"}:\n'
     '        raise ValueError(f"features must be \'base\', \'enhanced\', \'sasa\' or \'kitchen\', got {features}")\n'),
    ('    ds_cls = {"base": ChaosParquet3DDataset, "enhanced": EnhancedChaosParquet3DDataset,\n'
     '              "sasa": SasaChaosParquet3DDataset}[features]\n',
     '    ds_cls = {"base": ChaosParquet3DDataset, "enhanced": EnhancedChaosParquet3DDataset,\n'
     '              "sasa": SasaChaosParquet3DDataset, "kitchen": KitchenChaosParquet3DDataset}[features]\n'),
    ("from .data.dataset_3d_sasa import SasaChaosParquet3DDataset\n",
     "from .data.dataset_3d_sasa import SasaChaosParquet3DDataset\n"
     "from .data.dataset_3d_kitchen import KitchenChaosParquet3DDataset\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--file", default=str(TARGET))
    a = ap.parse_args()
    path = Path(a.file)
    src = path.read_text()
    if "SigmaKitchenSink" in src:
        print("already patched")
        return
    if "DimeNetPPSigmaGlobal" not in src:
        raise SystemExit("Apply scripts/patch_train_3d_physics.py first (its anchors are missing here).")

    variant = None
    for old_v, new_v in COMPUTE_TRIPLETS_VARIANTS:
        if src.count(old_v) == 1:
            variant = (old_v, new_v)
            break
    if variant is None:
        raise SystemExit("compute_triplets anchor not found in either expected form; "
                         "check the current text of that line in train_3d_physics.py manually.")
    edits = EDITS_FIXED + [variant]

    missing = [o[:60] for o, _ in edits if src.count(o) != 1]
    if missing:
        raise SystemExit("Anchors not found exactly once:\n  " + "\n  ".join(missing))
    if a.check:
        print("all anchors found; patch would apply cleanly")
        return
    shutil.copy(path, str(path) + ".bak_pre_kitchen")
    for o, n in edits:
        src = src.replace(o, n)
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
