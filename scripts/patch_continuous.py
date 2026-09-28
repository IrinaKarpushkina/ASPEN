"""Registers sigma_continuous_dimenet. Apply AFTER patch_train_3d_physics.py
(shares the "sigma_" prefix dispatch anchor). Order relative to
patch_kitchen_sink.py does not matter (different anchor).

    python scripts/patch_continuous.py --check
    python scripts/patch_continuous.py
"""
import argparse
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src" / "train_3d_physics.py"

COMPUTE_TRIPLETS_VARIANTS = [
    ("        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")),\n",
     "        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or name == \"sigma_continuous_dimenet\"),\n"),
    # variant after patch_kitchen_sink.py has already run (order-independent):
    ("        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or (name == \"sigma_kitchen_sink\" and cfg[\"model\"].get(\"backbone\", \"dimenet\") == \"dimenet\")),\n",
     "        compute_triplets=(name in _MODELS_NEEDING_TRIPLETS or name.startswith(\"sigma_dn\")\n"
     "                          or (name == \"sigma_kitchen_sink\" and cfg[\"model\"].get(\"backbone\", \"dimenet\") == \"dimenet\")\n"
     "                          or name == \"sigma_continuous_dimenet\"),\n"),
]

EDITS_FIXED = [
    ("from .physics_extension.sigma_global import DimeNetPPSigmaGlobal\n",
     "from .physics_extension.sigma_global import DimeNetPPSigmaGlobal\n"
     "from .physics_extension.sigma_continuous import SigmaContinuousDimeNet\n"),
    ("    if name.startswith(\"sigma_\"):\n        return DimeNetPPSigmaGlobal(**mc), name\n",
     "    if name == \"sigma_continuous_dimenet\":\n"
     "        return SigmaContinuousDimeNet(**mc), name\n"
     "    if name.startswith(\"sigma_\"):\n        return DimeNetPPSigmaGlobal(**mc), name\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--file", default=str(TARGET))
    a = ap.parse_args()
    path = Path(a.file)
    src = path.read_text()
    if "SigmaContinuousDimeNet" in src:
        print("already patched")
        return
    if "DimeNetPPSigmaGlobal" not in src:
        raise SystemExit("Apply scripts/patch_train_3d_physics.py first.")

    # compute_triplets line has two possible current forms depending on whether
    # patch_kitchen_sink.py already ran; pick whichever one matches.
    variant = None
    for old, new in COMPUTE_TRIPLETS_VARIANTS:
        if src.count(old) == 1:
            variant = (old, new)
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
    shutil.copy(path, str(path) + ".bak_pre_continuous")
    for o, n in edits:
        src = src.replace(o, n)
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
