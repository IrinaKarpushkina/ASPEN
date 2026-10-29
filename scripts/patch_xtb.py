"""Патчит src/train_3d_physics.py для xTB-экспериментов (идемпотентно, делает бэкап .bak_pre_xtb).

    python scripts/patch_xtb.py            # применить
    python scripts/patch_xtb.py --check    # только проверить, применимы ли замены

Что добавляется: features "xtb"/"xtb_sasa", модели dimenet_pp_{enhanced,delta}_xtb, dimenet_pp_delta_l1,
sigma_dn_geom_xtb(_big) (уже работает через sigma_dn*), sigma_surface_* (SigmaSurfaceField).
"""
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src" / "train_3d_physics.py"

EDITS = [
    # 1. imports
    ("from .data.dataset_3d_kitchen import KitchenChaosParquet3DDataset\n",
     "from .data.dataset_3d_kitchen import KitchenChaosParquet3DDataset\n"
     "from .data.dataset_3d_xtb import XtbChaosParquet3DDataset, XtbSasaChaosParquet3DDataset\n"
     "from .physics_extension.surface_field import SigmaSurfaceField\n"),
    # 2. триплеты для новых dimenet-моделей
    ('"dimenet_pp_global", "dimenet_pp_delta"}',
     '"dimenet_pp_global", "dimenet_pp_delta",\n'
     '                             "dimenet_pp_enhanced_xtb", "dimenet_pp_delta_xtb", "dimenet_pp_delta_l1"}'),
    # 3. реестр моделей
    ('        "dimenet_pp_delta": DimeNetPPDeltaPhysics,\n',
     '        "dimenet_pp_delta": DimeNetPPDeltaPhysics,\n'
     '        "dimenet_pp_enhanced_xtb": DimeNetPPEnhanced,\n'
     '        "dimenet_pp_delta_xtb": DimeNetPPDeltaPhysics,\n'
     '        "dimenet_pp_delta_l1": DimeNetPPDeltaPhysics,\n'),
    # 4. surface-модель (до общего sigma_*)
    ('    if name == "sigma_kitchen_sink":\n',
     '    if name.startswith("sigma_surface"):\n        return SigmaSurfaceField(**mc), name\n'
     '    if name == "sigma_kitchen_sink":\n'),
    # 5. допустимые features
    ('if features not in {"base", "enhanced", "sasa", "kitchen"}:',
     'if features not in {"base", "enhanced", "sasa", "kitchen", "xtb", "xtb_sasa"}:'),
    ("\"sasa\": SasaChaosParquet3DDataset, \"kitchen\": KitchenChaosParquet3DDataset}[features]",
     "\"sasa\": SasaChaosParquet3DDataset, \"kitchen\": KitchenChaosParquet3DDataset,\n"
     "              \"xtb\": XtbChaosParquet3DDataset, \"xtb_sasa\": XtbSasaChaosParquet3DDataset}[features]"),
    # 6. триплеты для surface
    ('or name == "sigma_continuous_dimenet"),',
     'or name == "sigma_continuous_dimenet" or name.startswith("sigma_surface")),'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--file", default=str(TARGET))
    a = ap.parse_args()
    path = Path(a.file)
    src = path.read_text()
    if "dataset_3d_xtb" in src:
        print("уже применено, ничего не делаю")
        return
    new, bad = src, []
    for i, (old, rep) in enumerate(EDITS, 1):
        if new.count(old) != 1:
            bad.append((i, old.strip()[:70], new.count(old)))
            continue
        new = new.replace(old, rep)
    if bad:
        print("НЕ ПРИМЕНИМО автоматически (ваш train_3d_physics.py отличается). Правки вручную по README_XTB.md §4:")
        for i, o, c in bad:
            print(f"  правка #{i}: найдено {c} вхождений: {o!r}")
        sys.exit(1)
    if a.check:
        print("все 6 замен применимы")
        return
    shutil.copy(path, str(path) + ".bak_pre_xtb")
    path.write_text(new)
    print(f"OK: {path} обновлён, бэкап {path}.bak_pre_xtb")


if __name__ == "__main__":
    main()
