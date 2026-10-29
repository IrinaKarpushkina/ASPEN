"""Пакетный запуск molecule_level_eval по всем metrics-json: config: config_path из json, иначе configs/3d/<prefix>.yaml или configs/3d/scaling/<prefix>.yaml, чекпоинт - из соседней
папки checkpoints/<имя json>.pt. Уже посчитанные .npz пропускаются.

  python -m scripts.molecule_eval_batch --dirs results/all_pred_ablation_experiments_results results/xtb \
      --out-dir results/per_mol_all --only 'dimenet_pp_enhanced*' sigma_dn_geom --seeds 0 1 2
--dirs: папки, внутри которых лежат metrics/ и checkpoints/. Если config_path из json не найден на диске,
ищется configs/3d/<имя>.yaml и configs/3d/scaling/<имя>.yaml.
"""
import argparse
import fnmatch
import glob
import json
import os

from scripts._runs import parse_stem
from scripts.molecule_level_eval import run


# Прогоны dimenet_pp_delta обучены по ..._fixed.yaml (stable=true, prior_floor=1e-3). У обычного dimenet_pp_delta.yaml
# те же имена параметров, поэтому чекпоинт грузится молча, но модель считает иначе -> сверка с json не сходится.
DEFAULT_CFG = {"dimenet_pp_delta": "configs/3d/dimenet_pp_delta_fixed.yaml"}


def find_cfg(cp, prefix, override=None):
    if override and prefix in override and os.path.exists(override[prefix]):
        return override[prefix]
    """config_path из json (есть только у train_3d.py) -> configs/3d/<prefix>.yaml -> configs/3d/scaling/<prefix>.yaml"""
    for c in (cp, f"configs/3d/{prefix}.yaml", f"configs/3d/scaling/{prefix}.yaml"):
        if c and os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", nargs="*", default=["*"], help="шаблоны по prefix (имя до _seed)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--mode", default="3d")
    ap.add_argument("--cfg-map", nargs="*", default=[], help="PREFIX=путь_к_конфигу, переопределяет подбор конфига")
    a = ap.parse_args()
    override = dict(DEFAULT_CFG)
    override.update(kv.split("=", 1) for kv in a.cfg_map)
    for d in a.dirs:
        for mj in sorted(glob.glob(os.path.join(d, "metrics", "*.json"))):
            stem = os.path.splitext(os.path.basename(mj))[0]
            prefix, _, seed, _ = parse_stem(stem)
            if seed is None or (a.seeds and seed not in a.seeds) or not any(fnmatch.fnmatchcase(prefix, p) for p in a.only):
                continue
            out = os.path.join(a.out_dir, f"{prefix}_seed{seed}.npz")
            if os.path.exists(out):
                print(f"[skip] {out}"); continue
            r = json.load(open(mj))
            cfg = find_cfg(r.get("config_path"), prefix, override)
            ck = os.path.join(d, "checkpoints", stem + ".pt")
            if not cfg or not os.path.exists(ck) or len(r.get("history", [])) < 30:
                print(f"[skip] {stem}: config={cfg} ckpt={'ok' if os.path.exists(ck) else 'нет'} эпох={len(r.get('history', []))}"); continue
            print(f"=== {stem}")
            run(cfg, ck, a.mode, "test", out, check_json=mj)


if __name__ == "__main__":
    main()
