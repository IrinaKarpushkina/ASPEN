"""Learning-curve test: is the model data-limited, capacity-limited or noise-limited?

    python -m scripts.make_learning_curve \
        --train /mnt/tank/.../chaos_atomic_train_with_coordinates.parquet \
        --base-config configs/3d/dimenet_pp_enhanced_sched.yaml \
        --fractions 0.125 0.25 0.5

Writes train subsets (split by MOLECULE, val/test untouched) next to the original parquet and one
yaml per fraction; prints the commands to run. Afterwards:

    python -m scripts.make_learning_curve --summarize results/lc_f0.125 results/lc_f0.25 results/lc_f0.5 results/physics_v2

Reading the curve (weighted_mae vs number of training molecules, log-log):
  * error keeps falling ~ n^-0.1 or steeper at 100%  -> more data / better inductive bias still helps
  * error flat between 50% and 100%                  -> saturated: data/label noise ceiling
    (architectures cannot fix it; ensembles/descriptors that add NEW information can)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def make(args):
    df = pd.read_parquet(args.train)
    ids = df["mol_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    base = yaml.safe_load(open(args.base_config))
    stem = Path(args.train).stem
    cfg_dir = Path(args.base_config).parent
    for f in args.fractions:
        keep = set(ids[: int(round(len(ids) * f))].tolist())
        out = str(Path(args.train).with_name(f"{stem}_frac{f}.parquet"))
        df[df["mol_id"].isin(keep)].to_parquet(out)
        cfg = dict(base)
        cfg.setdefault("data", {})["train_path"] = out
        cpath = cfg_dir / f"lc_f{f}_{Path(args.base_config).stem}.yaml"
        yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
        print(f"frac {f}: {len(keep)} molecules -> {out}\n   python -m src.train_3d_physics --config {cpath} "
              f"--seed 0 --output-dir results/lc_f{f}")


def summarize(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "metrics", "*.json")):
            r = json.load(open(f))
            rows.append((d, r["model"], r["seed"], r["test_metrics"]["weighted_mae"], r["test_metrics"]["weighted_r2"]))
    df = pd.DataFrame(rows, columns=["dir", "model", "seed", "wMAE", "R2"])
    print(df.groupby(["dir", "model"]).agg(wMAE=("wMAE", "mean"), R2=("R2", "mean"), n=("seed", "count")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--base-config")
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.125, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", nargs="*")
    a = ap.parse_args()
    summarize(a.summarize) if a.summarize else make(a)
