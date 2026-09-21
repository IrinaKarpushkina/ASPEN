
"""Is ~0.914 weighted-R2 a ceiling? Three data-only diagnostics (no training, no torch).

    python -m scripts.ceiling_analysis \
        --train /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/chaos_atomic_train_with_coordinates.parquet \
        --max-mols 4000 --model-wmae 0.109

A) AREA vs GEOMETRY.  R2 between log(sum_b y_b) and log(SASA) for three probe radii.
   High R2  -> the area of the profile is (almost) geometry -> use_geom_area=True is justified.
B) NEUTRALITY.  |sum_i q_i| / sum_i |q_i| with q_i = sum_b sigma_b*y_ib, compared with random-sign
   shuffles. Actual << shuffled -> screening charge cancels per molecule -> `neutral: true` is a
   legitimate hard constraint. Actual ~ shuffled (or ions present) -> do NOT enable it.
C) NOISE FLOOR (leave-one-out).  Atoms are grouped by Weisfeiler-Lehman hash of the covalent
   graph (radius 0..3). For each atom the 'prediction' is the mean target of the OTHER atoms of the
   group; error is the same variance-weighted MAE as in the benchmark. It is the error of a pure
   lookup-table that sees only the radius-r topology. If it is already ~= your model's 0.109 at
   r=3 -> the target is largely determined by local topology and remaining error is conformer /
   long-range / label noise (near ceiling). If it is much larger than 0.109 the 3D models are
   already using information the lookup cannot, so that number says nothing about a ceiling,
   and you should rely on the learning-curve / ensemble tests instead.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.constants import ELEMENT_TO_Z, SIGMA_BINS, EPS  # noqa: E402
from src.data.features_3d_sasa import PROBES, atom_sasa  # noqa: E402

COV_R = {1: .31, 5: .84, 6: .76, 7: .71, 8: .66, 9: .57, 14: 1.11, 15: 1.07, 16: 1.05,
         17: 1.02, 35: 1.20, 53: 1.39}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def load(path, max_mols, seed):
    df = pd.read_parquet(path)
    cols = sorted([c for c in df.columns if c.startswith("sigma_")], key=lambda c: int(c.split("_")[1]))
    ids = df["mol_id"].unique()
    if max_mols and len(ids) > max_mols:
        ids = np.random.default_rng(seed).choice(ids, max_mols, replace=False)
    df = df[df["mol_id"].isin(set(ids))]
    mols = []
    for _, g in df.groupby("mol_id", sort=False):
        g = g.sort_values("atom_index")
        z = np.array([ELEMENT_TO_Z.get(e, 6) for e in g["element"]])
        pos = g[["coord_x", "coord_y", "coord_z"]].to_numpy(np.float64)
        y = np.nan_to_num(g[cols].to_numpy(np.float64), nan=0.0)
        mols.append((z, pos, y))
    return mols


def r2(a, b):
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def analysis_area(mols):
    print("\n=== A) profile area vs geometric SASA ===")
    area, sasa = [], {p: [] for p in PROBES}
    zs = []
    for z, pos, y in mols:
        area.append(y.sum(1)); zs.append(z)
        for p in PROBES:
            sasa[p].append(atom_sasa(pos, z, p))
    area = np.concatenate(area); zs = np.concatenate(zs)
    ok = area > 1e-6
    print(f"atoms: {len(area)}, with area>0: {ok.mean():.3f}")
    best = None
    for p in PROBES:
        s = np.concatenate(sasa[p])
        m = ok & (s > 1e-3)
        lin = r2(area[m], s[m]); lg = r2(np.log(area[m]), np.log(s[m]))
        print(f"probe {p:>3} A: R2 linear {lin:.3f} | R2 log-log {lg:.3f}")
        if best is None or lg > best[1]:
            best = (p, lg)
    s = np.concatenate(sasa[best[0]])
    print(f"best probe: {best[0]} A  (set ref_probe_index accordingly; default index 0 = {PROBES[0]} A)")
    print("area/SASA by element (median, IQR) -> if tight, the area is nearly proportional to geometry:")
    for zi in sorted(set(zs.tolist())):
        m = ok & (zs == zi) & (s > 1e-3)
        if m.sum() > 30:
            r = area[m] / s[m]
            print(f"  {SYM.get(zi, zi):>2}: n={m.sum():6d} median={np.median(r):.4g} IQR/median={(np.subtract(*np.percentile(r,[75,25]))/np.median(r)):.3f}")


def analysis_neutrality(mols, seed):
    print("\n=== B) molecular neutrality of screening charge ===")
    rng = np.random.default_rng(seed)
    act, shuf = [], []
    for z, pos, y in mols:
        q = (y * SIGMA_BINS[None, :]).sum(1)
        den = np.abs(q).sum() + EPS
        act.append(abs(q.sum()) / den)
        shuf.append(abs((rng.choice([-1, 1], len(q)) * q).sum()) / den)
    act, shuf = np.array(act), np.array(shuf)
    qs = [50, 90, 99]
    print("|Q|/sum|q|  actual  :", {f"p{k}": round(float(np.percentile(act, k)), 4) for k in qs})
    print("|Q|/sum|q|  shuffled:", {f"p{k}": round(float(np.percentile(shuf, k)), 4) for k in qs})
    frac = float((act < 0.05).mean())
    print(f"fraction of molecules with |Q|/sum|q| < 0.05: {frac:.3f}")
    print("-> neutral: true is justified only if actual is far below shuffled AND the p90 is small;"
          " molecules in the tail are probably ions.")


def wl_labels(z, pos, radius):
    n = len(z)
    rc = np.array([COV_R.get(int(a), 1.0) for a in z])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    adj = (d < 1.2 * (rc[:, None] + rc[None])) & ~np.eye(n, dtype=bool)
    nbrs = [np.where(adj[i])[0] for i in range(n)]
    lab = [int(a) for a in z]
    out = [list(lab)]
    for _ in range(radius):
        lab = [hash((lab[i], tuple(sorted(lab[j] for j in nbrs[i])))) for i in range(n)]
        out.append(list(lab))
    return out  # out[r][i] = label of atom i after r rounds


def analysis_noise(mols, model_wmae, radii=(0, 1, 2, 3)):
    print("\n=== C) leave-one-out lookup-table error by local-topology radius ===")
    Y = np.concatenate([m[2] for m in mols])
    var = np.maximum(Y.var(0), EPS)
    w = var / var.sum()
    w = w / w.sum()
    labels = {r: [] for r in radii}
    for z, pos, y in mols:
        L = wl_labels(z, pos, max(radii))
        for r in radii:
            labels[r].extend(L[r])
    print(f"benchmark reference wMAE: {model_wmae}")
    for r in radii:
        groups = defaultdict(list)
        for i, lab in enumerate(labels[r]):
            groups[lab].append(i)
        errs, covered = [], 0
        for idx in groups.values():
            n = len(idx)
            if n < 2:
                continue
            Yg = Y[idx]
            loo = (Yg.sum(0, keepdims=True) - Yg) / (n - 1)
            errs.append((np.abs(Yg - loo) * w[None]).sum(1))
            covered += n
        if not errs:
            print(f"  r={r}: no groups")
            continue
        e = np.concatenate(errs)
        print(f"  r={r}: LOO wMAE {e.mean():.4f} on {covered/len(Y):.1%} of atoms in {len(groups)} groups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--max-mols", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-wmae", type=float, default=0.109)
    ap.add_argument("--skip", nargs="*", default=[], choices=["area", "neutral", "noise"])
    a = ap.parse_args()
    mols = load(a.train, a.max_mols, a.seed)
    print(f"molecules: {len(mols)}, atoms: {sum(len(m[0]) for m in mols)}")
    if "area" not in a.skip:
        analysis_area(mols)
    if "neutral" not in a.skip:
        analysis_neutrality(mols, a.seed)
    if "noise" not in a.skip:
        analysis_noise(mols, a.model_wmae)


if __name__ == "__main__":
    main()
