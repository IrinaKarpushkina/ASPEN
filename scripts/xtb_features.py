"""Считает GFN2-xTB single-point для каждой молекулы parquet-файла (по ВАШИМ координатам,
без оптимизации геометрии) и сохраняет атомные/молекулярные признаки в .npz.

Запуск (из корня репозитория, в окружении `sigma`):

    python -m scripts.xtb_features --parquet <train.parquet> <val.parquet> <test.parquet> --n-jobs 16

По умолчанию результат кладётся в  <папка parquet>/xtb/<имя parquet>.xtb.npz  — именно оттуда его
читает src/data/dataset_3d_xtb.py (папку можно переопределить переменной ASPEN_XTB_DIR).

Что сохраняется (порядок атомов/молекул ТОЧНО как в ChaosParquet3DDataset:
groupby('mol_id', sort=False) + сортировка по atom_index):
    z, pos          - для проверки совпадения с датасетом при загрузке
    q               - заряды Малликена GFN2 (e)                         (N_atoms,)
    bo_sum, bo_max  - сумма / максимум порядков связей атома           (N_atoms,)
    mol_charge      - полный заряд молекулы, взят из SMILES            (N_mols,)
    mol_dipole      - |дипольный момент| в Дебаях                       (N_mols,)
    mol_gap         - щель HOMO-LUMO в эВ                               (N_mols,)
    ok              - 1, если расчёт сошёлся                            (N_mols,)
Только GFN2 single-point: ~0.003-0.12 с на молекулу на одном ядре (измерено для 9-67 атомов).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # 1 поток на процесс; параллелим по молекулам
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.constants import ELEMENT_TO_Z  # noqa: E402

ANG2BOHR = 1.0 / 0.52917721067
AU2DEBYE = 2.541746473
HARTREE2EV = 27.211386


def _smiles_charge(smiles: str) -> tuple[int, bool]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return 0, False
        return int(sum(a.GetFormalCharge() for a in m.GetAtoms())), True
    except Exception:
        return 0, False


def _one(args):
    """-> dict с массивами для одной молекулы (при сбое ok=0 и нули)."""
    z, pos, charge = args
    n = len(z)
    out = dict(q=np.zeros(n, np.float32), bo_sum=np.zeros(n, np.float32), bo_max=np.zeros(n, np.float32),
               dipole=0.0, gap=0.0, ok=0)
    try:
        from tblite.interface import Calculator
        n_el = int(z.sum()) - int(charge)
        uhf = n_el % 2                                  # радикал/нечётное число электронов
        calc = Calculator("GFN2-xTB", z.astype(np.int32), pos.astype(np.float64) * ANG2BOHR,
                          charge=float(charge), uhf=int(uhf))
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        q = np.asarray(res.get("charges"), dtype=np.float64)
        bo = np.asarray(res.get("bond-orders"), dtype=np.float64)
        if bo.ndim == 3:
            bo = bo[..., 0]
        np.fill_diagonal(bo, 0.0)
        eps = np.asarray(res.get("orbital-energies"), dtype=np.float64).reshape(-1)
        occ = np.asarray(res.get("orbital-occupations"), dtype=np.float64).reshape(-1)
        occupied = occ > 0.5 * occ.max() if occ.size else np.zeros_like(occ, bool)
        gap = (eps[~occupied].min() - eps[occupied].max()) * HARTREE2EV if occupied.any() and (~occupied).any() else 0.0
        dip = np.linalg.norm(np.asarray(res.get("dipole"), dtype=np.float64)) * AU2DEBYE
        if not (np.isfinite(q).all() and np.isfinite(bo).all()):
            return out
        out.update(q=q.astype(np.float32), bo_sum=bo.sum(1).astype(np.float32),
                   bo_max=bo.max(1).astype(np.float32) if n > 1 else np.zeros(n, np.float32),
                   dipole=float(dip), gap=float(gap), ok=1)
    except Exception:
        pass
    return out


def load_molecules(parquet: str, max_mols: int | None):
    """Порядок как в dataset_3d.py: groupby('mol_id', sort=False), внутри sort по atom_index."""
    cols = ["mol_id", "atom_index", "element", "smiles", "coord_x", "coord_y", "coord_z"]
    df = pd.read_parquet(parquet, columns=cols)
    df["_ord"] = pd.factorize(df["mol_id"])[0]
    df = df.sort_values(["_ord", "atom_index"], kind="stable")
    counts = df.groupby("_ord", sort=True).size().to_numpy()
    if max_mols:
        counts = counts[:max_mols]
        df = df.iloc[: int(counts.sum())]
    ids = df.drop_duplicates("_ord")["mol_id"].to_numpy()
    smiles = df.drop_duplicates("_ord")["smiles"].to_numpy()
    z = np.array([ELEMENT_TO_Z.get(e, 6) for e in df["element"]], dtype=np.int16)
    pos = df[["coord_x", "coord_y", "coord_z"]].to_numpy(np.float32)
    off = np.concatenate([[0], np.cumsum(counts)])
    return ids, smiles, z, pos, off


def run_parquet(parquet: str, out_path: Path, n_jobs: int, max_mols: int | None, force: bool):
    if out_path.exists() and not force:
        print(f"[skip] {out_path} уже существует (--force чтобы пересчитать)")
        return
    t0 = time.time()
    ids, smiles, z, pos, off = load_molecules(parquet, max_mols)
    M = len(ids)
    charges, n_bad_smiles = [], 0
    for s in smiles:
        c, ok = _smiles_charge(s)
        charges.append(c)
        n_bad_smiles += (not ok)
    d_min = min(float(np.linalg.norm(pos[off[i]:off[i + 1]][:, None] - pos[off[i]:off[i + 1]][None], axis=-1)
                      [np.triu_indices(off[i + 1] - off[i], 1)].min())
                for i in range(min(M, 200)) if off[i + 1] - off[i] > 1)
    print(f"{os.path.basename(parquet)}: {M} молекул, {len(z)} атомов; SMILES не разобрались: {n_bad_smiles}; "
          f"мин. межатомное расстояние (200 первых мол.) = {d_min:.2f} Å  (должно быть ~0.9-1.1, иначе единицы НЕ ангстремы)")
    jobs = [(z[off[i]:off[i + 1]], pos[off[i]:off[i + 1]], charges[i]) for i in range(M)]

    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            results = []
            for k, r in enumerate(pool.imap(_one, jobs, chunksize=32)):
                results.append(r)
                if (k + 1) % 2000 == 0:
                    print(f"  {k + 1}/{M}  ({time.time() - t0:.0f} с)", flush=True)
    else:
        results = []
        for k, a in enumerate(jobs):
            results.append(_one(a))
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{M}  ({time.time() - t0:.0f} с)", flush=True)

    ok = np.array([r["ok"] for r in results], dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, mol_ids=np.asarray(ids), n_atoms=np.diff(off).astype(np.int32), z=z, pos=pos,
        q=np.concatenate([r["q"] for r in results]), bo_sum=np.concatenate([r["bo_sum"] for r in results]),
        bo_max=np.concatenate([r["bo_max"] for r in results]),
        mol_charge=np.array(charges, np.float32), mol_dipole=np.array([r["dipole"] for r in results], np.float32),
        mol_gap=np.array([r["gap"] for r in results], np.float32), ok=ok)
    bad = np.where(ok == 0)[0]
    print(f"[done] {out_path}  время {time.time() - t0:.0f} с, не сошлось: {len(bad)}/{M} ({100 * len(bad) / M:.2f}%)")
    if len(bad):
        print("  примеры mol_id с ошибкой:", ids[bad[:10]].tolist(),
              "\n  (для них признаки = 0 и флаг ok=0; модель это видит)")


def selftest():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smis = ["CCO", "O=S(=O)([O-])c1cccc(S(=O)(=O)[O-])c1", "ClC(Cl)(I)I", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
    for s in smis:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        AllChem.EmbedMolecule(m, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(m)
        z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        pos = m.GetConformer().GetPositions()
        t = time.time()
        r = _one((z, pos, Chem.GetFormalCharge(m)))
        print(f"{s[:38]:38s} atoms={len(z):3d} ok={r['ok']} {time.time() - t:.3f}s  sum(q)={r['q'].sum():+.2f}  "
              f"gap={r['gap']:.2f} eV  |mu|={r['dipole']:.2f} D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", help="train/val/test parquet с coord_x/y/z")
    ap.add_argument("--out-dir", default=None, help="по умолчанию <папка parquet>/xtb")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-mols", type=int, default=None, help="для быстрого теста")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="проверить, что tblite работает, и выйти")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.parquet:
        ap.error("--parquet обязателен (или используйте --selftest)")
    for p in a.parquet:
        out_dir = Path(a.out_dir or os.environ.get("ASPEN_XTB_DIR") or (Path(p).parent / "xtb"))
        run_parquet(p, out_dir / (os.path.basename(p) + ".xtb.npz"), a.n_jobs, a.max_mols, a.force)


if __name__ == "__main__":
    main()
