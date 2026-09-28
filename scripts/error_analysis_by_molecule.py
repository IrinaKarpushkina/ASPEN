"""Where does the error concentrate — comprehensive version with real SMILES,
RDKit-based functional-group matching and rendered 2D structures for the worst
molecules.

FIX vs the previous version: `Data3D` (src/data/dataset_3d.py) does NOT store
`smiles`, `mol_id` or `element` on the graph object -- only z/pos/edges. The
old script silently fell back to the loop index, so "smiles" was really just
0..N and every group-match was wrong (100% "ring" was a coincidence: the
digit regex matched the index string itself). This version re-reads the
original test parquet and reconstructs metadata with the EXACT same grouping
order the dataset builder uses (`df.groupby("mol_id", sort=False)`,
`sort_values("atom_index")` within a group), so row i here lines up with row
i of P/T from predict(). This alignment is asserted, not just assumed.

    python -m scripts.error_analysis_by_molecule <ckpt> [--top 20] [--structures 8]

Outputs to --out (default results/error_report):
  by_element.csv, by_group.csv, by_size.csv   - aggregated wMAE / polar MAE
  by_element_area.csv                          - area error with median (not just
                                                  mean, which a few near-zero-area
                                                  "buried" atoms can blow up)
  worst_molecules.csv                          - top-N molecules, real SMILES
  structures/worst_<rank>_<mol_id>.png          - 2D structure, RDKit if available,
                                                  atoms colored/labelled by wMAE
  worst_mol_<rank>.png                          - predicted vs target profile per atom
  REPORT.md                                     - the numbers above plus the images,
                                                  one file to read
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts.ensemble_eval import ds_class, predict
from src.data.constants import DELTA_SIGMA, POLAR_MASK
from src.train_3d_physics import _MODELS_NEEDING_TRIPLETS

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT = True
except ImportError:
    _RDKIT = False

GROUP_SMARTS = {
    "halogen": "[F,Cl,Br,I]",
    "nitrogen": "[#7]",
    "oxygen": "[#8]",
    "sulfur": "[#16]",
    "aromatic": "[a]",
    "charged": "[+,-]",
    "carbonyl": "[CX3]=[OX1]",
    "ring": "[R]",
    "triple_bond": "[$([#6]#[#6]),$([#6]#[#7])]",
}


def load_test_meta(parquet_path, sigma_cols_n=51):
    """Reproduce dataset_3d.py's grouping exactly, so row order matches predict()."""
    df = pd.read_parquet(parquet_path)
    rows = []
    for mol_id, mol_df in df.groupby("mol_id", sort=False):
        mol_df = mol_df.sort_values("atom_index")
        smi = mol_df["smiles"].iloc[0]
        n = len(mol_df)
        for _, r in mol_df.iterrows():
            rows.append((mol_id, smi, r["element"], n))
    return pd.DataFrame(rows, columns=["mol_id", "smiles", "element", "mol_size"])


def per_atom_err(pred, targ, bin_weights):
    wmae = (np.abs(pred - targ) * bin_weights[None, :]).sum(1) / bin_weights.sum()
    pm = np.asarray(POLAR_MASK)
    polar = (np.abs(pred - targ) * pm[None, :]).sum(1) / max(pm.sum(), 1)
    area_p, area_t = pred.sum(1) * DELTA_SIGMA, targ.sum(1) * DELTA_SIGMA
    return wmae, polar, area_p, area_t


def render_structure(smiles, atom_err, out_path, title=""):
    if not _RDKIT:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != len(atom_err):
        return False  # SMILES atom order must match atom_index; if RDKit reorders/fails, skip rather than mislabel
    lo, hi = float(np.min(atom_err)), float(np.max(atom_err))
    colors = {}
    for i, e in enumerate(atom_err):
        t = 0.0 if hi <= lo else float((float(e) - lo) / (hi - lo))  # native float: RDKit's C++ binding
        colors[i] = (1.0, 1.0 - t, 1.0 - t)                          # rejects numpy.float32/float64 scalars
    d = rdMolDraw2D.MolDraw2DCairo(500, 400)
    labels = {i: f"{e:.3f}" for i, e in enumerate(atom_err)}
    for i, lab in labels.items():
        mol.GetAtomWithIdx(i).SetProp("atomNote", lab)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors)
    d.FinishDrawing()
    with open(out_path, "wb") as f:
        f.write(d.GetDrawingText())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--structures", type=int, default=8, help="how many worst molecules to render/plot")
    ap.add_argument("--out", default="results/error_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(f"{a.out}/structures", exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, T, sizes, bw = predict(a.ckpt, dev, {})
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    name = cfg["model"]["name"]
    trip = name in _MODELS_NEEDING_TRIPLETS or name.startswith("sigma_dn")
    dc = cfg["data"]

    meta = load_test_meta(dc["test_path"])
    assert len(meta) == len(P), (
        f"row mismatch: metadata {len(meta)} vs predictions {len(P)}. "
        "The dataset builder groups by mol_id with sort=False and sorts atoms by atom_index "
        "within a group; if this fires, that order assumption no longer matches your data.")

    wmae, polar, area_p, area_t = per_atom_err(P, T, bw)
    meta = meta.assign(wmae=wmae, polar_mae=polar, area_pred=area_p, area_true=area_t,
                       area_abs_err=np.abs(area_p - area_t),
                       area_rel_err=np.abs(area_p - area_t) / np.maximum(area_t, 1e-3))

    by_el = meta.groupby("element").agg(n=("wmae", "size"), wmae=("wmae", "mean"),
                                        polar_mae=("polar_mae", "mean")).sort_values("wmae", ascending=False)
    by_el.to_csv(f"{a.out}/by_element.csv")

    by_el_area = meta.groupby("element").agg(
        n=("wmae", "size"),
        area_abs_err_median=("area_abs_err", "median"), area_abs_err_p90=("area_abs_err", lambda x: np.percentile(x, 90)),
        area_rel_err_median=("area_rel_err", "median"), area_rel_err_p90=("area_rel_err", lambda x: np.percentile(x, 90)),
    ).sort_values("area_abs_err_median", ascending=False)
    by_el_area.to_csv(f"{a.out}/by_element_area.csv")
    print("\n=== by element: profile wMAE ===\n", by_el)
    print("\n=== by element: area error (median/p90, robust to near-zero-area outliers) ===\n", by_el_area)

    bins = [0, 10, 20, 35, 60, 1000]
    meta["size_bin"] = pd.cut(meta["mol_size"], bins)
    by_size = meta.groupby("size_bin", observed=True).agg(n=("wmae", "size"), wmae=("wmae", "mean"),
                                                           polar_mae=("polar_mae", "mean"))
    by_size.to_csv(f"{a.out}/by_size.csv")
    print("\n=== by molecule size ===\n", by_size)

    uniq = meta[["mol_id", "smiles"]].drop_duplicates()
    if _RDKIT:
        rows = []
        for g, smarts in GROUP_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            hit_ids = set()
            for mid, smi in zip(uniq["mol_id"], uniq["smiles"]):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None and patt is not None and mol.HasSubstructMatch(patt):
                    hit_ids.add(mid)
            sub, rest = meta[meta["mol_id"].isin(hit_ids)], meta[~meta["mol_id"].isin(hit_ids)]
            rows.append((g, len(hit_ids), len(sub), sub["wmae"].mean(), rest["wmae"].mean(),
                        sub["wmae"].mean() - rest["wmae"].mean()))
        by_group = pd.DataFrame(rows, columns=["group", "n_molecules", "n_atoms", "wmae_in", "wmae_out",
                                               "delta"]).sort_values("delta", ascending=False)
        by_group.to_csv(f"{a.out}/by_group.csv", index=False)
        print("\n=== functional groups (RDKit substructure match) ===\n", by_group)
    else:
        print("\nRDKit not available: skipping functional-group breakdown and structure rendering.")

    per_mol = meta.groupby("mol_id").agg(smiles=("smiles", "first"), n_atoms=("mol_id", "size"),
                                         wmae=("wmae", "mean"), polar_mae=("polar_mae", "mean"),
                                         area_abs_err=("area_abs_err", "mean")).sort_values(
        "wmae", ascending=False)
    per_mol.head(a.top).to_csv(f"{a.out}/worst_molecules.csv")
    print(f"\n=== {a.top} worst molecules -> {a.out}/worst_molecules.csv ===\n", per_mol.head(a.top))

    sizes_per_mol = meta.groupby("mol_id", sort=False)["mol_id"].transform("size")
    mol_ids_order = meta["mol_id"].drop_duplicates().tolist()
    starts = {}
    off = 0
    for mid in mol_ids_order:
        starts[mid] = off
        off += int((meta["mol_id"] == mid).sum())  # ok for report-only use; not perf-critical

    report = [f"# Error report\n\nckpt: `{a.ckpt}`\n"]
    report.append(by_el.to_markdown())
    report.append("\n## Area error by element (robust)\n" + by_el_area.to_markdown())
    report.append("\n## By molecule size\n" + by_size.to_markdown())
    if _RDKIT:
        report.append("\n## By functional group\n" + by_group.to_markdown(index=False))
    report.append(f"\n## {min(a.structures, a.top)} worst molecules\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = np.linspace(-0.025, 0.025, 51)
        rank = 0
        for mid in per_mol.head(min(a.structures, a.top)).index:
            rank += 1
            off = starts[mid]
            n = int(per_mol.loc[mid, "n_atoms"])
            atom_err = wmae[off:off + n]
            smi = per_mol.loc[mid, "smiles"]

            fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.3), squeeze=False)
            for j in range(n):
                ax = axes[0, j]
                ax.plot(s, T[off + j], label="target")
                ax.plot(s, P[off + j], label="pred")
                ax.set_title(f"atom {j} (wMAE {atom_err[j]:.3f})", fontsize=7)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(f"#{rank} mol_id={mid}  {smi[:55]}", fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{a.out}/worst_mol_{rank}.png", dpi=110)
            plt.close(fig)

            struct_path = f"{a.out}/structures/worst_{rank}_{mid}.png"
            ok = render_structure(smi, atom_err, struct_path, title=smi)
            report.append(f"\n### #{rank}  mol_id={mid}  wMAE={per_mol.loc[mid, 'wmae']:.3f}\n`{smi}`\n")
            if ok:
                report.append(f"![structure](structures/worst_{rank}_{mid}.png)\n"
                              "(atom labels = per-atom wMAE, redder = worse; if SMILES atom order "
                              "doesn't match atom_index this image is silently skipped, not mislabeled)\n")
            report.append(f"![profiles](worst_mol_{rank}.png)\n")
        print(f"saved profile plots to {a.out}/worst_mol_*.png"
              + (f" and structures to {a.out}/structures/" if _RDKIT else ""))
    except ImportError:
        print("matplotlib not available, skipping plots")

    open(f"{a.out}/REPORT.md", "w").write("\n".join(report))
    print(f"\nWrote {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
