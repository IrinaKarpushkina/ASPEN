"""RUN THIS BEFORE TRUSTING THE BOND GRAPH / CHEMICAL FEATURES ON 3D DATA.

    python -m scripts.check_bond_graph_coverage --train <train.parquet> --test <test.parquet>

Reuses src/data/features.py's mol_from_smiles_validated (already used and presumably
working in your 2D benchmark) and reports what fraction of 3D molecules get a
validated RDKit atom-order match. features.py falls back to an empty/zero bond graph
per-molecule on mismatch (not a crash, not garbage), but if the match rate is low the
bond-graph/chemical-feature branch is only informative for a minority of your data and
the expected benefit shrinks accordingly - decide whether to enable
sigma_kitchen_sink.yaml's bond branch based on this number, not just on faith.
"""
import argparse

import pandas as pd

from src.data.constants import ELEMENT_TO_Z
from src.data.features import mol_from_smiles_validated


def check(path, max_mols):
    df = pd.read_parquet(path)
    n_ok, n_tot, bad_examples = 0, 0, []
    for mol_id, g in df.groupby("mol_id", sort=False):
        if max_mols and n_tot >= max_mols:
            break
        n_tot += 1
        smi = g["smiles"].iloc[0]
        g = g.sort_values("atom_index")
        z = [ELEMENT_TO_Z.get(e, -1) for e in g["element"]]
        mol = mol_from_smiles_validated(smi, z)
        if mol is not None:
            n_ok += 1
        elif len(bad_examples) < 10:
            bad_examples.append(str(smi))
    return n_ok, n_tot, bad_examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--max-mols", type=int, default=5000)
    a = ap.parse_args()
    for name, path in [("train", a.train), ("test", a.test)]:
        if not path:
            continue
        ok, tot, bad = check(path, a.max_mols)
        print(f"{name}: {ok}/{tot} molecules ({100*ok/max(tot,1):.1f}%) have a validated RDKit atom-order match")
        if bad:
            print(f"  examples that FAILED (mismatch or unparseable SMILES): {bad}")
    print("\nRule of thumb: >90% -> enable the bond branch with confidence. "
          "50-90% -> enable it, but expect a diluted effect. <50% -> the branch will mostly "
          "see empty graphs; fix the SMILES/atom_index source before spending GPU time on it.")
