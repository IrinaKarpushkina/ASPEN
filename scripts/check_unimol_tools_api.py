"""
check_unimol_tools_api.py - run this FIRST, before
UniMolPretrainedDataset on the full dataset, to confirm your installed
unimol_tools version's UniMolRepr.get_repr() actually accepts a custom
{'atoms', 'coordinates'} dict (used by dataset_unimol_pretrained.py to
reuse THIS benchmark's own 3D coordinates instead of letting unimol_tools
generate its own RDKit conformer -- see that file's module docstring for
why this matters).

Also downloads (if not already cached) the pretrained checkpoint from
Hugging Face on first run, so this doubles as a connectivity/setup check.

Usage:
    python -m scripts.check_unimol_tools_api
"""
import numpy as np


def main():
    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        print("FAIL: unimol_tools is not installed. Run:")
        print("  pip install unimol-tools huggingface_hub")
        return

    print("Loading UniMolRepr (downloads the pretrained checkpoint from "
          "Hugging Face on first run -- can take a while / needs network "
          "access to huggingface.co)...")
    try:
        repr_model = UniMolRepr(data_type="molecule", remove_hs=False)
    except Exception as e:
        print(f"FAIL: could not initialize UniMolRepr: {e}")
        print("Common cause: no network access to huggingface.co from this "
              "machine. If your cluster has restricted network egress, you "
              "may need to download the checkpoint on a machine WITH access "
              "and set HF_ENDPOINT to a mirror, or point "
              "pretrained_model_path/pretrained_dict_path at local files -- "
              "see UniMolRepr's own constructor arguments.")
        return

    # Toy molecule: ethanol (C-C-O), explicit Hs, with an arbitrary
    # (non-degenerate) coordinate set -- this script only checks that the
    # API ACCEPTS custom coordinates, not that the specific numbers are
    # chemically meaningful.
    atoms = [["C", "C", "O", "H", "H", "H", "H", "H", "H"]]
    rng = np.random.RandomState(0)
    coords = [rng.randn(9, 3).astype(np.float32) * 1.2]

    print("\nTrying UniMolRepr.get_repr({'atoms':..., 'coordinates':...}, "
          "return_atomic_reprs=True) ...")
    try:
        out = repr_model.get_repr(
            {"atoms": atoms, "coordinates": coords}, return_atomic_reprs=True,
        )
        atomic = np.asarray(out["atomic_reprs"][0])
        cls = np.asarray(out.get("cls_repr", [None])[0]) if "cls_repr" in out else None
        print(f"SUCCESS: got atomic_reprs of shape {atomic.shape} "
              f"(expected first dim == 9 atoms).")
        if atomic.shape[0] == 10 and cls is not None:
            # Confirmed finding (see dataset_unimol_pretrained.py): unimol_tools
            # prepends a whole-molecule [CLS] representation as row 0 (paper
            # Sec. 2.2, BERT-style convention). Verify it directly by comparing
            # row 0 against the separately-returned cls_repr, rather than just
            # assuming position from the count alone.
            matches_row0 = np.allclose(atomic[0], cls, atol=1e-4)
            print(f"NOTE: got {atomic.shape[0]} rows for 9 atoms (+1). Checking "
                  f"whether row 0 matches the separately-returned cls_repr "
                  f"(i.e. a prepended [CLS] token, per paper Sec. 2.2)... "
                  f"{'CONFIRMED' if matches_row0 else 'NOT confirmed'}.")
            if matches_row0:
                print("=> dataset_unimol_pretrained.py already strips row 0 "
                      "before the atom-count check -- no action needed.")
            else:
                print("=> row 0 does NOT match cls_repr -- the +1 row is "
                      "something else in your installed unimol_tools version. "
                      "dataset_unimol_pretrained.py's current fix (assumes "
                      "leading [CLS] at index 0) may be WRONG for your version "
                      "-- inspect out['atomic_reprs'][0] vs out['cls_repr'][0] "
                      "yourself before trusting per-atom alignment.")
        elif atomic.shape[0] != 9:
            print(f"WARNING: expected 9 atoms, got {atomic.shape[0]} (not the "
                  f"+1-for-[CLS] case checked above either). This might mean "
                  f"remove_hs behaved unexpectedly, or atoms were "
                  f"reordered/dropped -- investigate before trusting "
                  f"per-atom alignment on the real dataset.")
        print(f"repr_dim = {atomic.shape[1]} -- put this in "
              f"configs/unimol_pretrained/unimol_pretrained.yaml's "
              f"model.repr_dim if it differs from the default (512).")
        print("\n=> dataset_unimol_pretrained.py's use_own_coords=True path "
              "should work as written.")
    except TypeError as e:
        print(f"FAIL (TypeError): {e}")
        print("\n=> Your installed unimol_tools version's UniMolRepr.get_repr "
              "does NOT accept the {'atoms','coordinates'} dict format. "
              "Options:")
        print("  1. Check `pip show unimol-tools` version and compare "
              "against https://github.com/deepmodeling/unimol_tools "
              "(the dict format is documented for MolTrain/MolPredict; "
              "get_repr may need a different call convention in your "
              "installed version -- check its source directly:")
        print("       python -c \"import unimol_tools, inspect; "
              "print(inspect.getsource(unimol_tools.UniMolRepr.get_repr))\"")
        print("  2. As a fallback, set use_own_coords: false in the config "
              "(unimol_tools will generate its own RDKit conformer from "
              "SMILES instead of using this benchmark's coordinates -- "
              "understand this means different 3D geometry than every "
              "other model in the benchmark sees, see dataset docstring).")
    except Exception as e:
        print(f"FAIL (unexpected error): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
