# src/models/models_3d/ — 3D architectures

9 architectures: SchNet, PaiNN, DimeNet, DimeNet++, SphereNet, EGNN,
TorchMD-Net, MACE, Uni-Mol — each a separate file, registered in
`MODEL_REGISTRY_3D` (`__init__.py`), which `src/models/__init__.py`
(unmodified — it already anticipated this) merges into the top-level
`MODEL_REGISTRY` alongside the 2D architectures in `../models_2d/`.

Every file's docstring gives the paper, the official reference
implementation, the PyG layer reused (if any), and an explicit note of
what — if anything — had to be adapted for this benchmark's per-atom
sigma-profile task; see `PROVENANCE.md` §2.5 for the citation-indexed
summary of all 9.

The 3D featurizer lives in its own files (`src/data/geometry_3d.py`,
`constants_3d.py`, `features_3d.py`, `dataset_3d.py`) rather than as a
`mode=` branch inside the existing 2D `src/data/features.py` — see
`PROVENANCE.md` §1 "Bug #2" for why that separation matters.
