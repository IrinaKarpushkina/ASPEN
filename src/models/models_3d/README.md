# src/models/models_3d/ — placeholder

Empty on purpose. This repository is 2D-only (see top-level README.md and
PROVENANCE.md). Future 3D architectures (SchNet, PaiNN, DimeNet++, ...) go
here as separate files, registered in `src/models/__init__.py` alongside
the existing 2D `MODEL_REGISTRY` (or in a separate `MODEL_REGISTRY_3D` —
your call), each with the same provenance-comment style used in
`src/models/gcn.py` etc.: paper, official repo, PyG layer used, and an
explicit note of what (if anything) had to be adapted for this project's
per-atom sigma-profile task.

Keep the 3D featurizer in its own file (e.g. `src/data/features_3d.py`)
rather than adding a `mode=` branch to the existing 2D
`src/data/features.py` — see PROVENANCE.md §1 "Bug #2" for why that
separation matters.
