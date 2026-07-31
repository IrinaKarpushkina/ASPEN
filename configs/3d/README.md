# configs/3d/ — placeholder

Empty on purpose. This repository is 2D-only (see top-level README.md and
PROVENANCE.md). When a 3D benchmark is added (SchNet, PaiNN, ...), its
configs go here, following the same `defaults: ../base.yaml` pattern used
in `configs/*.yaml`, plus whatever 3D-specific `data:` keys the new
featurizer needs (cutoff, n_rbf, max_num_neighbors, ...).

Before writing these, read PROVENANCE.md §1 "Bug #3" — it documents a
specific pitfall (radius-graph density for architectures that ignore
edge_attr) worth avoiding from the start.
