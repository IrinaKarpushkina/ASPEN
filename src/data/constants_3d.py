"""
constants_3d.py — 3D-benchmark-only constants.

Kept in its own file rather than added to `constants.py` so that the 2D
benchmark's single source of truth is never touched (see PROVENANCE.md,
"3D benchmark" section, and `tests/test_features_2d_only.py`, which would
fail if 3D-only concepts like a cutoff radius leaked into the 2D module).

Design choice — ONE shared cutoff/neighbour graph for ALL 3D architectures
(see PROVENANCE.md, "Bug #3" in the original repo, which flagged this
exact risk): every 3D model in this benchmark consumes the SAME radius
graph, built once per molecule in `features_3d.py`. This is the 3D
equivalent of "every 2D model sees the same RDKit bond graph" — controlled
comparison per Dwivedi et al. (JMLR 2022).
"""

# Radius-graph cutoff (Angstrom) and per-atom neighbour cap, shared by every
# 3D architecture. 5.0 A / 32 neighbours matches the cutoff used to train
# the original SchNet/DimeNet/DimeNet++ QM9 models (Schutt et al. 2017;
# Klicpera et al. 2020) and keeps the graph sparse enough for small organic
# molecules and ions that GCN/GAT-style architectures without edge features
# do not degenerate into an oversmoothed near-complete graph (see
# PROVENANCE.md Bug #3).
CUTOFF = 5.0
MAX_NUM_NEIGHBORS = 32

# Radial basis function count, shared by SchNet/PaiNN/DimeNet-family/EGNN/
# TorchMD-Net/MACE distance embeddings (each model still uses its own,
# architecture-specific RBF *kind* — Gaussian vs Bessel — but the same
# *count*, so the amount of distance resolution given to every model is
# matched).
N_RBF = 32

# DimeNet / DimeNet++ / SphereNet spherical-basis order (number of
# spherical harmonics degrees used in the angular basis).
N_SPHERICAL = 7

# Node features (3D): reuses the SAME 14 physically-motivated atom features
# as the 2D benchmark (electronegativity, vdW radius, Z, mass, H-bond
# donor/acceptor, hybridisation, formal charge, aromaticity, ring
# membership, Gasteiger charge, polarizability — see src/data/features.py
# docstring) MINUS ring/hybridisation/aromaticity, which are 2D-topology
# concepts computed from RDKit's bond graph, not from geometry. The 3D
# featurizer instead adds nothing topological: each 3D model must recover
# any structural information purely from (Z, position). This is the
# deliberate 2D/3D symmetry-breaking point of the whole benchmark study —
# see PROVENANCE.md.
N_NODE_FEAT_3D = 6  # eneg, vdw, Z(scaled), mass, HB-donor, HB-acceptor
