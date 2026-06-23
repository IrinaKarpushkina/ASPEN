"""
constants.py — shared physical/chemical constants and the sigma-profile grid.

Single source of truth, imported by every dataset/model/feature module.
Any change here propagates consistently to ALL architectures (GCN, GAT,
GATv2, GINE, SchNet, final model) — this is what "fair comparison" requires.
"""

import numpy as np

# =============================================================================
# SIGMA GRID
# =============================================================================
SIGMA_BINS = np.linspace(-0.025, 0.025, 51)
DELTA_SIGMA = float(SIGMA_BINS[1] - SIGMA_BINS[0])
SIGMA_RANGE = float(SIGMA_BINS[-1] - SIGMA_BINS[0])
POLAR_THRESHOLD = 0.01
POLAR_MASK = (SIGMA_BINS <= -POLAR_THRESHOLD) | (SIGMA_BINS >= POLAR_THRESHOLD)
EPS = 1e-12

# =============================================================================
# ELEMENTS — extended to cover up to Rn (Z=86).
#
# NOTE: previously the GCN/GAT/GATv2/GINE baselines only covered up to Ba
# (Z=56) with nn.Embedding(57, ...), while the SchNet/PaiNN-style script
# covered up to Rn (Z=86). This caused an architecture-dependent feature
# inconsistency if the dataset contains elements with Z > 56 (they would
# silently collapse to a single clamped embedding index in some models but
# not others). We standardise on the larger table for ALL models; the
# z_embed layer size (Z_EMBED_SIZE) is shared too.
# =============================================================================
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
}

# Highest atomic number we explicitly support; nn.Embedding sizes derive from this.
MAX_Z = 86
Z_EMBED_SIZE = MAX_Z + 1  # indices 0..86

# =============================================================================
# ELECTRONEGATIVITY (Pauling scale). Default used for elements not listed.
# =============================================================================
ELECTRONEGATIVITY = {
    1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    10: 0.00, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    18: 0.00, 19: 0.82, 20: 1.00, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 46: 2.20, 47: 1.93, 48: 1.69, 50: 1.96, 52: 2.10,
    53: 2.66, 78: 2.28, 79: 2.54, 80: 2.00, 82: 2.33,
}
ENEG_DEFAULT = 2.0

# =============================================================================
# VAN DER WAALS RADII (Angstrom)
# =============================================================================
VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75,
    35: 1.85, 53: 1.98,
}
VDW_DEFAULT = 1.70

# =============================================================================
# ATOMIC MASSES (amu)
# =============================================================================
ATOMIC_MASS = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.81, 6: 12.01, 7: 14.01, 8: 16.00,
    9: 19.00, 10: 20.18, 11: 22.99, 12: 24.31, 13: 26.98, 14: 28.09, 15: 30.97,
    16: 32.06, 17: 35.45, 18: 39.95, 19: 39.10, 20: 40.08, 26: 55.85, 27: 58.93,
    28: 58.69, 29: 63.55, 30: 65.38, 35: 79.90, 47: 107.87, 53: 126.90,
}
MASS_DEFAULT = 40.0

# =============================================================================
# HYDROGEN-BOND DONOR / ACCEPTOR ATOMS
# =============================================================================
HB_DONOR_Z = {7, 8, 9}              # N, O, F
HB_ACCEPTOR_Z = {7, 8, 9, 16, 17}   # N, O, F, S, Cl

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
CUTOFF = 12.0          # Angstrom — radius graph cutoff, identical for ALL models
MAX_NUM_NEIGHBORS = 32  # cap applied identically across all models (see features.py)
N_RBF = 32             # number of Gaussian RBF centres for edge features (GINE/final)

# =============================================================================
# NODE / EDGE FEATURE DIMENSIONS
# =============================================================================
N_NODE_FEAT_BASE = 7    # eneg, vdw, Z/maxZ, mass, degree, HB donor, HB acceptor
