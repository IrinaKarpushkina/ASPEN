"""Geometric solvent-accessible-surface-area (SASA) features, numpy only.

Why: the integral of an atomic sigma-profile is the area of the cavity surface
that belongs to that atom. That area is a *geometric* quantity: it follows from
coordinates + radii. A cutoff-limited GNN has to re-learn it from a 5 A
neighbourhood; here it is computed exactly (Shrake-Rupley) and fed to the model.

Three probe radii are returned so that ``scripts/ceiling_analysis.py`` can tell
you empirically which one correlates best with the target area.
"""
from __future__ import annotations

import numpy as np

from .constants import VDW_RADII, VDW_DEFAULT

COSMO_RADIUS_SCALE = 1.17          # COSMO-style cavity radii ~ 1.17 x vdW
PROBES = (0.0, 0.8, 1.4)            # Angstrom; index 0 is the default reference
SASA_INPUT_SCALE = 30.0             # divide by this before feeding to the network


def _sphere_points(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_PTS = {}


def atom_sasa(pos: np.ndarray, z: np.ndarray, probe: float, n_points: int = 96,
              radius_scale: float = COSMO_RADIUS_SCALE) -> np.ndarray:
    """Per-atom SASA (A^2) with Shrake-Rupley. pos: (N,3), z: (N,)."""
    pos = np.asarray(pos, dtype=np.float64)
    n = len(z)
    pts = _PTS.setdefault(n_points, _sphere_points(n_points))
    r = np.array([VDW_RADII.get(int(zi), VDW_DEFAULT) for zi in z]) * radius_scale
    R = r + probe
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    out = np.zeros(n, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n):
        nb = np.where((d[i] < R[i] + R + 1e-6) & (idx != i))[0]
        if len(nb) == 0:
            exposed = n_points
        else:
            surf = pos[i] + R[i] * pts                                   # (P,3)
            dd = np.linalg.norm(surf[:, None, :] - pos[nb][None, :, :], axis=-1)  # (P,k)
            exposed = n_points - int((dd < R[nb][None, :]).any(axis=1).sum())
        out[i] = 4.0 * np.pi * R[i] ** 2 * exposed / n_points
    return out


def sasa_features_np(pos: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(N, len(PROBES)) float32, raw A^2 (not scaled)."""
    return np.stack([atom_sasa(pos, z, p) for p in PROBES], axis=1).astype(np.float32)
