"""Additional invariant 3D features for the physics ablation.

The six benchmark 3D features remain untouched.  This module adds 16 features
computed only from the already cached 3D graph, atomic numbers, coordinates,
and the existing physical constants in ``src.data.constants``.

No topology, SMILES-derived descriptors, target values, or duplicated element
lookup tables are introduced here.
"""
from __future__ import annotations

import numpy as np
import torch

from .constants import (
    ELECTRONEGATIVITY,
    ENEG_DEFAULT,
    POLARIZABILITY,
    POLARIZABILITY_DEFAULT,
    MAX_Z,
)

ENHANCED_FEATURE_DIM = 22  # 6 benchmark + 1 polarizability + 15 geometry


def build_local_geometry_features(data: object) -> torch.Tensor:
    """Build 15 rotation/translation-invariant local-environment features."""
    pos = data.pos.detach().cpu().numpy().astype(np.float32, copy=False)
    z_np = data.z.detach().cpu().numpy().astype(np.float32, copy=False)
    en_np = np.asarray(
        [ELECTRONEGATIVITY.get(int(z), ENEG_DEFAULT) for z in z_np],
        dtype=np.float32,
    )

    n = len(z_np)
    out = np.zeros((n, 15), dtype=np.float32)
    edge_index = data.edge_index.detach().cpu().numpy()
    edge_weight = data.edge_weight.detach().cpu().numpy().astype(np.float32, copy=False)

    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for e in range(edge_index.shape[1]):
        j = int(edge_index[0, e])
        i = int(edge_index[1, e])
        d = float(edge_weight[e])
        if i != j and np.isfinite(d) and d > 1e-6:
            nbrs[i].append((j, d))

    for i, items in enumerate(nbrs):
        if not items:
            continue

        js = np.asarray([j for j, _ in items], dtype=np.int64)
        d = np.asarray([dist for _, dist in items], dtype=np.float32)

        # 0-4: normalized neighbour counts in nested radial shells.
        for k, radius in enumerate((1.8, 2.2, 2.8, 3.5, 5.0)):
            out[i, k] = np.sum(d < radius) / 10.0

        # 5-7: radial density descriptors.
        out[i, 5] = np.sum(np.exp(-d / 1.0)) / 10.0
        out[i, 6] = np.sum(np.exp(-d / 2.0)) / 10.0
        out[i, 7] = np.sum(np.exp(-d / 3.0)) / 10.0

        # 8: short-range inverse-distance density.
        out[i, 8] = np.sum(1.0 / np.maximum(d, 0.5)) / 10.0

        # Distance-weighted neighbour chemistry.
        w = np.exp(-d / 2.0)
        w /= w.sum() + 1e-8
        en_mean = float(np.sum(w * en_np[js]))
        z_mean = float(np.sum(w * np.minimum(z_np[js], float(MAX_Z))))
        out[i, 9] = en_mean / 4.0
        out[i, 10] = z_mean / float(MAX_Z)
        out[i, 11] = np.sqrt(np.sum(w * (en_np[js] - en_mean) ** 2)) / 4.0

        # 12-14: eigenvalues of weighted directional covariance.
        vec = pos[js] - pos[i]
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        u = vec / np.maximum(norms, 1e-8)
        cov = (u.T * w) @ u
        try:
            eig = np.linalg.eigvalsh(cov)
            out[i, 12:15] = np.sort(np.clip(eig, 0.0, None))
        except np.linalg.LinAlgError:
            pass

    return torch.from_numpy(out)


def augment_3d_features(data: object) -> torch.Tensor:
    """Return the benchmark six features plus 16 additional features."""
    if data.x.ndim != 2 or data.x.shape[1] != 6:
        raise ValueError(
            f"Expected base 3D features with shape (N, 6), got {tuple(data.x.shape)}"
        )

    z = data.z.detach().cpu().tolist()
    pol = torch.tensor(
        [[POLARIZABILITY.get(int(zi), POLARIZABILITY_DEFAULT) / 6.0] for zi in z],
        dtype=torch.float32,
    )
    local = build_local_geometry_features(data)
    enhanced = torch.cat([data.x.float(), pol, local], dim=1)

    if enhanced.shape[1] != ENHANCED_FEATURE_DIM:
        raise RuntimeError(
            f"Enhanced feature dimension mismatch: expected {ENHANCED_FEATURE_DIM}, "
            f"got {enhanced.shape[1]}"
        )
    return enhanced
