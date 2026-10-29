"""Проверки surface-модели на игрушечных данных (CPU, без parquet/xTB).

    python -m pytest tests/test_surface_field.py -q
"""
import numpy as np
import torch

from src.data.features_3d_sasa import atom_sasa
from src.physics_extension.surface_field import (ANG2BOHR, fibonacci_sphere, random_rotation,
                                                 surface_point_features)


def _toy(seed=0, n=7):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * 1.6
    z = torch.tensor([6, 1, 8, 7, 17, 1, 6][:n])
    q = torch.randn(n, generator=g) * 0.3
    return pos, z, q


def test_rotation_covariance():
    pos, z, q = _toy()
    batch = torch.zeros(len(z), dtype=torch.long)
    R = random_rotation()
    f0, e0 = surface_point_features(pos, z, q, batch, 32, rot=None)
    f1, e1 = surface_point_features(pos @ R.T, z, q, batch, 32, rot=R)
    assert torch.allclose(e0, e1)
    assert torch.allclose(f0, f1, atol=2e-4), (f0 - f1).abs().max()


def test_exposure_matches_shrake_rupley():
    pos, z, q = _toy()
    batch = torch.zeros(len(z), dtype=torch.long)
    P = 96
    f, e = surface_point_features(pos, z, q, batch, P, use_field=False)
    from src.data.features_3d_sasa import COSMO_RADIUS_SCALE
    from src.data.constants import VDW_RADII, VDW_DEFAULT
    R = np.array([VDW_RADII.get(int(a), VDW_DEFAULT) for a in z]) * COSMO_RADIUS_SCALE
    mine = (4 * np.pi * R ** 2 * e.mean(1).numpy())
    ref = atom_sasa(pos.numpy(), z.numpy(), 0.0, n_points=P)
    assert np.allclose(mine, ref, rtol=1e-3, atol=1e-2), (mine, ref)


def test_potential_matches_bruteforce():
    pos, z, q = _toy()
    batch = torch.zeros(len(z), dtype=torch.long)
    P = 16
    f, e = surface_point_features(pos, z, q, batch, P, use_field=True)
    U = fibonacci_sphere(P)
    from src.physics_extension.surface_field import _vdw_lut
    i = 2
    R = _vdw_lut()[z[i]] * 1.17
    for p in (0, 5, 11):
        x = pos[i] + R * U[p]
        phi = 0.0
        E = torch.zeros(3)
        for j in range(len(z)):
            if j == i:
                continue
            r = (x - pos[j]) * ANG2BOHR
            rn = max(float(r.norm()), 1.0)
            phi += float(q[j]) / rn
            E += float(q[j]) * r / rn ** 3
        En = float(E @ U[p])
        assert abs(float(f[i, p, 0]) - float(torch.asinh(torch.tensor(phi / 0.02)))) < 1e-3
        assert abs(float(f[i, p, 1]) - float(torch.asinh(torch.tensor(En / 0.02)))) < 1e-3


def test_batching_equals_single():
    pos, z, q = _toy(1)
    pos2, z2, q2 = _toy(2, n=5)
    fb, _ = surface_point_features(torch.cat([pos, pos2]), torch.cat([z, z2]), torch.cat([q, q2]),
                                   torch.tensor([0] * 7 + [1] * 5), 24)
    f1, _ = surface_point_features(pos, z, q, torch.zeros(7, dtype=torch.long), 24)
    f2, _ = surface_point_features(pos2, z2, q2, torch.zeros(5, dtype=torch.long), 24)
    assert torch.allclose(fb, torch.cat([f1, f2]), atol=1e-5)


def test_chunking_equals_full():
    mols = [_toy(i, n=n) for i, n in enumerate([7, 5, 7, 6, 4])]
    pos = torch.cat([m[0] for m in mols]); z = torch.cat([m[1] for m in mols]); q = torch.cat([m[2] for m in mols])
    b = torch.cat([torch.full((len(m[1]),), i) for i, m in enumerate(mols)])
    full = surface_point_features(pos, z, q, b, 16, max_elems=1e12)
    part = surface_point_features(pos, z, q, b, 16, max_elems=3000)
    assert torch.allclose(full[0], part[0], atol=1e-5) and torch.allclose(full[1], part[1])
