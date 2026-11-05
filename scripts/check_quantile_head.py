"""check_quantile_head.py - проверки квантильной головы ДО обучения (запуск из корня проекта, ~1-3 мин, GPU не нужен).

1) потолок представления на реальных данных: каждый истинный атомный профиль переводится в квантильную функцию
   на сетке tau (ровно так, как её может выразить голова) и обратно в 51 бин; печатаются weighted R2 / wMAE /
   polar MAE / EMD восстановления. Это верхняя граница качества, которую голова может достичь в принципе.
   Если потолок R2 < ~0.995 или polar MAE потолка сравним с 0.014 - сетку tau надо менять ДО обучения.
2) корректность torch-кода головы: массы >= 0, сумма = 1, совпадение с numpy-эталоном, конечные градиенты,
   форма выхода модели и число параметров против sigma_dn_geom.

python -m scripts.check_quantile_head --parquet <test parquet> [--n-atoms 30000] [--config configs/3d/sigma_dn_geom_qf.yaml]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.data.constants import DELTA_SIGMA, SIGMA_BINS, POLAR_MASK

EDGES = np.concatenate([[SIGMA_BINS[0] - DELTA_SIGMA / 2], (SIGMA_BINS[:-1] + SIGMA_BINS[1:]) / 2,
                        [SIGMA_BINS[-1] + DELTA_SIGMA / 2]])


def tau_grid_np(k, kind):
    i = np.arange(k + 1, dtype=np.float64)
    t = (1 - np.cos(np.pi * i / k)) / 2 if kind == "cos" else i / k
    t[0], t[-1] = 0.0, 1.0
    return t


def hist_to_knots(mass, tau):
    """mass: (n, 51) нормированные массы бинов (масса равномерна внутри бина) -> (n, K+1) узлы Q(tau)."""
    F = np.concatenate([np.zeros((len(mass), 1)), np.cumsum(mass, 1)], 1)
    F[:, -1] = 1.0
    F = np.maximum.accumulate(F, axis=1)
    q = np.empty((len(mass), len(tau)))
    nz = mass > 0
    first = nz.argmax(1); last = 50 - nz[:, ::-1].argmax(1)
    for i, t in enumerate(tau):
        if t <= 0:
            q[:, i] = EDGES[first]; continue
        if t >= 1:
            q[:, i] = EDGES[last + 1]; continue
        j = np.clip((F < t).sum(1), 1, 51)                       # F[j-1] < t <= F[j]
        f0 = F[np.arange(len(F)), j - 1]; f1 = F[np.arange(len(F)), j]
        q[:, i] = EDGES[j - 1] + (t - f0) / np.maximum(f1 - f0, 1e-15) * DELTA_SIGMA
    return np.maximum.accumulate(q, axis=1)


def knots_to_mass_np(q, tau):
    a, b = q[:, :-1], q[:, 1:]
    w = np.maximum(b - a, 1e-12)
    dt = np.diff(tau)
    out = np.zeros((len(q), 51))
    for j in range(51):
        ov = np.clip(np.minimum(b, EDGES[j + 1]) - np.maximum(a, EDGES[j]), 0, None)
        out[:, j] = (dt[None] * ov / w).sum(1)
    return out


def metrics(t, p, w):
    w = w / w.sum()
    ss_res = (((t - p) ** 2) * w).sum(1).mean()
    mt = (t * w).sum(1, keepdims=True)
    ss_tot = (((t - mt) ** 2) * w).sum(1).mean()
    from scipy.stats import wasserstein_distance as wd
    sub = np.random.default_rng(0).choice(len(t), min(3000, len(t)), replace=False)
    emd = np.mean([wd(SIGMA_BINS, SIGMA_BINS, t[i] + 1e-12, p[i] + 1e-12) for i in sub])
    return dict(weighted_r2=1 - ss_res / ss_tot, weighted_mae=(np.abs(t - p) * w).sum(1).mean(),
                polar_mae=np.abs(t[:, POLAR_MASK] - p[:, POLAR_MASK]).mean(), emd_raw=emd)


def ceiling(parquet, n_atoms, grids):
    import pyarrow.parquet as pq
    names = pq.read_schema(parquet).names              # колонки бывают sigma_0 или sigma_00 - берём как в датасете
    cols = sorted([c for c in names if c.startswith("sigma_")], key=lambda c: int(c.split("_")[1]))
    assert len(cols) == 51, f"ожидалось 51 колонка sigma_*, найдено {len(cols)}"
    y = np.nan_to_num(pd.read_parquet(parquet, columns=cols).to_numpy(np.float64))
    w_bins = np.maximum(y.var(0), 1e-12)                          # веса бинов как в бенчмарке (здесь по этому файлу)
    rng = np.random.default_rng(0)
    y = np.maximum(y[rng.choice(len(y), min(n_atoms, len(y)), replace=False)], 0)
    area = y.sum(1) * DELTA_SIGMA
    ok = area > 1e-8
    mass = np.zeros_like(y); mass[ok] = y[ok] * DELTA_SIGMA / area[ok, None]
    print(f"\n[1] потолок представления: {len(y)} атомов из {parquet} ({(~ok).sum()} с нулевой площадью)")
    print(f"    {'сетка tau':18} " + " ".join(f"{k:>12}" for k in ("weighted_r2", "weighted_mae", "polar_mae", "emd_raw")))
    for k, kind in grids:
        tau = tau_grid_np(k, kind)
        rec = np.zeros_like(y)
        rec[ok] = knots_to_mass_np(hist_to_knots(mass[ok], tau), tau) * area[ok, None] / DELTA_SIGMA
        m = metrics(y, rec, w_bins)
        print(f"    K={k:<4} {kind:11} " + " ".join(f"{m[c]:12.6f}" for c in ("weighted_r2", "weighted_mae", "polar_mae", "emd_raw")))
    print("    для сравнения, лучшая модель sigma_dn_geom на test: R2 0.9159, wMAE 0.1066, polar 0.0141, EMD 0.00061")


def torch_checks(config):
    import torch
    from src.physics_extension.sigma_quantile import SigmaQuantileHead, quantiles_to_bin_mass, tau_grid
    print("\n[2] проверка torch-кода")
    torch.manual_seed(0)
    head = SigmaQuantileHead(118, use_geom_area=True, n_knots=64)
    with torch.no_grad():
        head.gaps[-1].weight.normal_(0, 0.5)                    # не нулевая инициализация - иначе тест тривиален
    h = torch.randn(500, 118, requires_grad=True)
    sasa = torch.rand(500) * 30 + 0.1
    out = head(h, sasa, torch.zeros(500, dtype=torch.long))
    q = head.knots(h)
    p = quantiles_to_bin_mass(q, head.dtau, head.edges)
    tau = tau_grid(64, "cos").numpy()
    p_np = knots_to_mass_np(q.detach().double().numpy(), tau)
    print(f"    выход {tuple(out.shape)}; min массы {p.min().item():.2e}; |sum-1| max {(p.sum(1) - 1).abs().max().item():.2e}; "
          f"монотонность Q: {bool((q[:, 1:] > q[:, :-1]).all())}; диапазон Q [{q.min().item():.4f}, {q.max().item():.4f}]")
    print(f"    torch vs numpy: max |Δ| = {np.abs(p.detach().double().numpy() - p_np).max():.2e}")
    out.pow(2).mean().backward()
    print(f"    градиенты конечны: {bool(torch.isfinite(h.grad).all())}, |grad| = {h.grad.norm().item():.3e}")
    if config:
        from src.config import load_config
        from src.train_3d_physics import build_model
        cfg = load_config(config)
        m, name = build_model(cfg)
        print(f"    build_model({config}) -> {type(m).__name__} '{name}', голова {type(m.head).__name__}, "
              f"параметров {sum(x.numel() for x in m.parameters() if x.requires_grad):,} (sigma_dn_geom: 719,820)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n-atoms", type=int, default=30000)
    ap.add_argument("--config", default="configs/3d/sigma_dn_geom_qf.yaml")
    ap.add_argument("--skip-torch", action="store_true")
    a = ap.parse_args()
    ceiling(a.parquet, a.n_atoms, [(32, "cos"), (64, "uniform"), (64, "cos"), (128, "cos")])
    if not a.skip_torch:
        torch_checks(a.config)


if __name__ == "__main__":
    main()
