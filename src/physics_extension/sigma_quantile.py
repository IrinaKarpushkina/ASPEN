"""sigma_dn_*_qf: та же модель, что sigma_dn_geom, но нормированное распределение атома задаётся
КВАНТИЛЬНОЙ ФУНКЦИЕЙ, а не softmax по 51 бину + проекцией первого момента.

Чистая абляция представления: backbone, признаки, голова площади (SASA x поправка), калибровка,
loss и метрики - те же. Выход модели - тот же (n_atoms, 51) профиль, area * p / DELTA_SIGMA,
поэтому train_3d_physics.py, evaluate.py, molecule_level_eval.py, stats_compare.py работают без изменений.

Параметризация (на атом):
  * K отрезков на сетке уровней tau_0=0 < ... < tau_K=1, сгущённой к краям (cos-сетка), чтобы хвосты
    (полярные бины, малая масса) описывались подробно;
  * MLP выдаёт K+2 логитов -> зазоры g = eps + (1-(K+2)eps) softmax(z), сумма 1;
    узлы Q_k = e_lo + R * cumsum(g)_k, k = 0..K. Q монотонна СТРОГО и лежит в [e_lo, e_hi] по построению
    (первый зазор = отступ от левого края сетки, последний = до правого);
  * Q кусочно-линейна по tau => распределение - смесь равномерных на [Q_k, Q_{k+1}] с массами dtau_k;
    масса бина j считается ТОЧНО как пересечение отрезков с бином: p_j = sum_k dtau_k |[Q_k,Q_{k+1}] ∩ bin_j| / (Q_{k+1}-Q_k).
    sum_j p_j = 1 точно, p_j >= 0, градиент по Q определён.
Среднее sigma атома = интеграл Q dtau (отдельная голова среднего не нужна и убрана).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.constants import DELTA_SIGMA, SIGMA_BINS
from .sigma_global import DimeNetPPSigmaGlobal

E_LO = float(SIGMA_BINS[0] - DELTA_SIGMA / 2)
E_HI = float(SIGMA_BINS[-1] + DELTA_SIGMA / 2)


def tau_grid(k: int, kind: str = "cos") -> torch.Tensor:
    i = torch.arange(k + 1, dtype=torch.float64)
    if kind == "cos":
        t = (1 - torch.cos(math.pi * i / k)) / 2
    elif kind == "uniform":
        t = i / k
    else:
        raise ValueError(f"unknown tau grid {kind}")
    t[0], t[-1] = 0.0, 1.0
    return t


def quantiles_to_bin_mass(q: torch.Tensor, dtau: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """q: (N, K+1) монотонные узлы; dtau: (K,); edges: (51+1,). -> (N, 51) массы бинов, сумма 1."""
    a, b = q[:, :-1], q[:, 1:]
    w = (b - a).clamp_min(1e-12)
    lo, hi = edges[:-1], edges[1:]
    ov = (torch.minimum(b.unsqueeze(-1), hi) - torch.maximum(a.unsqueeze(-1), lo)).clamp_min(0.0)   # (N, K, 51)
    return (dtau.view(1, -1, 1) * ov / w.unsqueeze(-1)).sum(1)


class SigmaQuantileHead(nn.Module):
    """Те же атрибуты use_geom_area / log_area_bias / log_area_scale, что у SigmaGeomHead (их использует calibrate)."""

    def __init__(self, hidden: int, use_geom_area: bool = True, n_knots: int = 64,
                 tau_kind: str = "cos", gap_eps: float = 1e-4):
        super().__init__()
        self.use_geom_area, self.neutral = use_geom_area, False
        self.n_knots, self.gap_eps = n_knots, gap_eps
        self.gaps = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.SiLU(), nn.Linear(hidden, n_knots + 2))
        self.area = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                  nn.SiLU(), nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.gaps[-1].weight); nn.init.zeros_(self.gaps[-1].bias)
        nn.init.zeros_(self.area[-1].weight); nn.init.zeros_(self.area[-1].bias)
        self.log_area_bias = nn.Parameter(torch.zeros(()))
        self.log_area_scale = nn.Parameter(torch.tensor(1.0 if use_geom_area else 0.0),
                                           requires_grad=use_geom_area)
        t = tau_grid(n_knots, tau_kind).float()
        self.register_buffer("dtau", t[1:] - t[:-1], persistent=False)
        edges = torch.tensor([E_LO] + list((SIGMA_BINS[:-1] + SIGMA_BINS[1:]) / 2) + [E_HI], dtype=torch.float32)
        self.register_buffer("edges", edges, persistent=False)

    def knots(self, h: torch.Tensor) -> torch.Tensor:
        """(N, K+1) узлы квантильной функции, sigma в e/A^2."""
        k2 = self.n_knots + 2
        g = self.gap_eps + (1.0 - k2 * self.gap_eps) * torch.softmax(self.gaps(h), dim=-1)
        c = torch.cumsum(g, dim=-1)[:, : self.n_knots + 1]
        return E_LO + (E_HI - E_LO) * c

    def forward(self, h, sasa, batch):
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            log_area = self.log_area_bias + self.area(h).squeeze(-1)
            if self.use_geom_area:
                log_area = log_area + self.log_area_scale * torch.log(sasa.float().clamp_min(1e-3))
            area = torch.exp(log_area.clamp(-20.0, 10.0))
            p = quantiles_to_bin_mass(self.knots(h), self.dtau, self.edges)
            return area.unsqueeze(-1) * p / DELTA_SIGMA


class DimeNetPPSigmaQuantile(DimeNetPPSigmaGlobal):
    """DimeNetPPSigmaGlobal с квантильной головой. Ключи yaml: n_knots, tau_kind, gap_eps."""

    def __init__(self, *args, n_knots: int = 64, tau_kind: str = "cos", gap_eps: float = 1e-4, **kwargs):
        if kwargs.get("neutral", False):
            raise ValueError("neutral=True не поддерживается квантильной головой")
        super().__init__(*args, **kwargs)
        hidden = kwargs.get("hidden_channels", 118)
        self.head = SigmaQuantileHead(hidden, use_geom_area=kwargs.get("use_geom_area", True),
                                      n_knots=n_knots, tau_kind=tau_kind, gap_eps=gap_eps)
