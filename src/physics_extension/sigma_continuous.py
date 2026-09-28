"""Continuous DimeNet++ + functional decoder, physical head kept intact.

    ContinuousDimeNetPPBackbone (Neural ODE, weight-tied interaction/output
    blocks, integrated over t in [0, T])
        -> FunctionalShapeDecoder (Fourier features of sigma + shared MLP,
           queried at the 51-bin grid by default - see functional_decoder.py's
           caveat about what "continuous" does and does not buy you here)
        -> physics_stable.moment_project_bisect (exact positivity, area,
           first moment - unchanged from sigma_kitchen_sink.py)

Deliberately does NOT include the bond graph / global-attention / element-prior
blocks from sigma_kitchen_sink.py, so that an ablation can isolate the ODE +
functional-decoder effect from those (see README_CONTINUOUS.md). Combine them
later if this shows a real effect on its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.constants import DELTA_SIGMA
from .continuous_dimenet import ContinuousDimeNetPPBackbone
from .functional_decoder import FunctionalShapeDecoder
from .physics import SIGMA_MAX
from .physics_stable import moment_project_bisect


class SigmaContinuousDimeNet(nn.Module):
    def __init__(self, input_dim: int = 25, hidden_channels: int = 118, dropout: float = 0.05,
                use_geom_area: bool = True, T: float = 1.0, n_steps: int = 6,
                use_torchdiffeq: bool = False, solver: str = "dopri5",
                n_fourier: int = 8, **kwargs):
        super().__init__()
        self.encoder = ContinuousDimeNetPPBackbone(
            input_dim, hidden_channels, dropout=dropout, T=T, n_steps=n_steps,
            use_torchdiffeq=use_torchdiffeq, solver=solver, **kwargs)
        self.decoder = FunctionalShapeDecoder(hidden_channels, n_freqs=n_fourier)
        self.mean = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels // 2),
                                  nn.SiLU(), nn.Linear(hidden_channels // 2, 1))
        self.area = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels // 2),
                                  nn.SiLU(), nn.Linear(hidden_channels // 2, 1))
        nn.init.zeros_(self.area[-1].weight); nn.init.zeros_(self.area[-1].bias)
        self.use_geom_area = use_geom_area
        self.log_area_bias = nn.Parameter(torch.zeros(()))
        self.log_area_scale = nn.Parameter(torch.tensor(1.0 if use_geom_area else 0.0),
                                           requires_grad=use_geom_area)

    @torch.no_grad()
    def calibrate(self, dataset):
        vals = []
        for d in dataset._data_list:
            a = d.y.clamp_min(0).sum(dim=1) * DELTA_SIGMA
            ok = a > 1e-6
            lr = torch.log(a.clamp_min(1e-8))
            if self.use_geom_area:
                if not hasattr(d, "sasa_ref"):
                    raise RuntimeError("use_geom_area=True needs features: sasa (data.sasa_ref missing)")
                lr = lr - torch.log(d.sasa_ref.clamp_min(1e-3))
            vals.append(lr[ok])
        b = torch.cat(vals).median()
        self.log_area_bias.fill_(float(b))

    def forward(self, data):
        h = self.encoder.encode(data)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h32 = h.float()
            logits = self.decoder(h32)                              # (N, 51), raw shape
            log_area = self.log_area_bias + self.area(h32).squeeze(-1)
            if self.use_geom_area:
                if not hasattr(data, "sasa_ref"):
                    raise RuntimeError("use_geom_area=True needs features: sasa")
                log_area = log_area + self.log_area_scale * torch.log(data.sasa_ref.float().clamp_min(1e-3))
            area = torch.exp(log_area.clamp(-20.0, 10.0))
            mean_sigma = SIGMA_MAX * torch.tanh(self.mean(h32).squeeze(-1))
            p = moment_project_bisect(logits, (mean_sigma / SIGMA_MAX).clamp(-0.98, 0.98))
            return area.unsqueeze(-1) * p / DELTA_SIGMA
