"""Functional decoder / Neural Field for the sigma-profile SHAPE.

Replaces a single Linear(hidden, 51) with a shared MLP queried at each sigma
value: p_i(sigma) = MLP([h_i, gamma(sigma)]), gamma = Fourier features. This
can be queried at ANY sigma in [-0.025, 0.025], not just the 51 training bins.

IMPORTANT CAVEAT NOT IN THE ORIGINAL PROPOSAL: your training targets are
themselves already binned onto exactly these 51 points at data-prep time by
linear interpolation (sp_create_right_area.py's bin_sigma_profile). There is
no finer-grained ground truth anywhere in the pipeline. Querying this decoder
between the 51 grid points is therefore an INTERPOLATION driven by the MLP's
own smoothness prior, not a recovery of information the data actually
contains. The realistic benefit is a smoothness inductive bias on the shape
(may help the polar tails, where the profile changes fastest), not
"eliminating discretisation artifacts" in the sense of resolving structure
finer than 0.001 sigma that your data doesn't have.

Kept separate from the physical (area/mean/positivity) guarantees: this module
only returns raw, UNNORMALISED shape logits over the query grid. Positivity,
exact area and exact first moment are still enforced by physics_stable.py's
softplus/bisection machinery downstream (see sigma_continuous.py), not by this
file - a plain Softplus-only decoder (as in the original proposal) would drop
those guarantees, which the rest of this project has been built around.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .physics import SIGMA, SIGMA_MAX


class FourierFeatures(nn.Module):
    def __init__(self, n_freqs: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.freqs.numel()

    def forward(self, sigma_norm: torch.Tensor) -> torch.Tensor:
        """sigma_norm: (...,) in [-1, 1]. Returns (..., 2*n_freqs)."""
        arg = sigma_norm.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class FunctionalShapeDecoder(nn.Module):
    def __init__(self, hidden: int, n_freqs: int = 8, mlp_hidden: int = 64):
        super().__init__()
        self.fourier = FourierFeatures(n_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + self.fourier.out_dim, mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("grid_norm", SIGMA.clone() / SIGMA_MAX)  # 51 default query points

    def forward(self, h: torch.Tensor, query_sigma_norm: torch.Tensor | None = None) -> torch.Tensor:
        """h: (N, hidden). query_sigma_norm: (Q,) in [-1, 1], default = the 51-bin grid.
        Returns (N, Q) raw logits (no softplus/softmax applied here)."""
        q = self.grid_norm if query_sigma_norm is None else query_sigma_norm
        gamma = self.fourier(q)                                   # (Q, F)
        N, Q = h.shape[0], q.shape[0]
        h_exp = h.unsqueeze(1).expand(N, Q, h.shape[-1])
        g_exp = gamma.unsqueeze(0).expand(N, Q, gamma.shape[-1])
        return self.mlp(torch.cat([h_exp, g_exp], dim=-1)).squeeze(-1)
