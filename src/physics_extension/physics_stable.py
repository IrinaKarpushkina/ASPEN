"""Opt-in, numerically safe variants of the physics head (does NOT touch physics.py).

Why this exists (checked on synthetic logits in numpy, see README_V2.md):
  * The original `_moment_project` runs 8 UN-damped Newton steps on the tilt
    parameter lambda with clamp(+-25). With a strongly peaked shape (e.g. an
    element prior whose empty bins sit at log(1e-12) = -27.6) it either stalls at the
    clamp (first moment NOT matched, error ~0.2 of sigma_max) or oscillates between
    +-25 (error ~1.4). Backprop then goes through 8 unrolled unstable steps.
  * The mean of a softmax-tilted distribution is monotone in lambda, so lambda can be
    found by BISECTION (always converges, no gradients needed) and the gradient taken
    with one differentiable Newton step (implicit-function gradient).
  * The area branch starts at softplus(0)=0.69 while the target area in the head's own
    units (sum(y)*delta_sigma) is ~0.008 -> initial profile loss ~7000. `calibrate_area`
    sets the bias from TRAIN targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.constants import DELTA_SIGMA
from .physics import SIGMA, SIGMA_MAX, PhysicsSigmaHead


def moment_project_bisect(logits: torch.Tensor, target_mean: torch.Tensor,
                          n_bisect: int = 32, lam_max: float = 200.0, damp: float = 1e-2):
    logits = logits.float()
    s = SIGMA.to(device=logits.device, dtype=logits.dtype) / SIGMA_MAX
    target = target_mean.float().clamp(-0.98, 0.98)
    with torch.no_grad():
        lg = logits.detach()
        lo = torch.full_like(target, -lam_max)
        hi = torch.full_like(target, lam_max)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            m = (torch.softmax(lg + mid.unsqueeze(-1) * s, dim=-1) * s).sum(-1)
            up = m < target.detach()
            lo = torch.where(up, mid, lo)
            hi = torch.where(up, hi, mid)
        lam = 0.5 * (lo + hi)
    p = torch.softmax(logits + lam.unsqueeze(-1) * s, dim=-1)       # lam constant in the graph
    m = (p * s).sum(-1)
    var = (p * (s - m.unsqueeze(-1)).square()).sum(-1)
    lam = lam + (target - m) / (var + damp)                            # ~0 in value, correct gradient
    return torch.softmax(logits + lam.unsqueeze(-1) * s, dim=-1)


class StablePhysicsSigmaHead(PhysicsSigmaHead):
    """Same parameters/keys as PhysicsSigmaHead (checkpoints are interchangeable)."""

    def forward(self, h: torch.Tensor, prior_logits: torch.Tensor | None = None):
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            logits = self.shape(h)
            if prior_logits is not None:
                logits = logits + self.prior_strength * prior_logits.float()
            area = F.softplus(self.area(h).squeeze(-1)) + 1e-7
            mean_sigma = SIGMA_MAX * torch.tanh(self.mean(h).squeeze(-1))
            p = moment_project_bisect(logits, mean_sigma / SIGMA_MAX)
            return area.unsqueeze(-1) * p / DELTA_SIGMA

    @torch.no_grad()
    def calibrate_area(self, dataset):
        vals = []
        for d in dataset._data_list:
            a = d.y.clamp_min(0).sum(dim=1) * DELTA_SIGMA
            vals.append(a[a > 1e-9])
        a0 = torch.cat(vals).median().clamp_min(1e-6)
        b = torch.log(torch.expm1(a0))                                # inverse softplus
        last = self.area[-1]
        last.weight.mul_(0.1)
        last.bias.fill_(float(b))
