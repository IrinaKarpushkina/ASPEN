"""Physics-informed decoder and loss for the unchanged 51-bin target."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.constants import DELTA_SIGMA, EPS, POLAR_MASK, SIGMA_BINS

SIGMA_MAX = float(max(abs(SIGMA_BINS[0]), abs(SIGMA_BINS[-1])))
SIGMA = torch.tensor(SIGMA_BINS, dtype=torch.float32)


def _moment_project(logits: torch.Tensor, target_mean: torch.Tensor, n_iter: int = 8):
    """Exponential-family projection onto a requested normalized first moment."""
    s = SIGMA.to(device=logits.device, dtype=logits.dtype) / SIGMA_MAX
    target = target_mean.clamp(-0.98, 0.98)
    lam = torch.zeros_like(target)

    for _ in range(n_iter):
        p = torch.softmax(logits + lam.unsqueeze(-1) * s, dim=-1)
        mean = (p * s).sum(dim=-1)
        var = (p * (s - mean.unsqueeze(-1)).square()).sum(dim=-1)
        lam = (lam + (target - mean) / (var + 1e-4)).clamp(-25.0, 25.0)

    return torch.softmax(logits + lam.unsqueeze(-1) * s, dim=-1)


class PhysicsSigmaHead(nn.Module):
    """Positive 51-bin profile with exact predicted-area normalization."""

    def __init__(self, hidden: int, out_dim: int = 51, prior_strength: float = 1.0):
        super().__init__()
        if out_dim != 51:
            raise ValueError("The physics decoder is defined for the fixed 51-bin target.")
        self.prior_strength = float(prior_strength)
        self.shape = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.area = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(hidden // 2, 1), 1),
        )
        self.mean = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    def forward(self, h: torch.Tensor, prior_logits: torch.Tensor | None = None):
        logits = self.shape(h)
        if prior_logits is not None:
            logits = logits + self.prior_strength * prior_logits

        area = F.softplus(self.area(h).squeeze(-1)) + 1e-7
        mean_sigma = SIGMA_MAX * torch.tanh(self.mean(h).squeeze(-1))
        p = _moment_project(logits, mean_sigma / SIGMA_MAX)

        # p is a discrete probability vector; multiplying by 1/delta_sigma
        # turns it into the same profile-density convention as the target.
        return area.unsqueeze(-1) * p / DELTA_SIGMA


class PhysicsSigmaLoss(nn.Module):
    """Dimensionless auxiliary physics losses plus the original profile error.

    All quantities are derived directly from the same 51-bin target. No new
    labels are introduced. ``profile_scale`` is computed from TRAIN only and
    makes the profile term dimensionless for stable weighting.
    """

    name = "physics"

    def __init__(
        self,
        bin_weights: torch.Tensor,
        profile_scale: float,
        w_profile: float = 1.0,
        w_area: float = 0.25,
        w_charge: float = 0.25,
        w_moments: float = 0.10,
        w_smooth: float = 0.05,
        w_wasserstein: float = 0.25,
        w_polar: float = 0.50,
    ):
        super().__init__()
        self.register_buffer("bin_weights", bin_weights)
        self.profile_scale = float(max(profile_scale, 1e-12))
        self.w_profile = float(w_profile)
        self.w_area = float(w_area)
        self.w_charge = float(w_charge)
        self.w_moments = float(w_moments)
        self.w_smooth = float(w_smooth)
        self.w_wasserstein = float(w_wasserstein)
        self.w_polar = float(w_polar)

    @staticmethod
    def _relative_abs(a: torch.Tensor, b: torch.Tensor, floor: float = 1e-6):
        scale = b.detach().abs().mean().clamp_min(floor)
        return (a - b).abs().mean() / scale

    def components(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        sigma = SIGMA.to(device=preds.device, dtype=preds.dtype)
        p = preds.clamp_min(0.0)
        t = targets.clamp_min(0.0)

        profile = ((preds - targets).square() * self.bin_weights).sum(dim=1).mean()
        profile = profile / self.profile_scale

        area_p = p.sum(dim=1) * DELTA_SIGMA
        area_t = t.sum(dim=1) * DELTA_SIGMA
        area = self._relative_abs(area_p, area_t)

        p_mass = p / p.sum(dim=1, keepdim=True).clamp_min(EPS)
        t_mass = t / t.sum(dim=1, keepdim=True).clamp_min(EPS)

        s_norm = sigma / SIGMA_MAX
        mean_p = (p_mass * s_norm).sum(dim=1)
        mean_t = (t_mass * s_norm).sum(dim=1)
        charge = (mean_p - mean_t).abs().mean()

        moments = preds.new_zeros(())
        for k in (2, 3, 4):
            mp = (p_mass * s_norm.pow(k)).sum(dim=1)
            mt = (t_mass * s_norm.pow(k)).sum(dim=1)
            moments = moments + (mp - mt).abs().mean()

        d2p = p_mass[:, 2:] - 2.0 * p_mass[:, 1:-1] + p_mass[:, :-2]
        d2t = t_mass[:, 2:] - 2.0 * t_mass[:, 1:-1] + t_mass[:, :-2]
        smooth = (d2p - d2t).abs().mean()

        cdf_p = torch.cumsum(p_mass, dim=1)
        cdf_t = torch.cumsum(t_mass, dim=1)
        wasserstein = ((cdf_p - cdf_t).abs() * DELTA_SIGMA).sum(dim=1).mean()
        wasserstein = wasserstein / (2.0 * SIGMA_MAX)

        polar_mask = torch.as_tensor(POLAR_MASK, device=preds.device, dtype=preds.dtype)
        polar_raw = ((p - t).abs() * polar_mask).mean()
        polar_scale = (t.detach().abs() * polar_mask).mean().clamp_min(1e-6)
        polar = polar_raw / polar_scale

        return {
            "profile": profile,
            "area": area,
            "charge": charge,
            "moments": moments,
            "smooth": smooth,
            "wasserstein": wasserstein,
            "polar": polar,
        }

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        c = self.components(preds, targets)
        return (
            self.w_profile * c["profile"]
            + self.w_area * c["area"]
            + self.w_charge * c["charge"]
            + self.w_moments * c["moments"]
            + self.w_smooth * c["smooth"]
            + self.w_wasserstein * c["wasserstein"]
            + self.w_polar * c["polar"]
        )
