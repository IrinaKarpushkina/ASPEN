"""
combined.py — multi-component loss for the final/best model.

Components and weights (ablation target):
  1.0 × WeightedMAE        — variance-weighted per-bin MAE
  0.5 × IntegralMAE        — conservation of total surface area
  0.5 × PolarMAE           — emphasis on chemically relevant polar region
  0.3 × MomentsLoss        — distribution moments (mean, variance, skew, kurtosis)
  1.0 × WassersteinCDF     — shape of the full distribution
  0.1 × ShannonEntropy     — entropy regularisation

Used ONLY in the ablation study (Uровень 2). Baseline models always use MSELoss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.constants import SIGMA_BINS, DELTA_SIGMA, SIGMA_RANGE, POLAR_MASK, EPS


class WeightedMAELoss(nn.Module):
    name = "weighted_mae"

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, preds, targets):
        return (torch.abs(preds - targets) * self.weights).sum(dim=1).mean()


class IntegralMAELoss(nn.Module):
    name = "integral_mae"

    def forward(self, preds, targets):
        return F.l1_loss(preds.sum(1) * DELTA_SIGMA, targets.sum(1) * DELTA_SIGMA)


class PolarMAELoss(nn.Module):
    name = "polar_mae"

    def __init__(self):
        super().__init__()
        mask = torch.tensor(POLAR_MASK, dtype=torch.float)
        self.register_buffer("mask", mask)

    def forward(self, preds, targets):
        return F.l1_loss(preds * self.mask, targets * self.mask)


class MomentsLoss(nn.Module):
    name = "moments"

    def __init__(self):
        super().__init__()
        sigma = torch.tensor(SIGMA_BINS, dtype=torch.float)
        self.register_buffer("sigma", sigma)

    def forward(self, preds, targets):
        p = preds   / (preds.sum(1, keepdim=True)   + EPS)
        t = targets / (targets.sum(1, keepdim=True) + EPS)
        loss = torch.tensor(0.0, device=preds.device)
        for k in range(1, 5):
            sk    = self.sigma ** k
            mp, mt = (p * sk).sum(1), (t * sk).sum(1)
            scale = torch.clamp(torch.max(mt.abs(), mp.abs()).mean(), min=1e-6)
            loss  = loss + F.l1_loss(mp / scale, mt / scale)
        return loss


class WassersteinCDFLoss(nn.Module):
    name = "wasserstein_cdf"

    def forward(self, preds, targets):
        p   = preds   / (preds.sum(1, keepdim=True)   + EPS)
        t   = targets / (targets.sum(1, keepdim=True) + EPS)
        w1  = (torch.abs(torch.cumsum(p, 1) - torch.cumsum(t, 1)) * DELTA_SIGMA).sum(1).mean()
        return w1 / SIGMA_RANGE


class ShannonEntropyLoss(nn.Module):
    name = "shannon_entropy"

    def forward(self, preds, targets):
        p  = preds   / (preds.sum(1, keepdim=True)   + EPS) + EPS
        t  = targets / (targets.sum(1, keepdim=True) + EPS) + EPS
        return F.l1_loss(-(p * p.log()).sum(1), -(t * t.log()).sum(1))


class CombinedLoss(nn.Module):
    """
    Комбинированный loss для финальной модели и ablation-а.

    Веса подобраны так, чтобы WassersteinCDF и WeightedMAE доминировали
    (наиболее релевантны для σ-профиля), а остальные компоненты служат
    регуляризаторами формы.
    """
    name = "combined"

    def __init__(
        self,
        bin_weights: torch.Tensor,
        w_wmae:    float = 1.0,
        w_integral: float = 0.5,
        w_polar:   float = 0.5,
        w_moments: float = 0.3,
        w_wass:    float = 1.0,
        w_entropy: float = 0.1,
    ):
        super().__init__()
        self.wmae     = WeightedMAELoss(bin_weights)
        self.integral = IntegralMAELoss()
        self.polar    = PolarMAELoss()
        self.moments  = MomentsLoss()
        self.wass     = WassersteinCDFLoss()
        self.entropy  = ShannonEntropyLoss()
        self.w = dict(
            wmae=w_wmae, integral=w_integral, polar=w_polar,
            moments=w_moments, wass=w_wass, entropy=w_entropy
        )

    def forward(self, preds, targets):
        return (
            self.w["wmae"]     * self.wmae(preds, targets)     +
            self.w["integral"] * self.integral(preds, targets) +
            self.w["polar"]    * self.polar(preds, targets)    +
            self.w["moments"]  * self.moments(preds, targets)  +
            self.w["wass"]     * self.wass(preds, targets)     +
            self.w["entropy"]  * self.entropy(preds, targets)
        )
