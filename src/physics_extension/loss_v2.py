"""PhysicsSigmaLoss + extra terms aligned with the evaluation metrics.

With every new option at its default the value equals PhysicsSigmaLoss exactly.

New options (yaml `loss:` block, name: physics_v2):
  profile_loss   "mse" | "l1"      l1 = variance-weighted MAE (the checkpoint-selection metric is wMAE)
  moments_mode   "raw" | "relative" 'relative' divides every moment/smoothness error by the target's own
                                    magnitude. In 'raw' mode the k=3,4 moments and the 2nd-difference term
                                    have tiny gradients (|s|<=1, s^4 ~ 0), so w_moments=0.10 / w_smooth=0.05
                                    are nearly inert; 'relative' makes their weights meaningful.
  w_cos          1 - cosine(pred, target) per atom (cosine_similarity metric)
  w_mol          relative L1 of the MOLECULAR profile (sum of atomic profiles) -> molecular_mae
  w_mol_cdf      relative area-aware CDF error of the molecular profile -> molecular_emd
Molecular terms need the batch vector; the trainer patch passes it (`needs_batch = True`).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.constants import EPS
from .physics import SIGMA, SIGMA_MAX, PhysicsSigmaLoss


class PhysicsSigmaLossV2(PhysicsSigmaLoss):
    name = "physics_v2"
    needs_batch = True

    def __init__(self, bin_weights, profile_scale, profile_loss: str = "mse",
                 moments_mode: str = "raw", w_cos: float = 0.0, w_mol: float = 0.0,
                 w_mol_cdf: float = 0.0, **kwargs):
        super().__init__(bin_weights, profile_scale, **kwargs)
        if profile_loss not in ("mse", "l1"):
            raise ValueError("profile_loss must be 'mse' or 'l1'")
        if moments_mode not in ("raw", "relative"):
            raise ValueError("moments_mode must be 'raw' or 'relative'")
        self.profile_loss, self.moments_mode = profile_loss, moments_mode
        self.w_cos, self.w_mol, self.w_mol_cdf = float(w_cos), float(w_mol), float(w_mol_cdf)

    def components(self, preds, targets, batch=None):
        preds, targets = preds.float(), targets.float()
        c = super().components(preds, targets)
        p, t = preds.clamp_min(0.0), targets.clamp_min(0.0)

        if self.profile_loss == "l1":
            c["profile"] = ((preds - targets).abs() * self.bin_weights).sum(dim=1).mean() / (self.profile_scale ** 0.5)

        if self.moments_mode == "relative":
            s = SIGMA.to(device=preds.device, dtype=preds.dtype) / SIGMA_MAX
            pm = p / p.sum(dim=1, keepdim=True).clamp_min(EPS)
            tm = t / t.sum(dim=1, keepdim=True).clamp_min(EPS)
            mom = preds.new_zeros(())
            for k in (2, 3, 4):
                mp, mt = (pm * s.pow(k)).sum(1), (tm * s.pow(k)).sum(1)
                mom = mom + (mp - mt).abs().mean() / mt.detach().abs().mean().clamp_min(1e-3)
            d2p = pm[:, 2:] - 2 * pm[:, 1:-1] + pm[:, :-2]
            d2t = tm[:, 2:] - 2 * tm[:, 1:-1] + tm[:, :-2]
            c["moments"] = mom
            c["smooth"] = (d2p - d2t).abs().mean() / d2t.detach().abs().mean().clamp_min(1e-6)

        c["cos"] = 1.0 - F.cosine_similarity(p, t, dim=1, eps=1e-8).mean()

        zero = preds.new_zeros(())
        c["mol"], c["mol_cdf"] = zero, zero
        if batch is not None and (self.w_mol > 0 or self.w_mol_cdf > 0):
            n = int(batch.max().item()) + 1
            Mp = preds.new_zeros(n, preds.shape[1]).index_add_(0, batch, p)
            Mt = preds.new_zeros(n, preds.shape[1]).index_add_(0, batch, t)
            tot = Mt.detach().sum(dim=1).clamp_min(EPS)
            c["mol"] = ((Mp - Mt).abs().sum(dim=1) / tot).mean()
            c["mol_cdf"] = ((Mp.cumsum(1) - Mt.cumsum(1)).abs().sum(dim=1) / (tot * Mp.shape[1])).mean()
        return c

    def forward(self, preds, targets, batch=None):
        c = self.components(preds, targets, batch)
        return (self.w_profile * c["profile"] + self.w_area * c["area"] + self.w_charge * c["charge"]
                + self.w_moments * c["moments"] + self.w_smooth * c["smooth"]
                + self.w_wasserstein * c["wasserstein"] + self.w_polar * c["polar"]
                + self.w_cos * c["cos"] + self.w_mol * c["mol"] + self.w_mol_cdf * c["mol_cdf"])
