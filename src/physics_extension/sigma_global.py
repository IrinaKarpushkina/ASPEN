"""DimeNet++/PaiNN encoder + global 3D attention + geometric-area physical head.

Three independent, switchable ideas (each can be ablated from the yaml):

1. ``use_global``  - the radius graph only sees 5 A. Sigma-profiles depend on the
   whole molecular electrostatics, so a small transformer over ALL atoms of a
   molecule (attention bias = learned function of the 3D distance, no cutoff)
   is added on top of the local encoder, as a gated residual (starts ~off).
2. ``use_geom_area`` - area = SASA(geometry) * exp(learned correction). The
   integral of the profile is a geometric quantity; the network only has to
   learn a residual instead of the whole magnitude.
3. ``neutral`` - hard molecular constraint: sum_i area_i * <sigma>_i = 0
   (screening charges of a neutral molecule cancel). Implemented as an exact
   projection that couples all atoms of the molecule. Enable ONLY after
   scripts/ceiling_analysis.py confirms the constraint holds in your data
   (ions / charged species violate it).

The output convention is identical to PhysicsSigmaHead:  area * p / DELTA_SIGMA
with p a probability vector on the 51 bins, so it is positive, has exactly the
predicted area and (after the projection) exactly the requested first moment.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch

from ..data.constants import DELTA_SIGMA
from .backbones.painn_sigma import IsolatedPaiNNSigmaBackbone
from .models import DimeNetPPExperimentBackbone
from .physics import SIGMA_MAX, _moment_project

logger = logging.getLogger(__name__)


class _GlobalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        assert hidden % heads == 0, "hidden_channels must be divisible by global_heads"
        self.heads, self.dh = heads, hidden // heads
        self.n1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.o = nn.Linear(hidden, hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        B, N, H = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        m = bias.float().masked_fill(~key_mask[:, None, None, :], -1e4).to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        a = a.transpose(1, 2).reshape(B, N, H)
        x = x + self.drop(self.o(a))
        return x + self.drop(self.ff(self.n2(x)))


class GlobalContext(nn.Module):
    """Dense per-molecule attention with a distance-dependent bias."""

    def __init__(self, hidden: int, layers: int = 2, heads: int = 4,
                 dropout: float = 0.05, n_rbf: int = 24, r_max: float = 12.0):
        super().__init__()
        self.layers, self.heads = layers, heads
        self.register_buffer("centers", torch.linspace(0.0, r_max, n_rbf))
        self.coef = -0.5 / (r_max / (n_rbf - 1)) ** 2
        self.bias_mlp = nn.Sequential(nn.Linear(n_rbf + 1, 32), nn.SiLU(),
                                      nn.Linear(32, layers * heads))
        self.blocks = nn.ModuleList([_GlobalBlock(hidden, heads, dropout) for _ in range(layers)])
        self.gate = nn.Parameter(torch.full((hidden,), 0.1))

    def forward(self, h, pos, batch):
        x, mask = to_dense_batch(h, batch)
        p, _ = to_dense_batch(pos, batch)
        d = torch.cdist(p.float(), p.float())                                   # (B,N,N)
        feat = torch.cat([torch.exp(self.coef * (d.unsqueeze(-1) - self.centers) ** 2),
                          torch.log1p(d).unsqueeze(-1) / 3.0], dim=-1)
        bias = self.bias_mlp(feat.to(x.dtype)).permute(0, 3, 1, 2)             # (B,L*H,N,N)
        x0 = x
        for l, blk in enumerate(self.blocks):
            x = blk(x, bias[:, l * self.heads:(l + 1) * self.heads], mask)
        delta = (x - x0)[mask]
        return h + self.gate * delta.to(h.dtype)


class SigmaGeomHead(nn.Module):
    def __init__(self, hidden: int, use_geom_area: bool = True, neutral: bool = False):
        super().__init__()
        self.use_geom_area, self.neutral = use_geom_area, neutral
        self.shape = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                   nn.SiLU(), nn.Linear(hidden, 51))
        self.area = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                  nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.mean = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                  nn.SiLU(), nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.shape[-1].weight); nn.init.zeros_(self.shape[-1].bias)
        nn.init.zeros_(self.area[-1].weight); nn.init.zeros_(self.area[-1].bias)
        self.log_area_bias = nn.Parameter(torch.zeros(()))
        self.log_area_scale = nn.Parameter(torch.tensor(1.0 if use_geom_area else 0.0),
                                           requires_grad=use_geom_area)

    def forward(self, h, sasa, batch):
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            logits = self.shape(h)
            log_area = self.log_area_bias + self.area(h).squeeze(-1)
            if self.use_geom_area:
                log_area = log_area + self.log_area_scale * torch.log(sasa.float().clamp_min(1e-3))
            area = torch.exp(log_area.clamp(-20.0, 10.0))
            mean = SIGMA_MAX * torch.tanh(self.mean(h).squeeze(-1))
            if self.neutral:
                n_mol = int(batch.max().item()) + 1
                Q = scatter(area * mean, batch, dim=0, dim_size=n_mol, reduce="sum")
                A = scatter(area, batch, dim=0, dim_size=n_mol, reduce="sum")
                mean = mean - (Q / A.clamp_min(1e-8))[batch]
            p = _moment_project(logits, (mean / SIGMA_MAX).clamp(-0.98, 0.98))
            return area.unsqueeze(-1) * p / DELTA_SIGMA


class DimeNetPPSigmaGlobal(nn.Module):
    """backbone='dimenet' (needs triplets) or 'painn'."""

    def __init__(self, input_dim: int = 25, hidden_channels: int = 118, dropout: float = 0.05,
                 backbone: str = "dimenet", use_global: bool = True, global_layers: int = 2,
                 global_heads: int = 4, use_geom_area: bool = True, neutral: bool = False, **kwargs):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "dimenet":
            self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        elif backbone == "painn":
            self.core = IsolatedPaiNNSigmaBackbone(
                hidden=hidden_channels, num_layers=kwargs.get("num_layers", 4),
                out_dim=51, dropout=dropout, input_dim=input_dim)
            self.core.readout = nn.Identity()
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.use_global = use_global
        self.context = GlobalContext(hidden_channels, global_layers, global_heads, dropout) if use_global else None
        self.head = SigmaGeomHead(hidden_channels, use_geom_area=use_geom_area, neutral=neutral)

    def _encode(self, data):
        if self.backbone_name == "dimenet":
            return self.encoder(data)
        return self.core.encode(data, stable_norm=True)

    @torch.no_grad()
    def calibrate(self, dataset):
        """Set the log-area offset from TRAIN targets so the very first
        predictions already have the right magnitude (initial profile loss was
        ~7000 in the delta run because the magnitude was ~100x off)."""
        from ..data.constants import DELTA_SIGMA as DS
        vals = []
        for d in dataset._data_list:
            a = d.y.clamp_min(0).sum(dim=1) * DS
            ok = a > 1e-6
            lr = torch.log(a.clamp_min(1e-8))
            if self.head.use_geom_area:
                if not hasattr(d, "sasa_ref"):
                    raise RuntimeError("use_geom_area=True needs features: sasa (data.sasa_ref missing)")
                lr = lr - torch.log(d.sasa_ref.clamp_min(1e-3))
            vals.append(lr[ok])
        b = torch.cat(vals).median()
        self.head.log_area_bias.fill_(float(b))
        logger.info("calibrate: log_area_bias = %.4f", float(b))

    def forward(self, data):
        h = self._encode(data)
        if self.context is not None:
            h = self.context(h, data.pos, data.batch)
        sasa = getattr(data, "sasa_ref", None) if self.head.use_geom_area else None
        if self.head.use_geom_area and sasa is None:
            raise RuntimeError("use_geom_area=True needs features: sasa")
        return self.head(h, sasa, data.batch)
