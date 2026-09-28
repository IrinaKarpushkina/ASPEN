"""'Kitchen sink' model: everything from this thread's experiments combined,
switchable per-block so it can be ablated down later (see README_KITCHEN.md).

    encoder (DimeNet++ or PaiNN, local 3D geometry, already has angle/triplet
             many-body messages)
      -> bond-graph branch (RDKit chemistry: hybridisation, formal charge,
         aromaticity, ring, bond order/conjugation - NOT derivable from
         coordinates, the one addition in this file backed by genuinely new
         information rather than a redundant geometric feature)
      -> global context (dense attention over the whole molecule, no 5A cutoff)
      -> element-prior residual (optional; train-only per-element mean shape,
         calibrate()'d, NOT enabled by default - see class docstring)
      -> physical head: SASA-informed area, bisection moment projection
         (physics_stable.py), optional exact neutrality projection
         (OFF by default: your own ceiling_analysis run showed molecular
         neutrality does NOT hold in this dataset - see conversation)

Requires features: kitchen (KitchenChaosParquet3DDataset) for the bond graph
and chemical node features; still works with features: sasa if you want to
disable the bond branch (pass use_bond_graph: false), since sigma_global.py's
model already covers that configuration.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..data.constants import DELTA_SIGMA
from .backbones.painn_sigma import IsolatedPaiNNSigmaBackbone
from .bond_branch import BondGraphBranch
from .models import DimeNetPPExperimentBackbone
from .physics import SIGMA_MAX
from .physics_stable import moment_project_bisect
from .sigma_global import GlobalContext

logger = logging.getLogger(__name__)


class ElementPriorResidual(nn.Module):
    """logits += strength * train-only per-element mean log-shape. Uses the
    SAME stable/floored construction as dimenet_pp_delta_fixed's `stable` +
    `prior_floor` options (physics_stable.py / patch_delta_and_loss_v2.py),
    not the original unfloored prior that made dimenet_pp_delta diverge."""

    def __init__(self, n_bins: int = 51, floor: float = 1e-3):
        super().__init__()
        self.register_buffer("prior_logp", torch.zeros(1, n_bins))  # filled by calibrate()
        self.strength = nn.Parameter(torch.tensor(0.3))  # small default, not 1.0
        self.floor = floor
        self._fitted = False

    @torch.no_grad()
    def calibrate(self, dataset, z_list):
        n_bins = self.prior_logp.shape[1]
        max_z = 100
        sums = torch.zeros(max_z, n_bins)
        counts = torch.zeros(max_z)
        for d in dataset._data_list:
            y = d.y.clamp_min(0)
            for i in range(d.num_nodes):
                z = int(d.z[i])
                if z < max_z:
                    sums[z] += y[i]
                    counts[z] += 1
        table = torch.zeros(max_z, n_bins)
        global_mean = (sums.sum(0) / counts.sum().clamp_min(1)).clamp_min(1e-8)
        for z in range(max_z):
            p = sums[z] / counts[z] if counts[z] > 5 else global_mean
            p = p / p.sum().clamp_min(1e-8)
            p = (p + self.floor) / (1.0 + n_bins * self.floor)
            table[z] = torch.log(p)
        self.register_buffer("prior_table", table)
        self._fitted = True
        logger.info("ElementPriorResidual: calibrated on %d elements (min count seen: %d)",
                   int((counts > 0).sum()), int(counts[counts > 0].min()) if (counts > 0).any() else 0)

    def prior_for(self, z):
        if not self._fitted:
            raise RuntimeError("ElementPriorResidual.calibrate() was not called")
        return self.prior_table.to(z.device)[z]


class SigmaKitchenSink(nn.Module):
    def __init__(self, input_dim: int = 33, hidden_channels: int = 118, dropout: float = 0.05,
                 backbone: str = "dimenet", use_bond_graph: bool = True, bond_layers: int = 2,
                 use_global: bool = True, global_layers: int = 2, global_heads: int = 2,
                 use_element_prior: bool = False, use_geom_area: bool = True, neutral: bool = False,
                 **kwargs):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "dimenet":
            self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        elif backbone == "painn":
            self.core = IsolatedPaiNNSigmaBackbone(hidden=hidden_channels, num_layers=kwargs.get("num_layers", 4),
                                                   out_dim=51, dropout=dropout, input_dim=input_dim)
            self.core.readout = nn.Identity()
        else:
            raise ValueError(f"unknown backbone {backbone}")

        self.use_bond_graph = use_bond_graph
        self.bond_branch = BondGraphBranch(hidden_channels, layers=bond_layers, dropout=dropout) \
            if use_bond_graph else None

        self.use_global = use_global
        self.context = GlobalContext(hidden_channels, global_layers, global_heads, dropout) \
            if use_global else None

        self.use_element_prior = use_element_prior
        self.prior = ElementPriorResidual() if use_element_prior else None

        self.use_geom_area, self.neutral = use_geom_area, neutral
        self.shape = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels),
                                   nn.SiLU(), nn.Linear(hidden_channels, 51))
        self.area = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels // 2),
                                  nn.SiLU(), nn.Linear(hidden_channels // 2, 1))
        self.mean = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels // 2),
                                  nn.SiLU(), nn.Linear(hidden_channels // 2, 1))
        nn.init.zeros_(self.shape[-1].weight); nn.init.zeros_(self.shape[-1].bias)
        nn.init.zeros_(self.area[-1].weight); nn.init.zeros_(self.area[-1].bias)
        self.log_area_bias = nn.Parameter(torch.zeros(()))
        self.log_area_scale = nn.Parameter(torch.tensor(1.0 if use_geom_area else 0.0),
                                           requires_grad=use_geom_area)

    def _encode(self, data):
        if self.backbone_name == "dimenet":
            return self.encoder(data)
        return self.core.encode(data, stable_norm=True)

    @torch.no_grad()
    def calibrate(self, dataset):
        vals = []
        for d in dataset._data_list:
            a = d.y.clamp_min(0).sum(dim=1) * DELTA_SIGMA
            ok = a > 1e-6
            lr = torch.log(a.clamp_min(1e-8))
            if self.use_geom_area:
                if not hasattr(d, "sasa_ref"):
                    raise RuntimeError("use_geom_area=True needs features: sasa or kitchen (data.sasa_ref missing)")
                lr = lr - torch.log(d.sasa_ref.clamp_min(1e-3))
            vals.append(lr[ok])
        b = torch.cat(vals).median()
        self.log_area_bias.fill_(float(b))
        if self.use_element_prior:
            self.prior.calibrate(dataset, None)
        logger.info("calibrate: log_area_bias = %.4f", float(b))

    def forward(self, data):
        h = self._encode(data)
        if self.use_bond_graph:
            if not hasattr(data, "bond_edge_index"):
                raise RuntimeError("use_bond_graph=True needs features: kitchen (data.bond_edge_index missing)")
            h = self.bond_branch(h, data.bond_edge_index, data.bond_edge_attr, h.shape[0])
        if self.use_global:
            h = self.context(h, data.pos, data.batch)

        h32 = h.float()
        logits = self.shape(h32)
        if self.use_element_prior:
            logits = logits + self.prior.strength * self.prior.prior_for(data.z)
        log_area = self.log_area_bias + self.area(h32).squeeze(-1)
        if self.use_geom_area:
            if not hasattr(data, "sasa_ref"):
                raise RuntimeError("use_geom_area=True needs features: sasa or kitchen")
            log_area = log_area + self.log_area_scale * torch.log(data.sasa_ref.float().clamp_min(1e-3))
        area = torch.exp(log_area.clamp(-20.0, 10.0))
        mean_sigma = SIGMA_MAX * torch.tanh(self.mean(h32).squeeze(-1))
        if self.neutral:
            from torch_geometric.utils import scatter
            n_mol = int(data.batch.max().item()) + 1
            Q = scatter(area * mean_sigma, data.batch, dim=0, dim_size=n_mol, reduce="sum")
            A = scatter(area, data.batch, dim=0, dim_size=n_mol, reduce="sum")
            mean_sigma = mean_sigma - (Q / A.clamp_min(1e-8))[data.batch]
        p = moment_project_bisect(logits, (mean_sigma / SIGMA_MAX).clamp(-0.98, 0.98))
        return area.unsqueeze(-1) * p / DELTA_SIGMA
