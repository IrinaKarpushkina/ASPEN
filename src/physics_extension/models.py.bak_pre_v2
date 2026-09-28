"""Physics experiments A-F using isolated copies of the benchmark backbones.

IMPORTANT: this module does not import or modify src.models.models_3d.dimenet.py
or src.models.models_3d.painn.py. The already-completed benchmark therefore
remains byte-for-byte independent of these experiments.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

from .backbones.dimenet_sigma import IsolatedDimeNetSigmaBackbone
from .backbones.painn_sigma import IsolatedPaiNNSigmaBackbone
from .physics import PhysicsSigmaHead
from src.models.layers import ResidualMLP


class DimeNetPPExperimentBackbone(nn.Module):
    """Isolated DimeNet++ backbone matching the benchmark configuration."""

    def __init__(self, input_dim: int, hidden_channels: int = 118, dropout: float = 0.05, **kwargs):
        super().__init__()
        self.core = IsolatedDimeNetSigmaBackbone(
            hidden_channels=hidden_channels,
            out_emb_channels=hidden_channels,
            num_blocks=kwargs.get("num_blocks", 3),
            int_emb_size=kwargs.get("int_emb_size", 32),
            basis_emb_size=kwargs.get("basis_emb_size", 8),
            num_spherical=kwargs.get("num_spherical", 7),
            num_radial=kwargs.get("num_radial", 6),
            num_before_skip=kwargs.get("num_before_skip", 1),
            num_after_skip=kwargs.get("num_after_skip", 2),
            num_output_layers=kwargs.get("num_output_layers", 2),
            cutoff=kwargs.get("cutoff", None) or self._default_cutoff(),
            dropout=dropout,
            pp=True,
            input_dim=input_dim,
        )
        # No unused 51-output readout is kept in the backbone.
        self.core.readout = nn.Identity()
        self.hidden = hidden_channels

    @staticmethod
    def _default_cutoff():
        from src.data.constants_3d import CUTOFF
        return CUTOFF

    def forward(self, data):
        return self.core.encode(data)


class DimeNetPPEnhanced(nn.Module):
    """A: enhanced invariant features + ordinary 51-bin regression head."""

    def __init__(self, input_dim=22, hidden_channels=118, dropout=0.05, **kwargs):
        super().__init__()
        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        self.head = ResidualMLP(hidden_channels, 51, dropout=dropout)

    def forward(self, data):
        return F.softplus(self.head(self.encoder(data)))


class DimeNetPPPhysics(nn.Module):
    """B/C: DimeNet++ + physics-constrained decoder."""

    def __init__(self, input_dim=6, hidden_channels=118, dropout=0.05, prior_path=None, prior_strength=1.0, **kwargs):
        super().__init__()
        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        self.head = PhysicsSigmaHead(hidden_channels, 51, prior_strength=prior_strength)
        if prior_path is not None:
            prior = np.load(prior_path)["prior_logp"]
            self.register_buffer("prior_logp", torch.as_tensor(prior, dtype=torch.float32))
        else:
            self.prior_logp = None

    def forward(self, data):
        h = self.encoder(data)
        prior = None
        if self.prior_logp is not None:
            z = data.z.clamp(0, self.prior_logp.shape[0] - 1)
            prior = self.prior_logp[z]
        return self.head(h, prior_logits=prior)


class DimeNetPPGlobalPhysics(nn.Module):
    """D: enhanced features + molecular FiLM context + physics head."""

    def __init__(self, input_dim=22, hidden_channels=118, dropout=0.05, **kwargs):
        super().__init__()
        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        self.context = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )
        self.head = PhysicsSigmaHead(hidden_channels, 51)

    def forward(self, data):
        h = self.encoder(data)
        c = scatter_mean(h, data.batch, dim=0)
        gamma, beta = self.context(c).chunk(2, dim=-1)
        gamma = torch.tanh(gamma[data.batch])
        beta = beta[data.batch]
        h = h * (1.0 + 0.25 * gamma) + 0.10 * beta
        return self.head(h)


class DimeNetPPDeltaPhysics(nn.Module):
    """F: train-only element prior in shape-logit space + physics head."""

    def __init__(self, input_dim=22, hidden_channels=118, dropout=0.05, prior_path=None, prior_strength=1.0, **kwargs):
        super().__init__()
        if not prior_path:
            raise ValueError("Experiment F requires model.prior_path")
        prior = np.load(prior_path)
        prior_arr = prior["prior_logp"]
        if prior_arr.shape != (87, 51):
            raise ValueError(f"Expected prior shape (87, 51), got {prior_arr.shape}")
        self.register_buffer("prior_logp", torch.as_tensor(prior_arr, dtype=torch.float32))
        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        self.head = PhysicsSigmaHead(hidden_channels, 51, prior_strength=prior_strength)

    def forward(self, data):
        h = self.encoder(data)
        z = data.z.clamp(0, self.prior_logp.shape[0] - 1)
        return self.head(h, prior_logits=self.prior_logp[z])


class PaiNNPhysics(nn.Module):
    """E: isolated PaiNN scalar representation + physics head."""

    def __init__(self, input_dim=22, hidden=192, num_layers=4, dropout=0.05, **kwargs):
        super().__init__()
        self.core = IsolatedPaiNNSigmaBackbone(
            hidden=hidden,
            num_layers=num_layers,
            out_dim=51,
            dropout=dropout,
            input_dim=input_dim,
        )
        self.core.readout = nn.Identity()
        self.head = PhysicsSigmaHead(hidden, 51)

    def forward(self, data):
        h = self.core.encode(data, stable_norm=True)
        return self.head(h)
