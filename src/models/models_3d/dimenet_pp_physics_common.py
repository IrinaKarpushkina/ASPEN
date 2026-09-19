"""Reusable wrappers around the benchmark's DimeNet++ implementation."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
from ...physics_extension.physics import PhysicsSigmaHead


class _CaptureReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, x):
        self.last = x
        return x


class DimeNetPPEncoder(nn.Module):
    """DimeNet++ encoder reusing the benchmark implementation exactly."""

    def __init__(self, input_dim: int, hidden_channels: int = 118,
                 num_blocks: int = 3, dropout: float = 0.05):
        super().__init__()
        from .dimenet import DimeNetSigmaModel
        self.core = DimeNetSigmaModel(
            hidden_channels=hidden_channels,
            out_emb_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=32,
            basis_emb_size=8,
            num_spherical=7,
            num_radial=6,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=2,
            out_dim=51,
            dropout=dropout,
            pp=True,
        )
        self.core.input_proj = nn.Linear(
            input_dim + hidden_channels // 4, hidden_channels
        )
        self.capture = _CaptureReadout()
        self.core.readout = self.capture
        self.hidden = hidden_channels

    def forward(self, data):
        _ = self.core(data)
        return self.capture.last


class DimeNetPPEnhancedPhysics(nn.Module):
    def __init__(self, input_dim=21, hidden_channels=118, dropout=0.05):
        super().__init__()
        self.encoder = DimeNetPPEncoder(input_dim, hidden_channels, dropout=dropout)
        self.head = PhysicsSigmaHead(hidden_channels, 51)

    def forward(self, data):
        h = self.encoder(data)
        return self.head(h)


class DimeNetPPGlobalPhysics(nn.Module):
    """DimeNet++ + molecular context FiLM.

    The local DimeNet representation is supplemented with a pooled molecular
    context. This is deliberately a post-DimeNet context block: it does not
    alter the benchmark's geometric message passing.
    """
    def __init__(self, input_dim=21, hidden_channels=118, dropout=0.05):
        super().__init__()
        self.encoder = DimeNetPPEncoder(input_dim, hidden_channels, dropout=dropout)
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
    """Element-specific sigma-profile prior + learned molecular residual."""

    def __init__(self, input_dim=21, hidden_channels=118,
                 dropout=0.05, prior_path=None):
        super().__init__()
        if prior_path is None:
            raise ValueError("DimeNetPPDeltaPhysics requires prior_path")
        prior = __import__("numpy").load(prior_path)["prior_logp"]
        self.register_buffer("prior_logp", torch.tensor(prior, dtype=torch.float32))
        self.encoder = DimeNetPPEncoder(input_dim, hidden_channels, dropout=dropout)
        self.head = PhysicsSigmaHead(hidden_channels, 51)

    def forward(self, data):
        h = self.encoder(data)
        prior_logits = self.prior_logp[data.z.clamp_min(0).clamp_max(self.prior_logp.size(0)-1)]
        return self.head(h, prior_logits=prior_logits)
