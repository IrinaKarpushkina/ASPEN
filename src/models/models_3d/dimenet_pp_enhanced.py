from __future__ import annotations
import torch.nn as nn
from .dimenet_pp_physics_common import DimeNetPPEncoder
from ..layers import ResidualMLP


class DimeNetPPEnhanced(nn.Module):
    """DimeNet++ + richer invariant atom/local-environment features."""
    def __init__(self, input_dim=21, hidden_channels=118, dropout=0.05):
        super().__init__()
        self.encoder = DimeNetPPEncoder(input_dim, hidden_channels, dropout=dropout)
        self.head = ResidualMLP(hidden_channels, 51, dropout=dropout)

    def forward(self, data):
        import torch.nn.functional as F
        return F.softplus(self.head(self.encoder(data)))
