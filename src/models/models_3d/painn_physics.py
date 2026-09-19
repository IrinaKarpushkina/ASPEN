from __future__ import annotations
import torch
import torch.nn as nn
from ...physics_extension.physics import PhysicsSigmaHead


class _CaptureReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None
    def forward(self, x):
        self.last = x
        return x


class PaiNNPhysicsModel(nn.Module):
    """Existing equivariant PaiNN backbone + the same physical sigma head."""
    def __init__(self, input_dim=21, hidden=192, num_layers=4, dropout=0.05):
        super().__init__()
        from .painn import PaiNNSigmaModel
        self.core = PaiNNSigmaModel(
            hidden=hidden, num_layers=num_layers, out_dim=51, dropout=dropout
        )
        self.core.input_proj = nn.Linear(
            input_dim + hidden // 4, hidden
        )
        self.capture = _CaptureReadout()
        self.core.readout = self.capture
        self.head = PhysicsSigmaHead(hidden, 51)

    def forward(self, data):
        _ = self.core(data)
        return self.head(self.capture.last)
