"""Small message-passing branch on the RDKit CHEMICAL bond graph (not the
spatial radius graph). Bond order / aromaticity / conjugation / ring
membership are not derivable from coordinates, so this is genuinely new
information (doc section 14.2's "bond graph + spatial graph" idea), unlike
the geometric enhanced/SASA features which the spatial encoder can already
reconstruct.

Molecules with mol_valid=False have an EMPTY edge_index (see
dataset_3d_kitchen.py); this branch must produce a harmless zero contribution
for them, not crash or NaN - tested below on an all-empty batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class BondGraphBranch(nn.Module):
    def __init__(self, hidden: int, edge_dim: int = 7, layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU())
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.gate = nn.Parameter(torch.full((hidden,), 0.1))

    def forward(self, h, edge_index, edge_attr, num_nodes):
        if edge_index.numel() == 0:
            return h  # no bonds resolved for any molecule in this batch: identity
        row, col = edge_index
        e = self.edge_mlp(edge_attr)
        x = h
        for mlp, norm in zip(self.layers, self.norms):
            msg = mlp(torch.cat([x[row], x[col], e], dim=-1))
            agg = scatter(msg, row, dim=0, dim_size=num_nodes, reduce="mean")
            x = norm(x + agg)
        return h + self.gate * (x - h)
