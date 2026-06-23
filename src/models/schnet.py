"""
schnet.py — SchNet baseline (Schütt et al., 2017), 3D-invariant message passing.

Differs from the 2D baselines (GCN/GAT/GATv2/GINE) in one fundamental way:
SchNet's continuous-filter convolutions operate directly on 3D interatomic
distances via its own internal interaction graph (built from data.pos and
data.batch with SchNet's own cutoff/max_num_neighbors). To keep the
"identical graph topology" property of the controlled comparison, we pass
SchNet the SAME cutoff and max_num_neighbors as the 2D models
(constants.CUTOFF, constants.MAX_NUM_NEIGHBORS) — previously SchNet used
max_num_neighbors=32 while the 2D baselines used an uncapped radius_graph;
that asymmetry is now resolved (2D models are capped too, see dataset.py).

Node features: SchNet's own learned Z-embedding is concatenated with the
SAME precomputed data.x (7 or 15-dim hand-crafted features) used by every
other model, via atom_proj — so SchNet sees the identical hand-crafted
feature set as GCN/GAT/GATv2/GINE, on top of its own geometric message
passing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SchNet

from .layers import ResidualMLP
from ..data.constants import CUTOFF, MAX_NUM_NEIGHBORS
from ..data.features import node_feat_dim


class SchNetSigmaModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int   = 128,
        num_filters:     int   = 128,
        num_interactions: int  = 6,
        num_gaussians:   int   = 50,
        out_dim:         int   = 51,
        dropout:         float = 0.05,
        use_extended:    bool  = True,
    ):
        super().__init__()
        n_feat = node_feat_dim(use_extended)

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=CUTOFF,
            max_num_neighbors=MAX_NUM_NEIGHBORS,
        )
        # Replace SchNet's own output MLP — we use the shared ResidualMLP head
        # on top of (interaction output + hand-crafted features).
        self.atom_proj = nn.Linear(hidden_channels + n_feat, hidden_channels)
        self.readout   = ResidualMLP(hidden_channels, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        h = self.schnet.embedding(data.z)

        # Use the SAME precomputed radius-graph edges as every other model
        # (data.edge_index, from dataset.py / features.radius_graph_pure),
        # instead of SchNet's internal interaction_graph(). This (a) avoids
        # the pyg-lib/torch_cluster dependency and (b) guarantees identical
        # graph topology across ALL architectures in the comparison.
        row, col = data.edge_index
        edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, data.edge_index, edge_weight, edge_attr)

        h = self.atom_proj(torch.cat([h, data.x], dim=-1))
        return F.softplus(self.readout(h))
