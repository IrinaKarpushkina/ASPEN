"""
gine.py — GINE (Hu et al., ICLR 2020), edge-feature extension of
GIN (Xu et al., ICLR 2019).

Papers:
  GIN:  "How Powerful are Graph Neural Networks?" — https://arxiv.org/abs/1810.00826
        Official: https://github.com/weihua916/powerful-gnns
  GINE: "Strategies for Pre-training Graph Neural Networks" — https://arxiv.org/abs/1905.12265
        Official: https://github.com/snap-stanford/pretrain-gnns
Layer used here: torch_geometric.nn.GINEConv (PyG's maintained implementation
        of GIN's sum-aggregation + edge-feature injection).

FIDELITY NOTE (see PROVENANCE.md):
GIN's own paper defines a molecule-level readout (sum-pooling + concatenation
across layers, "Jumping Knowledge") ONLY for graph-classification/regression
tasks. For node-level tasks, Xu et al. and the pretrain-gnns node-level
pretext tasks (context prediction / attribute masking) use the per-node
GIN/GINE output directly, with NO pooling — which is exactly our setting
(per-atom sigma-profile prediction).

The previous benchmark version added a shared, hand-written global mean+max
pooling block here (identical to five other architectures) — removed in
this version. This model now ends with the GINEConv stack only, matching
the node-level precedent set in the GINE paper itself.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim, edge_feat_dim


class GINESigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 213,
        num_layers: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        n_feat = node_feat_dim()
        e_feat = edge_feat_dim()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                    nn.SiLU(), nn.Linear(hidden, hidden),
                ),
                eps=0.0, train_eps=True, edge_dim=e_feat,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(self.dropout(self.act(conv(h, data.edge_index, data.edge_attr))) + h)

        return F.softplus(self.readout(h))
