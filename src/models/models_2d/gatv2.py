"""
gatv2.py — GATv2 (Brody, Alon & Yahav, ICLR 2022).

Paper:    "How Attentive are Graph Attention Networks?"
          https://arxiv.org/abs/2105.14491
Official: https://github.com/tech-srl/how_attentive_are_gats
Layer used here: torch_geometric.nn.GATv2Conv (PyG's maintained
          implementation of the fixed, dynamic-attention formulation).

Note: like GATConv, GATv2Conv does not consume edge_attr in the original
formulation — only node features feed the attention mechanism.

FIDELITY NOTE (see PROVENANCE.md): same as GAT — the original paper is a
node-level architecture with no molecule-level pooling. The shared
hand-written global mean+max pooling block from the previous benchmark
version has been removed; this model now ends with the attention-conv
stack only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim


class GATv2SigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 216,
        num_layers: int = 4,
        heads: int = 8,
        out_dim: int = 51,
        dropout: float = 0.05,
        share_weights: bool = False,
    ):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        n_feat = node_feat_dim()
        head_dim = hidden // heads

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden, head_dim, heads=heads, concat=True,
                      dropout=0.0, add_self_loops=True,
                      share_weights=share_weights)
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
            h = norm(self.dropout(self.act(conv(h, data.edge_index))) + h)

        return F.softplus(self.readout(h))
