"""
gat.py — GAT (Velickovic et al., ICLR 2018).

Paper:    "Graph Attention Networks" — https://arxiv.org/abs/1710.10903
Official: https://github.com/PetarV-/GAT (TensorFlow, author's repo)
Layer used here: torch_geometric.nn.GATConv (PyG's maintained implementation).

Note: GATConv does not consume edge_attr — attention weights are computed
from node features only, per the original paper. Chemical-bond topology
(edge_index) is used; the 7-dim bond-type/aromatic/ring edge features
built by src/data/features.py are simply not passed to this model
(GAT has no mechanism to use them, in the original paper or here).

FIDELITY NOTE (see PROVENANCE.md): original GAT is a node classification
architecture with no molecule-level pooling. The previous benchmark version
added a shared, hand-written global mean+max pooling block here (identical
to five other architectures) — removed in this version. This model now
ends with the attention-conv stack only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim


class GATSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        num_layers: int = 4,
        heads: int = 8,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        n_feat = node_feat_dim()
        head_dim = hidden // heads

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GATConv(hidden, head_dim, heads=heads,
                    concat=True, dropout=0.0, add_self_loops=True)
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
