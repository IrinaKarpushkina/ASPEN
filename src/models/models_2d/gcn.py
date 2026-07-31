"""
gcn.py — GCN (Kipf & Welling, ICLR 2017).

Paper:    "Semi-Supervised Classification with Graph Convolutional Networks"
          https://arxiv.org/abs/1609.02907
Official: https://github.com/tkipf/gcn (TensorFlow)
          https://github.com/tkipf/pygcn (author's own PyTorch port)
Layer used here: torch_geometric.nn.GCNConv (PyG's maintained, paper-checked
          implementation of the same propagation rule).

FIDELITY NOTE (see PROVENANCE.md):
The original GCN paper's task is node classification — a stack of GCNConv
layers followed directly by a per-node linear classifier. There is no
molecule-/graph-level pooling anywhere in the original architecture.

The previous version of this benchmark added an extra, hand-written
"global_mean_pool + global_max_pool -> MLP -> LayerNorm" block after the
conv stack, IDENTICAL across six different architectures. That block is
not part of the GCN paper, and because it was shared verbatim by most
models in the benchmark, it acted as a confound that compressed the
measured differences between architectures (most of the predictive signal
could come from the shared block rather than from the conv layers being
compared). It has been removed here: this model now ends with the conv
stack only, exactly as in the original paper/repo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim


class GCNSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        num_layers: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        n_feat = node_feat_dim()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(num_layers)])
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
