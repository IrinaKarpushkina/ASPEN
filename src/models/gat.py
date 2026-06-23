"""
gat.py — GAT baseline (Veličković et al., ICLR 2018).
mode="3d" / "2d_pure" — см. gcn.py.
GATConv не использует edge_attr, поэтому e_feat не нужен.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from .layers import ResidualMLP
from ..data.constants import MAX_Z
from ..data.features import node_feat_dim


class GATSigmaModel(nn.Module):
    def __init__(
        self,
        hidden:       int   = 256,
        num_layers:   int   = 4,
        heads:        int   = 8,
        out_dim:      int   = 51,
        dropout:      float = 0.05,
        use_extended: bool  = True,
        mode:         str   = "2d_pure",
    ):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        n_feat   = node_feat_dim(use_extended, mode=mode)
        head_dim = hidden // heads

        self.z_embed    = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GATConv(hidden, head_dim, heads=heads,
                    concat=True, dropout=0.0, add_self_loops=True)
            for _ in range(num_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.global_norm = nn.LayerNorm(hidden)
        self.readout     = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x     = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(self.dropout(self.act(conv(h, data.edge_index))) + h)

        g_mean = global_mean_pool(h, data.batch)[data.batch]
        g_max  = global_max_pool(h, data.batch)[data.batch]
        h = self.global_norm(h + self.global_proj(torch.cat([h, g_mean, g_max], dim=-1)))
        return F.softplus(self.readout(h))
