"""
attentive_fp.py — AtomAttentiveFP baseline для предсказания атомарных σ-профилей.

Оригинальная статья:
  Xiong et al., "Pushing the Boundaries of Molecular Representation for Drug
  Discovery with the Graph Attention Mechanism", J. Med. Chem. 2020.
  https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959

Реализация в PyG:
  https://pytorch-geometric.readthedocs.io/en/latest/generated/
  torch_geometric.nn.models.AttentiveFP.html

Внутренняя структура PyG AttentiveFP (проверено по исходникам v2.5+):
  lin1         — начальная проекция x → hidden
  gate_conv    — GATEConv (слой 0, использует edge_attr)
  gru          — GRUCell для слоя 0
  atom_convs   — ModuleList[GATConv] для слоёв 1..num_layers-1
  atom_grus    — ModuleList[GRUCell] для слоёв 1..num_layers-1
  mol_conv/mol_gru/lin2 — молекулярный readout (нам не нужен)

Адаптация для per-atom предсказания:
  Останавливаемся после atom_convs/atom_grus и получаем per-atom векторы,
  которые пропускаем через ResidualMLP → softplus.

attentive_fp.py — AttentiveFP для per-atom σ-profile prediction.
mode="3d"      — edge_attr: 39-dim
mode="2d_pure" — edge_attr:  7-dim (только топология связей)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import global_mean_pool, global_max_pool

from .layers import ResidualMLP
from ..data.constants import MAX_Z, N_RBF
from ..data.features import node_feat_dim, edge_feat_dim


class AtomAttentiveFPModel(nn.Module):
    def __init__(
        self,
        hidden:        int   = 128,
        num_layers:    int   = 4,
        num_timesteps: int   = 2,
        out_dim:       int   = 51,
        dropout:       float = 0.05,
        use_extended:  bool  = True,
        mode:          str   = "2d_pure",
        n_rbf:         int   = N_RBF,
    ):
        super().__init__()
        self.hidden  = hidden
        self.dropout = dropout
        n_feat = node_feat_dim(use_extended, mode=mode)
        e_feat = edge_feat_dim(use_extended, n_rbf=n_rbf, mode=mode)

        self.z_embed    = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.afp = AttentiveFP(
            in_channels=hidden, hidden_channels=hidden,
            out_channels=hidden, edge_dim=e_feat,
            num_layers=num_layers, num_timesteps=num_timesteps,
            dropout=dropout,
        )

        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.global_norm = nn.LayerNorm(hidden)
        self.readout     = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        # AFP atom-level pass (без молекулярного readout)
        x = F.leaky_relu_(self.afp.lin1(x))
        h = self.afp.gate_conv(x, data.edge_index, data.edge_attr)
        h = F.elu_(h)
        h = F.dropout(h, p=self.afp.dropout, training=self.training)
        x = self.afp.gru(h, x).relu_()

        for conv, gru in zip(self.afp.atom_convs, self.afp.atom_grus):
            h = conv(x, data.edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.afp.dropout, training=self.training)
            x = gru(h, x).relu()

        g_mean = global_mean_pool(x, data.batch)[data.batch]
        g_max  = global_max_pool(x, data.batch)[data.batch]
        h = self.global_norm(
            x + self.global_proj(torch.cat([x, g_mean, g_max], dim=-1))
        )
        return F.softplus(self.readout(h))
