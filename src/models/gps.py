"""
gps.py — GraphGPS baseline для предсказания атомарных σ-профилей.

Оригинальная статья:
  Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
  Transformer", NeurIPS 2022.
  https://arxiv.org/abs/2205.12454

Реализация GPSConv в PyG:
  https://pytorch-geometric.readthedocs.io/en/latest/generated/
  torch_geometric.nn.conv.GPSConv.html

Пример PyG:
  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py

Архитектура GPS (один слой):
  1. Positional Encoding (RWPE) — предвычислен в dataset.py, добавляется к x
  2. Local MPNN (GINEConv) — локальная передача сообщений
  3. Global MultiheadAttention — каждый атом видит все атомы молекулы
  4. Feed-Forward Network внутри GPSConv

Адаптация для per-atom предсказания:
  GPSConv работает на уровне узлов — атомные представления напрямую
  пропускаем в ResidualMLP без молекулярного pooling.
  Встроенный global attention уже даёт молекулярный контекст,
  поэтому дополнительный mean+max pooling не нужен.

Исправление v2: attn_dropout передаётся через attn_kwargs (не как
прямой аргумент GPSConv), согласно актуальной сигнатуре PyG.

gps.py — GraphGPS для per-atom σ-profile prediction.
mode="3d"      — edge_attr: 39-dim
mode="2d_pure" — edge_attr:  7-dim (только топология связей)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GPSConv, GINEConv

from .layers import ResidualMLP
from ..data.constants import MAX_Z, N_RBF
from ..data.features import node_feat_dim, edge_feat_dim

WALK_LENGTH = 16


class GPSSigmaModel(nn.Module):
    def __init__(
        self,
        hidden:       int   = 128,
        num_layers:   int   = 4,
        heads:        int   = 4,
        out_dim:      int   = 51,
        dropout:      float = 0.05,
        use_extended: bool  = True,
        mode:         str   = "2d_pure",
        n_rbf:        int   = N_RBF,
        walk_length:  int   = WALK_LENGTH,
    ):
        super().__init__()
        self.hidden      = hidden
        self.walk_length = walk_length
        n_feat = node_feat_dim(use_extended, mode=mode)
        e_feat = edge_feat_dim(use_extended, n_rbf=n_rbf, mode=mode)

        self.z_embed    = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.pe_proj    = nn.Linear(walk_length, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4 + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GPSConv(
                channels=hidden,
                conv=GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                        nn.SiLU(), nn.Linear(hidden, hidden),
                    ),
                    edge_dim=e_feat,
                ),
                heads=heads,
                dropout=dropout,
                norm="layer_norm",
                attn_kwargs={"dropout": dropout},
            )
            for _ in range(num_layers)
        ])
        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        pe = getattr(data, "pe", None)
        if pe is None:
            pe = torch.zeros(data.x.size(0), self.walk_length, device=data.x.device)
        else:
            pe = pe.to(data.x.device)

        z_emb  = self.z_embed(data.z.clamp(max=MAX_Z))
        pe_emb = self.pe_proj(pe)
        x = self.input_proj(torch.cat([data.x, z_emb, pe_emb], dim=-1))

        h = x
        for conv in self.convs:
            h = conv(h, data.edge_index, data.batch, edge_attr=data.edge_attr)

        return F.softplus(self.readout(h))
