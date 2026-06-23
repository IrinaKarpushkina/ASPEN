"""
final_model.py — placeholder для финальной/лучшей модели.

ВАЖНО: вы упоминали, что финальная архитектура скорее всего изменится
("моя лучшая модель... скорее всего, я её поменяю"). Этот файл — шаблон,
показывающий КАК интегрировать новую архитектуру в общий пайплайн, не
больше. Текущая реализация — это GINE + дополнительный 4-й слой и
share-параметр для CombinedLoss-ablation; замените FinalSigmaModel на
вашу актуальную архитектуру, сохранив контракт:

    def __init__(self, hidden, ..., out_dim=51, dropout=0.05, use_extended=True)
    def forward(self, data) -> Tensor (N_atoms, 51), non-negative

Требования для совместимости с train.py / evaluate.py / configs:
  - Вход: data.x (precomputed node features), data.edge_index,
    data.edge_attr, data.z, data.pos, data.batch — всё уже готово
    в dataset.py, ничего пересчитывать в forward() не нужно.
  - Выход: softplus (неотрицательный) (N_atoms, 51) тензор.
  - __init__ принимает use_extended (для node_feat_dim/edge_feat_dim).
  - Зарегистрировать в src/models/__init__.py -> MODEL_REGISTRY["final"].
  - Создать configs/final_model.yaml (defaults: base.yaml, loss: combined
    для ablation, или mse для controlled comparison).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

from .layers import ResidualMLP
from ..data.constants import MAX_Z, N_RBF
from ..data.features import node_feat_dim, edge_feat_dim


class FinalSigmaModel(nn.Module):
    """Temporary stand-in: GINE backbone with deeper stack (5 layers)."""

    def __init__(
        self,
        hidden:       int   = 213,
        num_layers:   int   = 5,
        out_dim:      int   = 51,
        dropout:      float = 0.05,
        use_extended: bool  = True,
        n_rbf:        int   = N_RBF,
    ):
        super().__init__()
        n_feat = node_feat_dim(use_extended)
        e_feat = edge_feat_dim(use_extended, n_rbf=n_rbf)

        self.z_embed    = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden),
                ),
                eps=0.0, train_eps=True, edge_dim=e_feat,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.global_norm = nn.LayerNorm(hidden)
        self.readout      = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x     = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = norm(self.dropout(self.act(conv(h, data.edge_index, data.edge_attr))) + h)

        g_mean = global_mean_pool(h, data.batch)[data.batch]
        g_max  = global_max_pool(h, data.batch)[data.batch]
        h = self.global_norm(h + self.global_proj(torch.cat([h, g_mean, g_max], dim=-1)))

        return F.softplus(self.readout(h))
