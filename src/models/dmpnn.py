"""
dmpnn.py — Directed MPNN (D-MPNN / Chemprop) для атомарных σ-профилей.

Оригинальная статья:
  Yang et al., "Analyzing Learned Molecular Representations for Property
  Prediction", J. Chem. Inf. Model. 2019.
  https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237
  arXiv: https://arxiv.org/abs/1904.01561

Репозиторий Chemprop:
  https://github.com/chemprop/chemprop

Ключевое отличие D-MPNN от стандартного MPNN:
  Сообщения передаются по РЁБРАМ (bond-centric), а не по атомам.
  Для ребра v→w при агрегации исключается обратное ребро w→v
  ("reverse edge masking"), что убирает "totter" эффект.

  h_{vw}^{t+1} = ReLU( W_i * init_{vw} + W_m * sum_{k→v, k≠w} h_{kv}^t )

  После T шагов атомные представления:
  h_v = ReLU( W_a * cat( x_v, sum_{w→v} h_{wv}^T ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter

from .layers import ResidualMLP
from ..data.constants import MAX_Z
from ..data.features import node_feat_dim, edge_feat_dim, N_RBF


def build_reverse_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Для каждого ребра e=(i→j) находит обратное ребро e'=(j→i) в edge_index.
    Возвращает тензор reverse_idx (E,), где reverse_idx[e] = e'.

    Использует словарный поиск — O(E), надёжно при любых edge_index.
    Предполагает, что для каждого ребра (i,j) существует обратное (j,i).
    """
    row = edge_index[0].tolist()
    col = edge_index[1].tolist()

    # Строим словарь: (j, i) -> индекс ребра e' в edge_index
    rev_dict = {}
    for e, (i, j) in enumerate(zip(row, col)):
        rev_dict[(j, i)] = e

    # Для каждого ребра e=(i,j) ищем его обратное (j,i)
    E = edge_index.size(1)
    reverse_idx = torch.zeros(E, dtype=torch.long, device=edge_index.device)
    for e, (i, j) in enumerate(zip(row, col)):
        reverse_e = rev_dict.get((j, i), e)   # fallback: сам на себя (не должно быть)
        reverse_idx[e] = reverse_e

    return reverse_idx


class DMPNNConv(nn.Module):
    """
    Один шаг directed message passing с reverse-edge masking.

    agg_{v\w} = sum_{k→v, k≠w} h_{kv}  ≡  sum_{k→v} h_{kv} - h_{wv}
    h_{vw}^{t+1} = ReLU( W_i * h_{vw}^0 + W_m * agg_{v\w} )
    """

    def __init__(self, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.W_m = nn.Linear(hidden, hidden, bias=False)
        self.W_i = nn.Linear(hidden, hidden, bias=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Xavier инициализация для стабильного старта
        nn.init.xavier_uniform_(self.W_m.weight)
        nn.init.xavier_uniform_(self.W_i.weight)

    def forward(
        self,
        h_edge: torch.Tensor,       # (E, hidden) текущие bond states
        h_init: torch.Tensor,       # (E, hidden) начальная инициализация
        edge_index: torch.Tensor,   # (2, E)
        reverse_idx: torch.Tensor,  # (E,)
        num_nodes: int,
    ) -> torch.Tensor:
        # Агрегируем incoming bond states в узлы (по target node = edge_index[1])
        # agg_full[v] = sum_{e: edge_index[1,e] == v} h_edge[e]
        agg_full = scatter(
            h_edge, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum"
        )  # (num_nodes, hidden)

        # Для ребра e=(v→w): agg_v = agg_full[v] - h_edge[reverse_idx[e]]
        # h_edge[reverse_idx[e]] — bond state обратного ребра (w→v)
        agg_edge = agg_full[edge_index[0]] - h_edge[reverse_idx]

        h_new = self.act(self.norm(self.W_i(h_init) + self.W_m(agg_edge)))
        return self.drop(h_new)


class DMPNNSigmaModel(nn.Module):
    """
    D-MPNN для per-atom σ-profile prediction.

    Pipeline:
      1. Atom features: precomputed data.x + Z-embedding → hidden
      2. Bond init: cat(atom_feat_source, edge_attr) → hidden
      3. T шагов directed message passing (reverse-edge masking)
      4. Atom readout: x_atom + sum(incoming bond states)
      5. Global mean+max context
      6. ResidualMLP → softplus → (N_atoms, 51)
    """

    def __init__(
        self,
        hidden: int = 243,
        num_steps: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
        use_extended: bool = True,
        n_rbf: int = N_RBF,
        mode: str = "3d",   # "3d" или "2d_pure"
    ):
        super().__init__()
        self.hidden = hidden
        self.num_steps = num_steps
        n_feat = node_feat_dim(use_extended, mode=mode)
        e_feat = edge_feat_dim(use_extended, n_rbf=n_rbf, mode=mode)

        # Atom feature projection
        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.atom_proj = nn.Linear(n_feat + hidden // 4, hidden)
        self.atom_norm = nn.LayerNorm(hidden)

        # Bond initialization
        self.bond_init = nn.Linear(hidden + e_feat, hidden, bias=False)
        self.bond_init_norm = nn.LayerNorm(hidden)

        # D-MPNN conv (веса разделяются между шагами — как в оригинальном Chemprop)
        self.conv = DMPNNConv(hidden, dropout=dropout)

        # Atom readout после message passing
        self.atom_readout = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Global context
        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.global_norm = nn.LayerNorm(hidden)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        N = data.z.size(0)
        E = data.edge_index.size(1)

        # 1. Atom features
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x_atom = self.atom_norm(
            F.relu(self.atom_proj(torch.cat([data.x, z_emb], dim=-1)))
        )

        # 2. Bond initialization
        src = data.edge_index[0]
        h_init = F.relu(
            self.bond_init_norm(
                self.bond_init(torch.cat([x_atom[src], data.edge_attr], dim=-1))
            )
        )
        # Защита от NaN на входе
        h_init = torch.nan_to_num(h_init, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. Reverse edge index (кешируем на объекте данных между эпохами)
        if not hasattr(data, "_rev_idx") or data._rev_idx is None:
            data._rev_idx = build_reverse_index(data.edge_index)
        rev_idx = data._rev_idx

        # 4. T шагов directed message passing
        h_bond = h_init
        for _ in range(self.num_steps):
            h_bond = self.conv(h_bond, h_init, data.edge_index, rev_idx, N)
            h_bond = torch.nan_to_num(h_bond, nan=0.0)

        # 5. Atom readout: суммируем incoming bond states
        bond_agg = scatter(
            h_bond, data.edge_index[1], dim=0, dim_size=N, reduce="sum"
        )

        h = self.atom_readout(torch.cat([x_atom, bond_agg], dim=-1))

        # 6. Global context
        g_mean = global_mean_pool(h, data.batch)[data.batch]
        g_max = global_max_pool(h, data.batch)[data.batch]
        h = self.global_norm(h + self.global_proj(torch.cat([h, g_mean, g_max], dim=-1)))

        return F.softplus(self.readout(h))
