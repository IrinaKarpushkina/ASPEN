"""
egnn.py — E(n)-Equivariant Graph Neural Network (EGNN) for per-atom
sigma-profile prediction.

Paper:    Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural
          Networks", ICML 2021. https://arxiv.org/abs/2102.09844
Official: https://github.com/vgsatorras/egnn

No PyG built-in layer exists for EGNN, so this is a direct
reimplementation of the paper's Equal Convolutional (EGCL) layer,
equations (3)-(6):

    m_ij   = phi_e( h_i, h_j, ||x_i - x_j||^2, a_ij )                  (3)
    x_i'   = x_i + C * sum_{j != i} (x_i - x_j) * phi_x(m_ij)          (4)
    m_i    = sum_{j in N(i)} m_ij                                       (5)
    h_i'   = phi_h( h_i, m_i )                                          (6)

with phi_e, phi_x, phi_h small MLPs (exact layer sizes below follow the
official repo's `models/egnn_clean/egnn_clean.py::E_GCL`, up to
substituting the invariant edge attribute `a_ij` with this benchmark's
edge attribute — here simply omitted, since the shared radius-graph
featurizer (`src/data/features_3d.py`) has no bond-order/edge-type
information (see `constants_3d.py` docstring) and squared distance alone
is the model's only edge signal, per the paper's own QM9 property-
prediction setup (Sec. 4.2 / Appendix, which uses no edge attributes for
QM9 either).

Using squared Euclidean distance ||x_i - x_j||^2 as the only edge
invariant (rather than raw distance) and updating positions x_i (Eq. 4)
by a `sum_{j != i}` over the OWN input graph's neighbourhood keeps every
quantity invariant to global rotation/translation of the input (the
positions only ever appear inside a difference vector times an invariant
scalar phi_x(m_ij), which is exactly why EGNN doesn't need spherical
harmonics or explicit equivariant tensor features to be E(3)-equivariant
in x and invariant in h) — verified for this implementation in
`tests/test_equivariance_3d.py`.

Coordinate updates (Eq. 4) are kept in this benchmark (as in the official
repo's default QM9 property-prediction script,
`qm9/models.py::EGNN.forward`, which does update x every layer even
though only h is read out at the end) for architectural fidelity; only
the final h_i is fed to the per-atom sigma-profile head — the coordinate
channel is discarded after the last layer, exactly as in the official
property-prediction pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features_3d import node_feat_dim_3d


class EGCL(nn.Module):
    """One E(n)-Equivariant Graph Convolutional Layer (Satorras et al.,
    Eqs. 3-6), edge_index convention: edge_index[0]=j (neighbour/source),
    edge_index[1]=i (center/target), message flows j -> i."""

    def __init__(self, hidden: int, dropout: float = 0.05, coord_update: bool = True):
        super().__init__()
        self.coord_update = coord_update
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        # zero-init the last coordinate-update layer, as in the official
        # repo (`egnn_clean.py`: `nn.init.xavier_uniform_(..., gain=0.001)`)
        # so training starts near the identity coordinate map.
        nn.init.xavier_uniform_(self.phi_x[-1].weight, gain=0.001)

        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, x, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]
        diff = x[i] - x[j]
        dist2 = (diff ** 2).sum(dim=-1, keepdim=True)

        m_ij = self.phi_e(torch.cat([h[i], h[j], dist2], dim=-1))

        if self.coord_update:
            C = 1.0 / (scatter(torch.ones_like(j, dtype=torch.float), i,
                               dim=0, dim_size=num_nodes, reduce="sum").clamp(min=1.0))
            coord_msg = diff * self.phi_x(m_ij)
            x_update = scatter(coord_msg, i, dim=0, dim_size=num_nodes, reduce="sum")
            x = x + C.unsqueeze(-1) * x_update

        m_i = scatter(m_ij, i, dim=0, dim_size=num_nodes, reduce="sum")
        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))
        h = self.norm(self.drop(h_new) + h)
        return h, x


class EGNNSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        num_layers: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
        coord_update: bool = True,
    ):
        super().__init__()
        n_feat = node_feat_dim_3d()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.layers = nn.ModuleList([
            EGCL(hidden, dropout=dropout, coord_update=coord_update)
            for _ in range(num_layers)
        ])

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        h = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        x = data.pos
        n = h.size(0)

        for layer in self.layers:
            h, x = layer(h, x, data.edge_index, n)

        return F.softplus(self.readout(h))
