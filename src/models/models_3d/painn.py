"""
painn.py — PaiNN (Polarizable Atom Interaction Neural Network) for
per-atom sigma-profile prediction.

Paper:    Schutt, Unke & Gastegger, "Equivariant Message Passing for the
          Prediction of Tensorial Properties and Molecular Spectra",
          ICML 2021. https://proceedings.mlr.press/v139/schutt21a.html
Official: https://github.com/atomistic-machine-learning/schnetpack
          (`schnetpack.representation.PaiNN`)

`torch_geometric.nn.models` has no built-in PaiNN layer (checked against
PyG 2.8, the version this repo targets — see requirements.txt), so this
is a direct reimplementation of the paper's message/update equations
(Eqs. 5-13), matched against the official SchNetPack implementation's
module structure (`PaiNNInteraction` / `PaiNNMixing`).

Each atom carries a SCALAR feature s_i in R^F and a VECTOR feature v_i in
R^(F x 3) (initialised to zero), and the two channels are kept coupled
through the whole network so that s_i stays rotation-INVARIANT and v_i
stays rotation-EQUIVARIANT throughout (verified in
`tests/test_equivariance_3d.py`).

Message block (paper Eqs. 5-8), edge j -> i, r_ij = x_i - x_j, d_ij =
||r_ij||, u_ij = r_ij / d_ij:

    phi_s, phi_vv, phi_vs = split( MLP(s_j) )                      (per-channel, dim F each)
    W_s, W_vv, W_vs       = split( Linear(RBF(d_ij)) * f_cut(d_ij) )
    ds_ij  = phi_s  * W_s
    dv_ij  = phi_vv * W_vv (x) v_j  +  phi_vs * W_vs (x) u_ij
    Delta_s_i = sum_j ds_ij ,   Delta_v_i = sum_j dv_ij

Update block (paper Eqs. 9-13), no message passing, purely per-atom:

    Uv_i, Vv_i = U(v_i), V(v_i)                 (channel-wise linear maps,
                                                  applied identically to
                                                  each of the 3 spatial
                                                  components -> equivariant)
    a_vv, a_sv, a_ss = split( MLP(cat[s_i, ||Vv_i||])  )
    Delta_v_i = a_vv * Uv_i
    Delta_s_i = a_ss + a_sv * <Uv_i, Vv_i>       (per-channel dot product
                                                  over the 3 spatial dims
                                                  -> rotation-invariant)

Radial basis / cutoff: PaiNN's own paper (Sec. 4.1) uses the same
Gaussian-RBF + smooth cosine cutoff SchNet uses; this benchmark reuses
`torch_geometric.nn.models.schnet.GaussianSmearing` for exactly that
reason (same distance-embedding "budget", `constants_3d.N_RBF`, is given
to SchNet and PaiNN, so the comparison isolates the *update rule*, not the
radial basis).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import N_RBF, CUTOFF
from ...data.features_3d import node_feat_dim_3d


def _cosine_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(d * torch.pi / cutoff) + 1.0) * (d < cutoff).float()


class VectorLinear(nn.Module):
    """Channel-wise linear map applied identically to each of the 3 spatial
    components of a (N, F_in, 3) vector-feature tensor -> (N, F_out, 3).
    No bias (a bias would break rotation equivariance)."""

    def __init__(self, f_in: int, f_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(f_out, f_in))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("of,nfd->nod", self.weight, v)


class PaiNNInteraction(nn.Module):
    """Message block: Eqs. 5-8."""

    def __init__(self, hidden: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.hidden = hidden
        self.cutoff = cutoff
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden),
        )
        self.rbf_lin = nn.Linear(num_rbf, 3 * hidden)

    def forward(self, s, v, pos, rbf, edge_weight, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]  # source j, target i

        r_ij = pos[i] - pos[j]
        u_ij = r_ij / edge_weight.clamp(min=1e-6).unsqueeze(-1)

        filt = (self.rbf_lin(rbf) *
                _cosine_cutoff(edge_weight, self.cutoff).unsqueeze(-1))
        phi_s, phi_vv, phi_vs = self.phi(s[j]).chunk(3, dim=-1)
        w_s, w_vv, w_vs = filt.chunk(3, dim=-1)

        ds_ij = phi_s * w_s                                          # (E, F)
        dv_ij = (phi_vv * w_vv).unsqueeze(-1) * v[j] + \
                (phi_vs * w_vs).unsqueeze(-1) * u_ij.unsqueeze(1)     # (E, F, 3)

        ds_i = scatter(ds_ij, i, dim=0, dim_size=num_nodes, reduce="sum")
        dv_i = scatter(dv_ij, i, dim=0, dim_size=num_nodes, reduce="sum")

        return s + ds_i, v + dv_i


class PaiNNMixing(nn.Module):
    """Update block: Eqs. 9-13 (no message passing, per-atom only)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.U = VectorLinear(hidden, hidden)
        self.V = VectorLinear(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden),
        )

    def forward(self, s, v):
        Uv = self.U(v)                       # (N, F, 3)
        Vv = self.V(v)                       # (N, F, 3)
        Vv_norm = Vv.norm(dim=-1)            # (N, F) -- rotation invariant

        a_vv, a_sv, a_ss = self.mlp(torch.cat([s, Vv_norm], dim=-1)).chunk(3, dim=-1)

        dot = (Uv * Vv).sum(dim=-1)          # (N, F) -- rotation invariant
        ds = a_ss + a_sv * dot
        dv = a_vv.unsqueeze(-1) * Uv

        return s + ds, v + dv


class PaiNNSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 224,
        num_layers: int = 4,
        num_rbf: int = N_RBF,
        cutoff: float = CUTOFF,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        n_feat = node_feat_dim_3d()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_rbf)

        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden, num_rbf, cutoff) for _ in range(num_layers)
        ])
        self.mixings = nn.ModuleList([
            PaiNNMixing(hidden) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        s = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        n = s.size(0)
        v = torch.zeros(n, s.size(-1), 3, device=s.device, dtype=s.dtype)

        rbf = self.distance_expansion(data.edge_weight)

        for interaction, mixing, norm in zip(self.interactions, self.mixings, self.norms):
            s, v = interaction(s, v, data.pos, rbf, data.edge_weight, data.edge_index, n)
            s, v = mixing(s, v)
            s = norm(self.dropout(s))

        return F.softplus(self.readout(s))
