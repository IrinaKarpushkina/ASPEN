"""
torchmdnet.py — TorchMD-Net (Equivariant Transformer, "ET") for per-atom
sigma-profile prediction.

Paper:    Tholke & De Fabritiis, "TorchMD-NET: Equivariant Transformers
          for Neural Network based Molecular Potentials", ICLR 2022.
          https://arxiv.org/abs/2202.02541
Official: https://github.com/torchmd/torchmd-net
          (`torchmdnet.models.torchmd_et.TorchMD_ET`,
          `torchmdnet.models.utils.EquivariantMultiHeadAttention`)

No PyG built-in layer exists, so this reimplements the Equivariant
Transformer's attention block directly from the paper's equations
(Sec. 3.2 / Fig. 2), matched against the module structure of the official
repo's `EquivariantMultiHeadAttention`:

For each atom, a scalar channel x_i in R^F and a vector channel
vec_i in R^(F x 3) (initialised to zero) are maintained, exactly as in
PaiNN — TorchMD-Net's ET layer is, in the official paper's own framing,
a dot-product-attention generalisation of PaiNN's convolutional message
(replace the fixed radial filter with a learned, multi-head
attention-weighted one):

    q_i = W_Q x_i ,  k_j = W_K x_j ,  v_j = W_V x_j          (per head)
    dk_ij = phi_dk(RBF(d_ij))                                (distance filter)
    attn_ij = SiLU( (q_i * k_j * dk_ij).sum(-1) ) * f_cut(d_ij)   (Eq. "cosine-cutoff SiLU attention")
    s_ij = attn_ij * v_j                                     (scalar message)
    dvec_ij = attn_ij * (v_j (x) as vector-gate) combined with u_ij (x) another gate
    x_i'   = x_i + sum_j s_ij       (+ a vector-channel gated update, see below)
    vec_i' = vec_i + sum_j [ dvec_ij ]

exactly mirroring PaiNN's vector-channel construction (this is intentional
— TorchMD-Net's paper explicitly describes its vector update as
"following PaiNN", replacing PaiNN's *fixed* RBF-linear filter with an
*attention*-modulated one). We reuse the same `VectorLinear`/cosine-cutoff
utilities as `models_3d/painn.py` for that reason, and the same PaiNN-style
per-atom update block (paper's own "update function" is the same as
PaiNN's Eqs. 9-13) for the mixing step.

Multi-head attention here uses H heads over the F scalar channels (F must
be divisible by H), each head's attention weight additionally gating both
the scalar message AND the two vector-channel gates, per the paper.
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
from .painn import VectorLinear, PaiNNMixing, _cosine_cutoff


class EquivariantAttention(nn.Module):
    """One Equivariant Transformer layer (attention-modulated message +
    PaiNN-style vector update), Tholke & De Fabritiis 2022."""

    def __init__(self, hidden: int, num_heads: int, num_rbf: int, cutoff: float,
                 dropout: float = 0.05):
        super().__init__()
        assert hidden % num_heads == 0
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.cutoff = cutoff

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, 3 * hidden)  # scalar-msg, vec-gate1, vec-gate2

        self.dk_proj = nn.Linear(num_rbf, hidden)
        self.du_proj = nn.Linear(num_rbf, hidden)  # gates the u_ij (direction) term

        self.vec_proj = VectorLinear(hidden, 2 * hidden)  # splits into U(v_j), gate-mix

        self.out_s = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, v, pos, rbf, edge_weight, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]
        E = j.size(0)
        H, D = self.num_heads, self.head_dim

        r_ij = pos[i] - pos[j]
        u_ij = r_ij / edge_weight.clamp(min=1e-6).unsqueeze(-1)

        q = self.q_proj(s)[i].view(E, H, D)
        k = self.k_proj(s)[j].view(E, H, D)
        val_s, gate1, gate2 = self.v_proj(s)[j].split(self.hidden, dim=-1)
        val_s = val_s.view(E, H, D)

        dk = self.dk_proj(rbf).view(E, H, D)
        cutoff_w = _cosine_cutoff(edge_weight, self.cutoff).view(E, 1, 1)

        attn = F.silu((q * k * dk).sum(dim=-1)) * cutoff_w.squeeze(-1)  # (E, H)
        attn = self.dropout(attn)

        s_msg = (attn.unsqueeze(-1) * val_s).reshape(E, self.hidden)    # (E, F)

        # vector channel: gate1 scales the neighbour's own vector feature,
        # gate2 scales the unit bond-direction vector -- exactly PaiNN's
        # dv_ij construction, but with an attention-weighted filter instead
        # of PaiNN's fixed RBF-linear filter.
        du = self.du_proj(rbf)
        Uv_j = self.vec_proj(v)[j, : self.hidden]                        # (E, F, 3)
        vec_msg = (attn.repeat_interleave(D, dim=-1) * gate1).unsqueeze(-1) * Uv_j + \
                  (attn.repeat_interleave(D, dim=-1) * gate2 * du).unsqueeze(-1) * u_ij.unsqueeze(1)

        ds = scatter(s_msg, i, dim=0, dim_size=num_nodes, reduce="sum")
        dv = scatter(vec_msg, i, dim=0, dim_size=num_nodes, reduce="sum")

        s = s + self.out_s(ds)
        v = v + dv
        return s, v


class TorchMDNetSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 192,
        num_layers: int = 4,
        num_heads: int = 8,
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

        self.attn_layers = nn.ModuleList([
            EquivariantAttention(hidden, num_heads, num_rbf, cutoff, dropout=dropout)
            for _ in range(num_layers)
        ])
        # per-atom PaiNN-style mixing block after every attention layer,
        # matching the official ET layer's structure (attention block +
        # gated-update block per layer).
        self.mixings = nn.ModuleList([PaiNNMixing(hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        s = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        n = s.size(0)
        v = torch.zeros(n, s.size(-1), 3, device=s.device, dtype=s.dtype)

        rbf = self.distance_expansion(data.edge_weight)

        for attn, mixing, norm in zip(self.attn_layers, self.mixings, self.norms):
            s, v = attn(s, v, data.pos, rbf, data.edge_weight, data.edge_index, n)
            s, v = mixing(s, v)
            s = norm(self.dropout(s))

        return F.softplus(self.readout(s))
