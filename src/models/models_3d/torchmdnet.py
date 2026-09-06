"""
torchmdnet.py — TorchMD-Net (Equivariant Transformer, "ET") for per-atom
sigma-profile prediction.

Paper:    Tholke & De Fabritiis, "TorchMD-NET: Equivariant Transformers
          for Neural Network based Molecular Potentials", ICLR 2022.
          https://arxiv.org/abs/2202.02541
Official: https://github.com/torchmd/torchmd-net
          (`torchmdnet.models.torchmd_et.TorchMD_ET`,
          `torchmdnet.models.utils.EquivariantMultiHeadAttention`,
          `torchmdnet.models.utils.NeighborEmbedding`)

REVISION NOTE: an earlier version of this file reimplemented the ET
attention block from the paper's prose description alone, without
checking it against the official source. Direct line-by-line comparison
against `torchmdnet/models/torchmd_et.py` and `utils.py` found one
functional bug and one structural simplification; both are fixed below
rather than kept as a "documented deviation", since nothing here requires
diverging from the reference:

  1. BUG (fixed): the vector-channel gates (`gate1`/`gate2` below) were
     additionally multiplied by the attention weight `attn`. In the
     official `message()`, `attn` gates ONLY the scalar value path
     (`x_ = x_ * attn.unsqueeze(2)`); the vector-channel gates
     (`vec1_`/`vec2_` there) are used un-weighted by attention -- the
     vector update is a pure distance/geometry-gated (PaiNN-style)
     filter, not an attention-weighted one. Fixed: `attn` no longer
     multiplies the vector-channel terms.
  2. STRUCTURE (fixed): the official architecture fuses attention and
     the scalar/vector "update" into a SINGLE block per layer (there is
     no separate PaiNN-style mixing block run afterwards): `vec_proj`
     is applied ONCE to the pre-message vector feature to obtain three
     chunks (vec1, vec2, vec3); `vec_dot = <vec1,vec2>` is a per-atom
     "self" term computed BEFORE message passing; after the
     attention-weighted message is aggregated, `o_proj` splits the
     aggregated scalar feature into (o1,o2,o3), and
         dx   = vec_dot * o2 + o3
         dvec = vec3 * o1 + (aggregated vector message)
     are added as the layer's residual. An earlier version of this file
     instead ran a full, separate `PaiNNMixing` block after each
     attention layer -- a different (not wrong per se, but not what the
     paper/official code does) parametrization. Replaced with the exact
     fused structure above; `PaiNNMixing` is no longer used here.

Additionally, the official default configuration (`neighbor_embedding=
True`) applies an initial `NeighborEmbedding` step (paper Eq. 3) before
the attention layers: each atom's embedding is mixed with a
distance-weighted embedding of its neighbours' types. This is
implemented below as `_NeighborEmbedding`, adapted to this benchmark's
node-feature convention (every architecture here embeds raw atomic
number separately and concatenates it with engineered features via
`input_proj`, rather than official ET/PaiNN's simpler raw-z-only
embedding) -- `_NeighborEmbedding` here mixes that already-combined
`s_0` with a neighbour-weighted raw atomic-number embedding, preserving
the mechanism (atom embedding + neighbour-type embedding weighted by a
distance filter, combined by a linear layer) rather than the exact
input representation, which this benchmark standardizes across every
architecture for a fair comparison.

Two other benchmark-wide, shared-infrastructure choices (unchanged from
before, and shared with `models_3d/painn.py`/`models_3d/schnet.py`):
  - Radial basis: `GaussianSmearing` (SchNet-style) rather than the
    official ET's default `ExpNormalSmearing`, so that every 3D
    architecture in this benchmark is given the same distance-embedding
    "budget" (`constants_3d.N_RBF`) -- isolating each architecture's
    update/attention rule from its choice of radial basis.
  - A radius-graph `edge_index`/`edge_weight` (shared featurizer,
    `src/data/features_3d.py`) is used in place of the official
    `OptimizedDistance` neighbour-list module (which additionally
    supports periodic boundary conditions, not needed here) and its
    CUDA-kernel-backed pair search.
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
from .painn import VectorLinear, _cosine_cutoff


class _NeighborEmbedding(nn.Module):
    """Initial neighbour-type embedding (paper Eq. 3, official
    `NeighborEmbedding`): mixes each atom's own scalar feature with a
    distance-weighted embedding of its neighbours' atomic numbers,
    before entering the attention layers."""

    def __init__(self, hidden: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.z_embed = nn.Embedding(MAX_Z + 1, hidden)
        self.dist_proj = nn.Linear(num_rbf, hidden)
        self.combine = nn.Linear(2 * hidden, hidden)

    def forward(self, s0, z, rbf, edge_weight, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]
        cutoff_w = _cosine_cutoff(edge_weight, self.cutoff).unsqueeze(-1)
        w = self.dist_proj(rbf) * cutoff_w                       # (E, F)
        msg = w * self.z_embed(z)[j]                              # (E, F)
        neighbor_s = scatter(msg, i, dim=0, dim_size=num_nodes, reduce="sum")
        return self.combine(torch.cat([s0, neighbor_s], dim=-1))


class EquivariantAttention(nn.Module):
    """One Equivariant Transformer layer, fused attention + update block,
    matching the official `EquivariantMultiHeadAttention` exactly
    (Tholke & De Fabritiis 2022)."""

    def __init__(self, hidden: int, num_heads: int, num_rbf: int, cutoff: float,
                 dropout: float = 0.05):
        super().__init__()
        assert hidden % num_heads == 0
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.cutoff = cutoff

        self.layernorm = nn.LayerNorm(hidden)

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, 3 * hidden)
        self.o_proj = nn.Linear(hidden, 3 * hidden)
        self.vec_proj = VectorLinear(hidden, 3 * hidden)  # bias-free by construction

        self.dk_proj = nn.Linear(num_rbf, hidden)
        self.dv_proj = nn.Linear(num_rbf, 3 * hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, s, v, pos, rbf, edge_weight, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]
        E = j.size(0)
        H, D = self.num_heads, self.head_dim

        r_ij = pos[i] - pos[j]
        u_ij = r_ij / edge_weight.clamp(min=1e-6).unsqueeze(-1)   # (E, 3)

        x = self.layernorm(s)                                     # official: layernorm on scalar only

        q = self.q_proj(x)[i].view(E, H, D)
        k = self.k_proj(x)[j].view(E, H, D)

        # vec_proj applied ONCE to the pre-message vector feature (self term,
        # not per-edge) -- official: vec1,vec2,vec3 = split(vec_proj(vec))
        vec1, vec2, vec3 = self.vec_proj(v).chunk(3, dim=1)        # each (N, hidden, 3)
        vec_dot = (vec1 * vec2).sum(dim=-1)                        # (N, hidden), per-atom self term

        dk = self.dk_proj(rbf).view(E, H, D)
        dv = self.dv_proj(rbf).view(E, H, 3 * D)

        v_j = self.v_proj(x)[j].view(E, H, 3 * D) * dv
        val_x, gate1, gate2 = v_j.split(D, dim=-1)                 # each (E, H, D)

        cutoff_w = _cosine_cutoff(edge_weight, self.cutoff)
        attn = F.silu((q * k * dk).sum(dim=-1)) * cutoff_w.unsqueeze(-1)  # (E, H)
        attn = self.dropout(attn)

        # scalar message: attention-weighted (official: x_ = x_ * attn)
        x_msg = (attn.unsqueeze(-1) * val_x).reshape(E, self.hidden)

        # vector message: NOT attention-weighted -- pure distance/geometry
        # gated (PaiNN-style), matching the official message() exactly
        v_flat = v.view(-1, H, D, 3)                               # reshape (N,hidden,3) per-head
        vec_j = v_flat[j]                                          # (E, H, D, 3)
        vec_msg = vec_j * gate1.unsqueeze(-1) + gate2.unsqueeze(-1) * u_ij.view(E, 1, 1, 3)
        vec_msg = vec_msg.reshape(E, self.hidden, 3)

        ds = scatter(x_msg, i, dim=0, dim_size=num_nodes, reduce="sum")
        dv_agg = scatter(vec_msg, i, dim=0, dim_size=num_nodes, reduce="sum")

        o1, o2, o3 = self.o_proj(ds).chunk(3, dim=-1)              # each (N, hidden)

        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(-1) + dv_agg

        return s + dx, v + dvec


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
        self.neighbor_embedding = _NeighborEmbedding(hidden, num_rbf, cutoff)

        self.attn_layers = nn.ModuleList([
            EquivariantAttention(hidden, num_heads, num_rbf, cutoff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)  # official: applied ONCE, after all layers

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_clamped = data.z.clamp(max=MAX_Z)
        z_emb = self.z_embed(z_clamped)
        s = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        n = s.size(0)
        v = torch.zeros(n, s.size(-1), 3, device=s.device, dtype=s.dtype)

        rbf = self.distance_expansion(data.edge_weight)

        s = self.neighbor_embedding(
            s, z_clamped, rbf, data.edge_weight, data.edge_index, n
        )

        for attn in self.attn_layers:
            s, v = attn(s, v, data.pos, rbf, data.edge_weight, data.edge_index, n)

        s = self.out_norm(s)

        return F.softplus(self.readout(s))
