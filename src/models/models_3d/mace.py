"""
mace.py — MACE (higher-order equivariant message passing) for per-atom
sigma-profile prediction.

Paper:    Batatia, Kovacs, Simm, Ortner & Csanyi, "MACE: Higher Order
          Equivariant Message Passing Neural Networks for Fast and
          Accurate Force Fields", NeurIPS 2022.
          https://arxiv.org/abs/2206.07697
Official: https://github.com/ACEsuit/mace

MACE's core idea (Sec. 3 of the paper): instead of increasing body order
by stacking more message-passing rounds (as SchNet/DimeNet/EGNN/PaiNN do),
each MACE layer builds HIGH body-order features in a single step, using
the Atomic Cluster Expansion (ACE) formalism — a generalized
Clebsch-Gordan "product basis" that symmetrically self-tensor-products
the layer's pooled 2-body features (their "A" functions, paper Eq. 9-10)
up to a chosen correlation order nu (paper uses nu up to 3).

This benchmark uses `e3nn` (https://e3nn.org, MIT-licensed equivariant
neural network library; see requirements-3d.txt) to build genuine
Clebsch-Gordan tensor products, rather than reimplementing spherical
harmonics/CG coefficients from scratch:

  1. Two-body message ("A" function, paper Eq. 8-10): for edge j->i,
         msg_ij = TensorProduct( h_j, Y_l(r_hat_ij) ; weights = R(d_ij) )
     where Y_l are real spherical harmonics of the edge direction
     (`e3nn.o3.spherical_harmonics`) and R(d_ij) is a learned radial
     function of a Bessel-embedded distance (matching MACE's own
     "radial embedding -> weights of the tensor product" mechanism,
     paper Eq. 8, and analogous to NequIP's interaction block, which the
     paper explicitly builds on). A_i = sum_j msg_ij (paper Eq. 10).
  2. Product basis / higher body order (paper Eq. 11-13, the "B"
     functions): this benchmark computes ONE symmetric self-tensor-product
     of A_i with itself, B_i = TensorProduct(A_i, A_i), which is exactly
     the correlation-order nu=2 case of the paper's generalized-CG product
     basis (a true, rotation-equivariant 3-body correlation feature, since
     A_i itself already aggregates 2-body edge information — squaring it
     gives a genuine 3-body invariant/equivariant descriptor, matching the
     mathematical definition of the ACE product basis at that order).

     FIDELITY NOTE (explicit, documented simplification — same spirit as
     this repo's other adaptations, see PROVENANCE.md): the official MACE
     implementation supports higher correlation orders (nu up to 3, i.e.
     an additional triple self-tensor-product for 4-body features) via a
     custom `SymmetricContraction` module with a weight-sharing/
     normalisation scheme tuned for numerical stability at high nu. This
     benchmark caps correlation order at nu=2 (one product-basis
     multiplication per layer) to keep the implementation auditable with
     plain `e3nn.o3.FullyConnectedTensorProduct` calls; this is a reduced-
     order MACE, not the full paper's default configuration — documented
     here rather than silently presented as identical.
  3. Update: h_i <- Linear( concat[A_i, B_i] ) + Linear(h_i)  (residual),
     projected back to the layer's working irreps. Stack `num_layers`
     such layers (paper defaults to 2).
  4. Readout: the invariant (0e, i.e. rotation-scalar) subspace of the
     final layer's h_i is sliced out and fed to this repo's shared
     ResidualMLP head — every other architecture in this benchmark
     also ends in an invariant per-atom scalar before that same head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import scatter
from e3nn import o3

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import N_RBF, CUTOFF
from ...data.features_3d import node_feat_dim_3d


def _cosine_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(d * torch.pi / cutoff) + 1.0) * (d < cutoff).float()


class MACELayer(nn.Module):
    def __init__(self, h_irreps: o3.Irreps, msg_irreps: o3.Irreps,
                 lmax_sh: int, num_rbf: int, cutoff: float, dropout: float = 0.05):
        super().__init__()
        self.cutoff = cutoff
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax_sh)

        # Two-body message: weighted tensor product h_j (x) Y(r_hat_ij).
        self.tp = o3.FullyConnectedTensorProduct(
            h_irreps, self.sh_irreps, msg_irreps,
            shared_weights=False, internal_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_rbf, 64), nn.SiLU(), nn.Linear(64, self.tp.weight_numel),
        )

        # Body-order-2 product basis: self-tensor-product of the pooled
        # 2-body features A_i (see module docstring, point 2).
        self.tp_product = o3.FullyConnectedTensorProduct(msg_irreps, msg_irreps, msg_irreps)

        self.update_lin = o3.Linear(msg_irreps + msg_irreps, h_irreps)
        self.skip_lin = o3.Linear(h_irreps, h_irreps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, rbf, edge_weight, sh, edge_index, num_nodes):
        j, i = edge_index[0], edge_index[1]
        w = self.radial_mlp(rbf) * _cosine_cutoff(edge_weight, self.cutoff).unsqueeze(-1)
        msg_ij = self.tp(h[j], sh, w)
        A_i = scatter(msg_ij, i, dim=0, dim_size=num_nodes, reduce="sum")

        B_i = self.tp_product(A_i, A_i)

        combined = torch.cat([A_i, B_i], dim=-1)
        h_new = self.update_lin(combined) + self.skip_lin(h)
        return h_new


class MACESigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 96,
        num_layers: int = 2,
        lmax_sh: int = 2,
        num_rbf: int = N_RBF,
        cutoff: float = CUTOFF,
        out_dim: int = 51,
        dropout: float = 0.05,
        hidden_l1: int = None,
        hidden_l2: int = None,
    ):
        """`hidden` is the single tunable width knob (matches every other
        architecture in this benchmark, and lets
        `scripts/count_params.py --auto-tune` — unmodified — retarget this
        model's parameter budget exactly like it does for the others).
        It sets the invariant (0e) channel count directly; the l=1/l=2
        channel counts scale proportionally (hidden/3, hidden/6) unless
        overridden explicitly via `hidden_l1`/`hidden_l2`.
        """
        super().__init__()
        hidden_scalar = hidden
        hidden_l1 = hidden_l1 if hidden_l1 is not None else max(hidden // 3, 8)
        hidden_l2 = hidden_l2 if hidden_l2 is not None else max(hidden // 6, 4)

        n_feat = node_feat_dim_3d()

        self.h_irreps = o3.Irreps(f"{hidden_scalar}x0e + {hidden_l1}x1o + {hidden_l2}x2e")
        self.msg_irreps = o3.Irreps(f"{hidden_scalar}x0e + {hidden_l1}x1o + {hidden_l2}x2e")
        self.n_scalar = hidden_scalar  # size of the invariant (0e) subspace, always listed first

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden_scalar // 4)
        self.input_proj = nn.Linear(n_feat + hidden_scalar // 4, hidden_scalar)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_rbf)

        self.layers = nn.ModuleList([
            MACELayer(self.h_irreps, self.msg_irreps, lmax_sh, num_rbf, cutoff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_scalar) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = ResidualMLP(hidden_scalar, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        s0 = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        n = s0.size(0)

        # h starts as a pure-scalar (0e) irrep tensor; l=1/l=2 components
        # are initialised to zero (no orientation information at input,
        # exactly like PaiNN/TorchMD-Net's vector channel, since raw atom
        # features carry no directional information before the first
        # message).
        h = torch.zeros(n, self.h_irreps.dim, device=s0.device, dtype=s0.dtype)
        h[:, : self.n_scalar] = s0

        edge_index = data.edge_index
        j, i = edge_index[0], edge_index[1]
        r_ij = data.pos[i] - data.pos[j]
        sh = o3.spherical_harmonics(
            self.layers[0].sh_irreps, r_ij, normalize=True, normalization="component",
        )
        rbf = self.distance_expansion(data.edge_weight)

        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, rbf, data.edge_weight, sh, edge_index, n)
            h_scalar = norm(self.dropout(h[:, : self.n_scalar]))
            h = torch.cat([h_scalar, h[:, self.n_scalar:]], dim=-1)

        s_final = h[:, : self.n_scalar]
        return F.softplus(self.readout(s_final))
