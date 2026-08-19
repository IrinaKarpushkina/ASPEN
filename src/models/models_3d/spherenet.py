"""
spherenet.py — SphereNet for per-atom sigma-profile prediction.

Paper:    Liu, Wang, Liu, Lin, Zhang, Oztekin & Ji, "Spherical Message
          Passing for 3D Molecular Graphs", ICLR 2022.
          https://openreview.net/forum?id=givsRXsOt9r (arXiv:
          https://arxiv.org/abs/2102.05013 as supplied by the user)
Official: https://github.com/divelab/DIG
          (`dig/threedgraph/method/spherenet/spherenet.py`,
          `spherenet_utils.py::xyz_to_dat`)

SphereNet's central idea, beyond DimeNet's (distance, angle): every
message additionally carries a TORSION (dihedral) angle, computed from a
4th atom, giving spherical message passing its full local-frame
"distance + angle + torsion" description of 3D structure (Sec. 3 of the
paper) — this is what lets it in principle distinguish local
3D arrangements (e.g. chirality-adjacent environments) that (distance,
angle)-only architectures like DimeNet cannot.

FIDELITY NOTE (documented explicitly, in the same spirit as this repo's
other citation-honest simplifications — see PROVENANCE.md): the official
DIG implementation builds its torsion index and 2D "torsion basis" via a
specific local reference-frame convention (`xyz_to_dat`) that additionally
depends on `torch_sparse`/`torch_scatter`, extensions this repository does
not depend on (see requirements-3d.txt and `src/data/geometry_3d.py`
module docstring). This implementation instead:

  1. Reuses DimeNet++'s (distance, angle) triplet machinery UNCHANGED
     (same `BesselBasisLayer`/`SphericalBasisLayer`/`InteractionPPBlock`-
     style down-projection, see `models_3d/dimenet.py`), so the
     distance+angle channel is identical in spirit to DimeNet++'s.
  2. Adds an explicit TORSION channel: for each (k, j, i) triplet, a 4th
     atom l (bonded to k, see `geometry_3d.build_torsions`) gives a proper
     dihedral angle tau_{lkji} (`geometry_3d.compute_torsion`), embedded
     with a small Fourier basis (cos(n*tau), sin(n*tau) for
     n=0..num_torsional-1) — the standard way to embed a periodic (2*pi)
     scalar, used e.g. by SchNetPack's `torsion` featurizers and by
     several later spherical-harmonics-free re-implementations of
     SphereNet-style dihedral terms.
  3. The torsion embedding GATES the (distance, angle) spherical basis
     multiplicatively (`combined = sbf * torsion_gate`) before the same
     down-project / bilinear aggregation DimeNet++ uses — mirroring the
     official implementation's own combined (dist, angle, torsion) basis
     product, just built from an explicit Fourier torsion embedding
     instead of the official code's specific local-frame spherical
     harmonics. When no 4th atom exists for a triplet (`has_l=False`,
     e.g. a terminal/degree-1 atom), the torsion gate degrades gracefully
     to the identity (angle-only, i.e. exactly DimeNet++'s message for
     that triplet) rather than a fabricated value.

Everything else (edge embedding via our own atom features, atom-wise
readout via `OutputPPBlock`, stopping before molecule-level pooling) is
identical to `models_3d/dimenet.py` — see that file's docstring for the
rationale, which applies here unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer, SphericalBasisLayer, ResidualLayer, OutputPPBlock,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import CUTOFF
from ...data.features_3d import node_feat_dim_3d
from ...data.geometry_3d import compute_torsion
from .dimenet import _EdgeEmbeddingBlock


class TorsionBasisLayer(nn.Module):
    """Fourier embedding of a periodic dihedral angle tau in [-pi, pi]:
    [cos(0*tau)=1, cos(tau), sin(tau), cos(2*tau), sin(2*tau), ...]."""

    def __init__(self, num_torsional: int):
        super().__init__()
        self.num_torsional = num_torsional
        freqs = torch.arange(0, num_torsional, dtype=torch.float)
        self.register_buffer("freqs", freqs)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        angles = tau.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (E, K)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (E, 2K)


class SphereInteractionBlock(nn.Module):
    """DimeNet++-style InteractionPPBlock, with the (distance, angle)
    spherical basis additionally gated by a torsion Fourier embedding
    (see module docstring)."""

    def __init__(self, hidden_channels: int, int_emb_size: int, basis_emb_size: int,
                 num_spherical: int, num_radial: int, num_torsional: int,
                 num_before_skip: int, num_after_skip: int, act):
        super().__init__()
        self.act = act

        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        self.lin_tbf = nn.Linear(2 * num_torsional, int_emb_size)  # torsion gate

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)])

    def forward(self, x, rbf, sbf, tbf, has_l, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        rbf = self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))

        sbf_emb = self.lin_sbf2(self.lin_sbf1(sbf))                   # (n_triplets, int_emb)
        torsion_gate = torch.sigmoid(self.lin_tbf(tbf))               # (n_triplets, int_emb), in (0,1)
        # graceful fallback to angle-only when no 4th atom exists for a triplet:
        torsion_gate = torch.where(has_l.unsqueeze(-1), torsion_gate,
                                   torch.ones_like(torsion_gate))
        combined = sbf_emb * torsion_gate

        x_kj = x_kj[idx_kj] * combined
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return h


class SphereNetSigmaModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_emb_channels: int = 128,
        num_blocks: int = 3,
        int_emb_size: int = 32,
        basis_emb_size: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        num_torsional: int = 4,
        cutoff: float = CUTOFF,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 2,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.cutoff = cutoff
        n_feat = node_feat_dim_3d()
        act = activation_resolver("swish")

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden_channels // 4)
        self.input_proj = nn.Linear(n_feat + hidden_channels // 4, hidden_channels)

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.tbf = TorsionBasisLayer(num_torsional)
        self.emb = _EdgeEmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                         hidden_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])
        self.interaction_blocks = nn.ModuleList([
            SphereInteractionBlock(hidden_channels, int_emb_size, basis_emb_size,
                                   num_spherical, num_radial, num_torsional,
                                   num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.readout = ResidualMLP(hidden_channels, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        pos, z = data.pos, data.z
        i, j = data.edge_index[1], data.edge_index[0]
        idx_i, idx_j, idx_k = data.tri_idx_i, data.tri_idx_j, data.tri_idx_k
        idx_kj, idx_ji = data.tri_idx_kj, data.tri_idx_ji
        idx_l, has_l = data.tor_idx_l, data.tor_has_l
        # Stored as int32 on disk to shrink the cache (see PROVENANCE.md
        # Bug #7); cast to int64 here because torch_geometric's `scatter`
        # (and, on some torch_geometric versions, advanced indexing on
        # CUDA) requires int64 index tensors.
        idx_i, idx_j, idx_k = idx_i.long(), idx_j.long(), idx_k.long()
        idx_kj, idx_ji = idx_kj.long(), idx_ji.long()
        idx_l = idx_l.long()

        z_emb = self.z_embed(z.clamp(max=MAX_Z))
        h_atom = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        dist = data.edge_weight

        # (distance, angle) channel: DimeNet++ convention.
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        # torsion channel: proper dihedral l-k-j-i (0 where has_l is False;
        # masked out via the sigmoid gate falling back to 1 in that case).
        torsion = compute_torsion(pos, idx_l, idx_k, idx_j, idx_i)
        torsion = torch.where(has_l, torsion, torch.zeros_like(torsion))

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        tbf = self.tbf(torsion)

        x = self.emb(h_atom, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, tbf, has_l, idx_kj, idx_ji)
            x = self.dropout(x)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        return F.softplus(self.readout(P))

