"""
spherenet.py — SphereNet for per-atom sigma-profile prediction.

Paper:    Liu, Wang, Liu, Lin, Zhang, Oztekin & Ji, "Spherical Message
          Passing for 3D Molecular Graphs", ICLR 2022.
          https://openreview.net/forum?id=givsRXsOt9r
Official: https://github.com/divelab/DIG
          (`dig/threedgraph/method/spherenet/spherenet.py`,
          `dig/threedgraph/method/spherenet/features.py`,
          `dig/threedgraph/utils/geometric_computing.py::xyz_to_dat`)

REVISION NOTE: an earlier version of this file avoided `torch_sparse`/
`torch_scatter` (not otherwise needed by this benchmark) by (a)
approximating the official torsion angle with a different geometric
quantity (a 4th atom bonded to k, rather than the official's "second
neighbour of the middle atom j, minimum angle over all such neighbours"
definition) and (b) embedding it with a hand-rolled Fourier basis
instead of the official's real-spherical-harmonic/Bessel product basis.
Both are now replaced with verbatim ports of the official code
(`src/data/geometry_spherenet.py::xyz_to_dat`,
`src/data/spherenet_features.py::{dist_emb,angle_emb,torsion_emb}`),
computed at forward-time directly from this benchmark's shared
edge_index/pos (not from the dataset's precomputed
`tri_idx_*`/`tor_idx_l`/`tor_has_l` fields, which every OTHER
DimeNet-family architecture in this benchmark still uses unchanged).
`torch_scatter`/`torch_sparse` are now benchmark dependencies for this
architecture only (see requirements-3d.txt).

The interaction block below now mirrors the official `update_e` exactly:
distance, angle, and torsion bases are projected and multiplied together
directly (no sigmoid gate, no graceful "angle-only" fallback for
triplets lacking a further neighbour -- the official code has neither;
see `geometry_spherenet.xyz_to_dat`'s docstring for what happens to such
triplets under the official torsion definition).

WHAT STILL DIFFERS FROM THE OFFICIAL ARCHITECTURE, AND WHY (scope/
integration choices, not correctness gaps in the ported math):
  - Atom input embedding: official SphereNet embeds raw atomic number
    via a single `nn.Embedding(95, hidden_channels)` (`init.forward`).
    This benchmark instead uses its own shared atom-feature convention
    (`input_proj` over engineered features + a separate Z-embedding),
    identical to every other architecture here, so that all 9
    architectures see the same node-level information budget.
  - Output/readout: this benchmark reuses `torch_geometric.nn.models.
    dimenet.OutputPPBlock` (as established by `models_3d/dimenet.py`)
    rather than the official SphereNet's own `update_v` module. The two
    are structurally analogous (up-project -> stack of hidden
    layers+activation -> final linear) but not the same code; this
    predates the present revision and was not in scope for it.
  - Per-atom, not per-molecule: this benchmark's task is atom-resolved
    regression, so there is no final `update_u` (molecule-level) pooling
    step -- the per-atom `OutputPPBlock` output is summed across
    interaction blocks and used directly, as in `models_3d/dimenet.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import OutputPPBlock, ResidualLayer
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import CUTOFF
from ...data.features_3d import node_feat_dim_3d
from ...data.geometry_spherenet import xyz_to_dat
from ...data.spherenet_features import dist_emb, angle_emb, torsion_emb
from .dimenet import _EdgeEmbeddingBlock


class SphereInteractionBlock(nn.Module):
    """Verbatim structural port of the official `update_e` (distance,
    angle, torsion bases projected and multiplied directly, no gating)."""

    def __init__(self, hidden_channels: int, int_emb_size: int,
                 basis_emb_size_dist: int, basis_emb_size_angle: int,
                 basis_emb_size_torsion: int, num_spherical: int, num_radial: int,
                 num_before_skip: int, num_after_skip: int, act):
        super().__init__()
        self.act = act

        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)

        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)

        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial,
                                basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)])

    def forward(self, x, rbf, sbf, tbf, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        rbf = self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf2(self.lin_sbf1(sbf))
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t2(self.lin_t1(tbf))
        x_kj = x_kj * t

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
        basis_emb_size_dist: int = 8,
        basis_emb_size_angle: int = 8,
        basis_emb_size_torsion: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
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

        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)

        # separate BesselBasisLayer-equivalent (`dist_emb` above) feeds the
        # interaction blocks' (angle, torsion) bases; the atom-embedding /
        # output-block radial features reuse the same PyG BesselBasisLayer
        # convention `models_3d/dimenet.py` uses (num_radial-dim rbf,
        # envelope built in), for consistency with this benchmark's other
        # DimeNet-family models.
        from torch_geometric.nn.models.dimenet import BesselBasisLayer
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.emb = _EdgeEmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                         hidden_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])
        self.interaction_blocks = nn.ModuleList([
            SphereInteractionBlock(hidden_channels, int_emb_size,
                                   basis_emb_size_dist, basis_emb_size_angle,
                                   basis_emb_size_torsion, num_spherical, num_radial,
                                   num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.readout = ResidualMLP(hidden_channels, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        pos, z = data.pos, data.z
        num_nodes = pos.size(0)

        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(
            pos, data.edge_index, num_nodes, use_torsion=True
        )

        z_emb = self.z_embed(z.clamp(max=MAX_Z))
        h_atom = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        rbf = self.rbf(dist)
        sbf = self.angle_emb(dist, angle, idx_kj)
        tbf = self.torsion_emb(dist, angle, torsion, idx_kj)

        x = self.emb(h_atom, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=num_nodes)

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, tbf, idx_kj, idx_ji)
            x = self.dropout(x)
            P = P + output_block(x, rbf, i, num_nodes=num_nodes)

        return F.softplus(self.readout(P))
