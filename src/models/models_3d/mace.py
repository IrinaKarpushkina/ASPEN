"""
mace.py — MACE (higher-order equivariant message passing) for per-atom
sigma-profile prediction.

Paper:    Batatia, Kovacs, Simm, Ortner & Csanyi, "MACE: Higher Order
          Equivariant Message Passing Neural Networks for Fast and
          Accurate Force Fields", NeurIPS 2022.
          https://arxiv.org/abs/2206.07697
Official: https://github.com/ACEsuit/mace
          (`mace.modules.blocks.RealAgnosticResidualInteractionBlock`,
          `mace.modules.blocks.EquivariantProductBasisBlock`)

REVISION HISTORY:
  v1: hand-rolled two-body message (plain `e3nn.o3.FullyConnectedTensorProduct`)
      and a self-tensor-product approximation of the product basis
      (correlation-order 2 only), explicitly documented as reduced-order.
  v2: replaced the product-basis approximation with the OFFICIAL
      `EquivariantProductBasisBlock` (correlation=3), but kept the
      hand-rolled two-body message/skip-connection around it.
  v3 (this version): replaced the remaining hand-rolled pieces with the
      OFFICIAL `RealAgnosticResidualInteractionBlock` too, fixing three
      further fidelity gaps found on review:
        (a) the official interaction block divides its aggregated
            message by `avg_num_neighbors` (a normalization constant
            independent of any one molecule's local density); v2 had no
            such normalization.
        (b) the official residual/"self-connection" term (`sc`) is
            SPECIES-CONDITIONED -- a tensor product of the node's own
            features with its one-hot atom-type attribute
            (`skip_tp(node_feats, node_attrs)`), not a plain
            atom-type-agnostic linear map. v2's `skip_lin` was a plain
            `o3.Linear`, identical for every element.
        (c) v1/v2 additionally used TWO separate, differently-shaped
            irreps ("h_irreps" for the residual channel, "msg_irreps"
            for the message/product-basis channel, with the former
            given non-uniform per-l multiplicities: hidden, hidden/3,
            hidden/6). The official architecture uses a SINGLE, uniform
            hidden_irreps space throughout the whole network (required
            for `reshape_irreps`/symmetric contraction to be well-defined
            in the first place -- see v2's own fix). This version
            collapses to that single-irreps-space design, matching the
            official architecture's actual structure rather than an
            approximation of it.
      With (a)-(c) fixed, MACELayer below is now a thin wrapper around
      two OFFICIAL mace-torch building blocks used back-to-back exactly
      as the official codebase does, rather than a parallel
      reimplementation of what they do.

WHAT STILL DIFFERS FROM THE OFFICIAL ARCHITECTURE, AND WHY (these are
this benchmark's own shared, deliberately-applied conventions across ALL
9 architectures -- not MACE-specific fidelity gaps, and changing them
for MACE alone would break the controlled comparison the benchmark is
built around, see PROVENANCE.md / Dwivedi et al. 2022):
  - Radial basis: `GaussianSmearing` (this benchmark's shared choice,
    also used by SchNet/PaiNN/TorchMD-Net/MACE here) rather than the
    official `RadialEmbeddingBlock`'s Bessel-with-polynomial-cutoff --
    every architecture in this benchmark is given the same
    distance-embedding "budget" (`constants_3d.N_RBF`), isolating each
    architecture's own update/message rule from its choice of radial
    basis. The cutoff ENVELOPE itself (a cosine cutoff, applied
    separately from the raw radial features, exactly mirroring
    `RadialEmbeddingBlock(apply_cutoff=False)`'s documented usage
    pattern of returning `(radial, cutoff)` for interaction blocks that
    take `cutoff` as its own forward argument) is unchanged from the
    official mechanism.
  - Atom input embedding: this benchmark's shared `input_proj` over
    engineered atom features + a separate Z-embedding (identical
    convention to every other architecture here), rather than the
    official `LinearNodeEmbeddingBlock`'s pure one-hot-species linear
    map -- so that all 9 architectures see the same node-level
    information budget. The one-hot atom-type attribute the official
    architecture ALSO needs (for `node_attrs`, feeding the
    species-conditioned skip and the product basis) is still computed
    separately and exactly as the official code expects -- only the
    scalar FEATURE embedding (`h`'s initial value) differs.
  - No atomic-energies / scale-shift readout block: those are
    energy-prediction-specific components (this benchmark predicts
    atomic sigma-profiles, not energies), not applicable here regardless
    of fidelity.
  - No periodic boundary conditions / LAMMPS ghost-atom handling: not
    applicable to isolated molecules.
  - `avg_num_neighbors` (used by the official interaction block to
    normalize message magnitude) is a fixed constant here rather than
    precomputed from the true training-set average node degree under
    this benchmark's shared radius graph (`constants_3d.CUTOFF`,
    `MAX_NUM_NEIGHBORS`) -- see `MACESigmaModel`'s `avg_num_neighbors`
    argument; recalibrate it via a quick one-off script before final
    training runs (average `edge_index.shape[1] / num_nodes` over a
    representative sample of the training set).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import scatter
from e3nn import o3

try:
    from mace.modules.blocks import (
        EquivariantProductBasisBlock,
        RealAgnosticResidualInteractionBlock,
    )
except ImportError as e:
    raise ImportError(
        "mace.py requires the official 'mace-torch' package (see this "
        "file's module docstring for why hand-rolled substitutes were "
        "removed). Run: pip install mace-torch"
    ) from e

from ..layers import ResidualMLP
from ...data.constants import MAX_Z, ELEMENT_TO_Z
from ...data.constants_3d import N_RBF, CUTOFF
from ...data.features_3d import node_feat_dim_3d

# This dataset's actual element vocabulary (all three splits combined),
# found via check_element_coverage.py -- 25 distinct elements. Kept local
# to this file (not added to constants.py, which is intentionally
# generic/architecture-agnostic, see that file's own docstring) since it
# is specific to what EquivariantProductBasisBlock's per-element weights
# need. Re-run check_element_coverage.py and update this list if the
# dataset's composition ever changes.
ELEMENTS_PRESENT = [
    'As', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C', 'Cl', 'F', 'Ga', 'Ge', 'H', 'I',
    'In', 'N', 'O', 'P', 'Po', 'S', 'Sb', 'Se', 'Si', 'Sr', 'Te', 'Xe',
]


def _cosine_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(d * torch.pi / cutoff) + 1.0) * (d < cutoff).float()


class MACELayer(nn.Module):
    """One MACE layer: the official interaction block (species-conditioned
    residual, avg-num-neighbors-normalized message) followed by the
    official product-basis block (correlation=3 symmetric contraction),
    exactly as the official codebase chains them."""

    def __init__(self, node_attrs_irreps: o3.Irreps, hidden_irreps: o3.Irreps,
                 edge_attrs_irreps: o3.Irreps, edge_feats_irreps: o3.Irreps,
                 avg_num_neighbors: float, num_elements: int, correlation: int = 3):
        super().__init__()
        self.interaction = RealAgnosticResidualInteractionBlock(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=hidden_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=True,  # species-conditioned residual, see module docstring
        )

    def forward(self, h, node_attrs, edge_attrs, edge_feats, edge_index, cutoff_env):
        message, sc = self.interaction(
            node_attrs=node_attrs, node_feats=h, edge_attrs=edge_attrs,
            edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff_env,
        )
        h_new = self.product(message, sc, node_attrs)
        return h_new


class MACESigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        num_layers: int = 2,
        lmax_sh: int = 2,
        num_rbf: int = N_RBF,
        cutoff: float = CUTOFF,
        out_dim: int = 51,
        dropout: float = 0.05,
        num_elements: int = 25,       # see module docstring / ELEMENTS_PRESENT
        correlation: int = 3,         # paper's own default
        avg_num_neighbors: float = 16.0,  # PLACEHOLDER -- recalibrate, see docstring
    ):
        """`hidden` is the single tunable width knob (matches every other
        architecture in this benchmark, and lets
        `scripts/count_params.py --auto-tune` — unmodified — retarget this
        model's parameter budget). It sets the UNIFORM per-component
        multiplicity of `hidden_irreps` (same channel count for 0e/1o/2e)
        -- required by the official symmetric-contraction machinery, see
        module docstring's REVISION HISTORY (c).
        """
        super().__init__()
        n_feat = node_feat_dim_3d()

        self.hidden_irreps = o3.Irreps(f"{hidden}x0e + {hidden}x1o + {hidden}x2e")
        self.n_scalar = hidden  # multiplicity of the 0e (invariant) block, listed first
        self.num_elements = num_elements
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax_sh)
        self.node_attrs_irreps = o3.Irreps(f"{num_elements}x0e")
        self.edge_feats_irreps = o3.Irreps(f"{num_rbf}x0e")
        self.cutoff = cutoff

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_rbf)

        self.layers = nn.ModuleList([
            MACELayer(self.node_attrs_irreps, self.hidden_irreps,
                     self.sh_irreps, self.edge_feats_irreps,
                     avg_num_neighbors=avg_num_neighbors,
                     num_elements=num_elements, correlation=correlation)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Explicit Z -> [0, num_elements) lookup, built from this dataset's
        # real element vocabulary (ELEMENTS_PRESENT), NOT a raw-Z clamp --
        # atomic numbers are not a dense range (H=1, C=6, O=8, ... has
        # gaps), so a clamp would silently collide unrelated elements into
        # the same one-hot slot.
        z_to_index = torch.full((MAX_Z + 1,), -1, dtype=torch.long)
        present_z = sorted(ELEMENT_TO_Z[e] for e in ELEMENTS_PRESENT)
        assert len(present_z) == num_elements, (
            f"num_elements={num_elements} does not match len(ELEMENTS_PRESENT)="
            f"{len(present_z)} -- update one or the other (see module docstring)."
        )
        for idx, z in enumerate(present_z):
            z_to_index[z] = idx
        self.register_buffer("z_to_index", z_to_index)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_clamped = data.z.clamp(max=MAX_Z)
        z_emb = self.z_embed(z_clamped)
        s0 = self.input_proj(torch.cat([data.x, z_emb], dim=-1))
        n = s0.size(0)

        # h starts as a pure-scalar (0e) irrep tensor; l=1/l=2 components
        # are initialised to zero (no orientation information at input,
        # exactly like PaiNN/TorchMD-Net's vector channel).
        h = torch.zeros(n, self.hidden_irreps.dim, device=s0.device, dtype=s0.dtype)
        h[:, : self.n_scalar] = s0

        node_attrs = F.one_hot(
            self.z_to_index[z_clamped], num_classes=self.num_elements
        ).to(dtype=s0.dtype)

        edge_index = data.edge_index
        j, i = edge_index[0], edge_index[1]
        r_ij = data.pos[i] - data.pos[j]
        edge_attrs = o3.spherical_harmonics(
            self.sh_irreps, r_ij, normalize=True, normalization="component",
        )
        edge_feats = self.distance_expansion(data.edge_weight)  # raw radial features, no envelope
        cutoff_env = _cosine_cutoff(data.edge_weight, self.cutoff).unsqueeze(-1)  # applied separately,
        # mirroring the official RadialEmbeddingBlock(apply_cutoff=False)
        # usage pattern (radial features and cutoff envelope passed
        # separately into the interaction block, see module docstring).

        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, node_attrs, edge_attrs, edge_feats, edge_index, cutoff_env)
            h_scalar = norm(self.dropout(h[:, : self.n_scalar]))
            h = torch.cat([h_scalar, h[:, self.n_scalar:]], dim=-1)

        s_final = h[:, : self.n_scalar]
        return F.softplus(self.readout(s_final))
