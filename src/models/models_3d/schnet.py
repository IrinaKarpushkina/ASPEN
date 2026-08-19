"""
schnet.py — SchNet for per-atom sigma-profile prediction.

Paper:    Schutt et al., "SchNet: A Continuous-filter Convolutional Neural
          Network for Modeling Quantum Interactions", NeurIPS 2017.
          https://papers.nips.cc/paper/6700
Official: https://github.com/atomistic-machine-learning/SchNet
PyG implementation used here: torch_geometric.nn.models.schnet
          (InteractionBlock, CFConv, GaussianSmearing, ShiftedSoftplus —
          verified against PyG source, pyg-team/pytorch_geometric,
          torch_geometric/nn/models/schnet.py).

Architecture (per the paper): atomwise embedding -> T interaction blocks,
each a continuous-filter convolution (CFConv) that filters neighbour
features by an MLP of the (Gaussian-expanded) interatomic distance,
gated by a smooth cosine cutoff envelope -> atomwise output network.
SchNet's own QM9 pipeline pools this atomwise representation to a
molecule-level scalar only at the very last step (`self.lin1/act/lin2`
followed by `self.readout(...)`, see PyG's `SchNet.forward`). Everything
before that final pooling is already a genuine per-atom representation.

ADAPTATION NOTE (matches this repo's established pattern for AttentiveFP/
GPS — see PROVENANCE.md): rather than reimplementing CFConv/InteractionBlock
ourselves, we instantiate PyG's own `InteractionBlock` and `GaussianSmearing`
directly and call them in sequence, stopping BEFORE SchNet's final
molecule-level pooling — i.e. we keep `h` as a per-atom tensor and feed it
to this repo's shared `ResidualMLP` head, instead of calling
`self.readout(h, batch)`. We do not use `RadiusInteractionGraph` (which
calls `torch_geometric.nn.radius_graph`, requiring the compiled `pyg-lib`/
`torch_cluster` extension this repo does not depend on) — the radius graph
is instead precomputed once per molecule in `src/data/features_3d.py`
(`geometry_3d.radius_graph_single`) and passed in directly, exactly as
CFConv's own `forward(x, edge_index, edge_weight, edge_attr)` signature
expects.

Node embedding: SchNet's original embedding is a bare `nn.Embedding(Z)`.
This benchmark additionally concatenates the same six physically-motivated,
non-topological atom features used by every 3D architecture here (see
`src/data/constants_3d.py`) before projecting to `hidden_channels` — the
identical pattern already used for every 2D model in this repo
(`z_embed` + `input_proj`, see e.g. `src/models/models_2d/gcn.py`), so
that "how much side information about the atom every model gets" is
controlled across architectures, and only the message-passing scheme
differs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import N_RBF, CUTOFF
from ...data.features_3d import node_feat_dim_3d


class SchNetSigmaModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 256,
        num_filters: int = 256,
        num_interactions: int = 4,
        num_gaussians: int = N_RBF,
        cutoff: float = CUTOFF,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden_channels
        n_feat = node_feat_dim_3d()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden_channels // 4)
        self.input_proj = nn.Linear(n_feat + hidden_channels // 4, hidden_channels)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])
        self.dropout = nn.Dropout(dropout)

        self.readout = ResidualMLP(hidden_channels, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        h = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        edge_index, edge_weight = data.edge_index, data.edge_weight
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + self.dropout(interaction(h, edge_index, edge_weight, edge_attr))

        return F.softplus(self.readout(h))
