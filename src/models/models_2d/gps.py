"""
gps.py — GraphGPS for per-atom sigma-profile prediction.

Paper:    Rampasek et al., "Recipe for a General, Powerful, Scalable Graph
          Transformer", NeurIPS 2022. https://arxiv.org/abs/2205.12454
Official: https://github.com/rampasek/GraphGPS
Layer used here: torch_geometric.nn.GPSConv (PyG's maintained
          implementation), following the official PyG example:
          https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py

Architecture (one GPS layer, per the paper/example):
  1. Positional Encoding (RWPE) — precomputed in dataset.py from edge_index,
     concatenated onto the input node features.
  2. Local MPNN (GINEConv) — local message passing.
  3. Global multi-head self-attention — every atom attends to every other
     atom IN THE SAME MOLECULE (batched via PyG's internal to_dense_batch).
  4. Feed-forward network inside GPSConv.

FIDELITY NOTE (see PROVENANCE.md): this model required no change in this
rewrite. GPSConv already provides molecule-wide context through its
built-in global attention — exactly as specified by the paper — so this
was the one architecture in the original benchmark that did NOT need (and
did not have) the shared, hand-written global mean+max pooling block found
in the other five models. It is kept unchanged here as the reference case:
"what a model with a genuine, paper-specified global-context mechanism
looks like".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GPSConv, GINEConv

from ..layers import ResidualMLP
from ...data.constants import MAX_Z, WALK_LENGTH
from ...data.features import node_feat_dim, edge_feat_dim


class GPSSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
        walk_length: int = WALK_LENGTH,
    ):
        super().__init__()
        self.hidden = hidden
        self.walk_length = walk_length
        n_feat = node_feat_dim()
        e_feat = edge_feat_dim()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.pe_proj = nn.Linear(walk_length, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4 + hidden // 4, hidden)

        self.convs = nn.ModuleList([
            GPSConv(
                channels=hidden,
                conv=GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                        nn.SiLU(), nn.Linear(hidden, hidden),
                    ),
                    edge_dim=e_feat,
                ),
                heads=heads,
                dropout=dropout,
                norm="layer_norm",
                attn_kwargs={"dropout": dropout},
            )
            for _ in range(num_layers)
        ])
        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        pe = getattr(data, "pe", None)
        if pe is None:
            pe = torch.zeros(data.x.size(0), self.walk_length, device=data.x.device)
        else:
            pe = pe.to(data.x.device)

        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        pe_emb = self.pe_proj(pe)
        x = self.input_proj(torch.cat([data.x, z_emb, pe_emb], dim=-1))

        h = x
        for conv in self.convs:
            h = conv(h, data.edge_index, data.batch, edge_attr=data.edge_attr)

        return F.softplus(self.readout(h))
