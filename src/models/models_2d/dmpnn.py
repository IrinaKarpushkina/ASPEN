"""
dmpnn.py — Directed MPNN (D-MPNN / Chemprop) for per-atom sigma-profiles.

Paper:    Yang et al., "Analyzing Learned Molecular Representations for
          Property Prediction", J. Chem. Inf. Model. 2019.
          https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237
          arXiv: https://arxiv.org/abs/1904.01561
Official: https://github.com/chemprop/chemprop

Key idea of D-MPNN: messages are passed along DIRECTED BONDS (edge-centric),
not atoms. For an edge v->w, the incoming-message aggregation explicitly
EXCLUDES the reverse edge w->v ("reverse-edge masking"), which removes the
"totter" (immediate back-and-forth) effect that plain atom-centric MPNNs
have.

    h_{vw}^{t+1} = ReLU( W_i * init_{vw} + W_m * sum_{k->v, k != w} h_{kv}^t )

After T steps, atom representations are read out as:

    h_v = ReLU( W_a * cat( x_v, sum_{w->v} h_{wv}^T ) )

Chemprop v2 supports an explicit atom-level prediction mode (as opposed to
its usual sum-to-molecule + FFN readout for whole-molecule properties);
we use the equivalent here — the atom readout h_v above IS the final
per-atom representation, with no additional sum-to-molecule step.

FIDELITY NOTE (see PROVENANCE.md): the previous benchmark version added a
shared, hand-written global mean+max pooling block after this atom
readout (identical to five other architectures) — removed in this
version. The D-MPNN core (this file's DMPNNConv + reverse-edge masking)
is unchanged and is a faithful reimplementation of the equations above;
only the un-cited, architecture-agnostic pooling add-on was removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim, edge_feat_dim


def build_reverse_index(edge_index: torch.Tensor) -> torch.Tensor:
    """For each edge e=(i->j), finds the reverse edge e'=(j->i) in
    edge_index. Returns reverse_idx (E,) with reverse_idx[e] = e'.

    O(E) dictionary lookup. Assumes every edge (i,j) has a matching
    reverse edge (j,i) in edge_index — true here since
    features.build_edge_index() always adds both directions of a bond.
    """
    row = edge_index[0].tolist()
    col = edge_index[1].tolist()

    # Map each directed edge (i, j) to its own index...
    edge_to_idx = {}
    for e, (i, j) in enumerate(zip(row, col)):
        edge_to_idx[(i, j)] = e

    # ...then, for edge e=(i, j), look up the edge going the OTHER way, (j, i).
    E = edge_index.size(1)
    reverse_idx = torch.zeros(E, dtype=torch.long, device=edge_index.device)
    for e, (i, j) in enumerate(zip(row, col)):
        reverse_idx[e] = edge_to_idx.get((j, i), e)  # fallback: self (should not occur)

    return reverse_idx


class DMPNNConv(nn.Module):
    """One step of directed message passing with reverse-edge masking.

    agg_{v\\w} = sum_{k->v, k != w} h_{kv}  ==  sum_{k->v} h_{kv} - h_{wv}
    h_{vw}^{t+1} = ReLU( W_i * h_{vw}^0 + W_m * agg_{v\\w} )
    """

    def __init__(self, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.W_m = nn.Linear(hidden, hidden, bias=False)
        self.W_i = nn.Linear(hidden, hidden, bias=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.W_m.weight)
        nn.init.xavier_uniform_(self.W_i.weight)

    def forward(
        self,
        h_edge: torch.Tensor,       # (E, hidden) current bond states
        h_init: torch.Tensor,       # (E, hidden) initial bond states
        edge_index: torch.Tensor,   # (2, E)
        reverse_idx: torch.Tensor,  # (E,)
        num_nodes: int,
    ) -> torch.Tensor:
        agg_full = scatter(h_edge, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        agg_edge = agg_full[edge_index[0]] - h_edge[reverse_idx]

        h_new = self.act(self.norm(self.W_i(h_init) + self.W_m(agg_edge)))
        return self.drop(h_new)


class DMPNNSigmaModel(nn.Module):
    """D-MPNN for per-atom sigma-profile prediction.

    Pipeline:
      1. Atom features: x + Z-embedding -> hidden
      2. Bond init: cat(atom_feat_source, edge_attr) -> hidden
      3. T steps of directed message passing (reverse-edge masking,
         weights shared across steps, matching original Chemprop)
      4. Atom readout: x_atom + sum(incoming bond states)
      5. ResidualMLP -> softplus -> (N_atoms, 51)
    """

    def __init__(
        self,
        hidden: int = 243,
        num_steps: int = 4,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_steps = num_steps
        n_feat = node_feat_dim()
        e_feat = edge_feat_dim()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.atom_proj = nn.Linear(n_feat + hidden // 4, hidden)
        self.atom_norm = nn.LayerNorm(hidden)

        self.bond_init = nn.Linear(hidden + e_feat, hidden, bias=False)
        self.bond_init_norm = nn.LayerNorm(hidden)

        self.conv = DMPNNConv(hidden, dropout=dropout)

        self.atom_readout = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        N = data.z.size(0)

        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x_atom = self.atom_norm(F.relu(self.atom_proj(torch.cat([data.x, z_emb], dim=-1))))

        src = data.edge_index[0]
        h_init = F.relu(
            self.bond_init_norm(self.bond_init(torch.cat([x_atom[src], data.edge_attr], dim=-1)))
        )
        h_init = torch.nan_to_num(h_init, nan=0.0, posinf=1.0, neginf=-1.0)

        if not hasattr(data, "_rev_idx") or data._rev_idx is None:
            data._rev_idx = build_reverse_index(data.edge_index)
        rev_idx = data._rev_idx

        h_bond = h_init
        for _ in range(self.num_steps):
            h_bond = self.conv(h_bond, h_init, data.edge_index, rev_idx, N)
            h_bond = torch.nan_to_num(h_bond, nan=0.0)

        bond_agg = scatter(h_bond, data.edge_index[1], dim=0, dim_size=N, reduce="sum")
        h = self.atom_readout(torch.cat([x_atom, bond_agg], dim=-1))

        return F.softplus(self.readout(h))
