"""
attentive_fp.py — AttentiveFP for per-atom sigma-profile prediction.

Paper:    Xiong et al., "Pushing the Boundaries of Molecular Representation
          for Drug Discovery with the Graph Attention Mechanism",
          J. Med. Chem. 2020. https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
Official reference implementation:
          https://github.com/OpenDrugAI/AttentiveFP
PyG implementation used here:
          torch_geometric.nn.models.AttentiveFP
          (verified against PyG source, pyg-team/pytorch_geometric,
          torch_geometric/nn/models/attentive_fp.py)

PyG's AttentiveFP.forward has two stages:
  1. Atom embedding:   lin1 -> gate_conv (GATEConv, uses edge_attr) -> gru
                        -> [atom_convs (GATConv) -> atom_grus (GRUCell)] x (L-1)
  2. Molecule embedding: a "super-node" readout — every atom is connected to
                        a single virtual molecule-node via mol_conv (a GATConv
                        with add_self_loops=False), followed by mol_gru, then
                        lin2 projects to the graph-level output.

FIDELITY NOTE (see PROVENANCE.md):
The previous benchmark version used stage 1 (atom embedding) only, then
discarded stage 2 entirely and bolted on a generic, hand-written global
mean+max pooling block shared with five other architectures. That throws
away exactly the part of AttentiveFP that is specific and citable (the
super-node attention readout) in favour of a mechanism the paper never
proposed.

This version instead REUSES the model's own stage-2 modules (mol_conv,
mol_gru, both instantiated by PyG's AttentiveFP exactly as in the official
implementation) to compute one molecule-level embedding per graph via the
paper's own super-node attention mechanism, and broadcasts it back to every
atom before the final per-atom head. This is the "adapt for per-atom
output" analogue of what the original model does for whole-molecule output
(lin2 in the original just maps the molecule embedding to the target dim;
here we concatenate it back onto every atom instead of predicting from it
alone, since our task is per-atom, not per-molecule).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import AttentiveFP

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.features import node_feat_dim, edge_feat_dim


class AtomAttentiveFPModel(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 4,
        num_timesteps: int = 2,
        out_dim: int = 51,
        dropout: float = 0.05,
        use_molecule_context: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.use_molecule_context = use_molecule_context
        n_feat = node_feat_dim()
        e_feat = edge_feat_dim()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        # We instantiate PyG's official AttentiveFP wholesale (including its
        # mol_conv/mol_gru/lin2 modules) and reuse its internal layers below,
        # rather than reimplementing GATEConv/attention ourselves.
        self.afp = AttentiveFP(
            in_channels=hidden, hidden_channels=hidden,
            out_channels=hidden, edge_dim=e_feat,
            num_layers=num_layers, num_timesteps=num_timesteps,
            dropout=dropout,
        )

        head_in = hidden * 2 if use_molecule_context else hidden
        self.readout = ResidualMLP(head_in, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        x = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        # ── Stage 1: atom embedding (identical to PyG's AttentiveFP.forward) ──
        x = F.leaky_relu_(self.afp.lin1(x))
        h = self.afp.gate_conv(x, data.edge_index, data.edge_attr)
        h = F.elu_(h)
        h = F.dropout(h, p=self.afp.dropout, training=self.training)
        x = self.afp.gru(h, x).relu_()

        for conv, gru in zip(self.afp.atom_convs, self.afp.atom_grus):
            h = conv(x, data.edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.afp.dropout, training=self.training)
            x = gru(h, x).relu()

        if not self.use_molecule_context:
            return F.softplus(self.readout(x))

        # ── Stage 2: molecule embedding via the model's OWN super-node
        # attention readout (mol_conv/mol_gru), copied verbatim from PyG's
        # official AttentiveFP.forward (pyg-team/pytorch_geometric,
        # torch_geometric/nn/models/attentive_fp.py), then broadcast back
        # to every atom instead of being consumed by lin2 alone. This is
        # architecture-specific global context (attention-based super-node),
        # not a generic shared pooling block.
        row = torch.arange(data.batch.size(0), device=data.batch.device)
        super_edge_index = torch.stack([row, data.batch], dim=0)
        out = global_add_pool(x, data.batch).relu_()
        for _ in range(self.afp.num_timesteps):
            h = F.elu_(self.afp.mol_conv((x, out), super_edge_index))
            h = F.dropout(h, p=self.afp.dropout, training=self.training)
            out = self.afp.mol_gru(h, out).relu_()

        mol_context = out[data.batch]  # broadcast molecule embedding back to atoms
        h_final = torch.cat([x, mol_context], dim=-1)

        return F.softplus(self.readout(h_final))
