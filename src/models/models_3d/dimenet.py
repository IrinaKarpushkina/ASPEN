"""
dimenet.py — DimeNet and DimeNet++ for per-atom sigma-profile prediction.

Papers:
  DimeNet:    Klicpera, Groß & Gunnemann, "Directional Message Passing for
              Molecular Graphs", ICLR 2020. https://arxiv.org/abs/2003.03123
  DimeNet++:  Klicpera, Giri, Margraf & Gunnemann, "Fast and
              Uncertainty-Aware Directional Message Passing for
              Non-Equilibrium Molecules", NeurIPS-W 2020.
              https://arxiv.org/abs/2011.14115 (the "++" referenced
              alongside DimeNet in the same arXiv id in some citation
              lists; PyG documents it under the id above)
Official: https://github.com/gasteigerjo/dimenet (TensorFlow)
PyG implementation used here: torch_geometric.nn.models.dimenet
          (BesselBasisLayer, SphericalBasisLayer, InteractionBlock /
          InteractionPPBlock, OutputBlock / OutputPPBlock — verified
          against PyG source, pyg-team/pytorch_geometric,
          torch_geometric/nn/models/dimenet.py).

Key idea: messages are passed along DIRECTED BONDS/edges (like D-MPNN in
the 2D part of this repo), and each message from (k -> j) into (j -> i) is
additionally modulated by the ANGLE between the two bonds via a spherical
Bessel/harmonic basis (SBF) — this is what makes the architecture sensitive
to bond angles and 3D geometry, not just pairwise distances (unlike
SchNet). DimeNet++ (`pp=True`, default) uses a factorised/down-projected
version of the same interaction block for efficiency, per the paper.

ADAPTATION NOTES (see PROVENANCE.md, "3D benchmark" section, for the full
discussion):

1. Triplets, precomputed. PyG's `DimeNet.forward` builds its own
   `edge_index` via `torch_geometric.nn.radius_graph` and its own triplets
   via `torch_geometric.nn.models.dimenet.triplets()`, which internally
   uses `SparseTensor` (`torch_sparse`) — a compiled extension this repo
   does not depend on (see requirements-3d.txt). Both are precomputed once
   per molecule instead, in `src/data/features_3d.py` /
   `src/data/geometry_3d.py` (radius graph + a `SparseTensor`-free
   reimplementation of `triplets()`), and simply passed in as tensors.

2. Node embedding. DimeNet's own `EmbeddingBlock` starts from a bare
   `nn.Embedding(95, hidden)` on `z` alone. Every other architecture in
   this benchmark (2D and 3D) additionally conditions on the six
   physically-motivated atom features in `src/data/constants_3d.py`
   (electronegativity, vdW radius, Z, mass, H-bond donor/acceptor) via a
   `z_embed + input_proj` pair, so this file reimplements `EmbeddingBlock`
   as `_EdgeEmbeddingBlock`, IDENTICAL to PyG's version except that it
   reads atom embeddings produced by our own `z_embed`/`input_proj`
   instead of instantiating its own bare `nn.Embedding` — controlling how
   much side information about each atom every architecture receives.

3. Atom-wise (not molecule-wise) output. `OutputBlock`/`OutputPPBlock`
   already scatter their (edge-level) input back to ATOM-level via
   `scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')` (see PyG
   source) — this per-atom tensor `P` is exactly what `DimeNet.forward`
   sums over `batch` at its very last line for a molecule-level QM9
   target. We use `P` directly (before that final sum) and feed it to
   this repo's shared `ResidualMLP` head, the same "stop before the
   molecule-level pooling" adaptation already used for GCN/GAT/GATv2/
   GINE/D-MPNN in the 2D part of this repo (see PROVENANCE.md Bug #2) and
   for SchNet (`models_3d/schnet.py`) above.

Hyperparameter note: `num_radial`/`num_spherical` are kept at the ORIGINAL
paper's defaults (6 / 7) rather than this benchmark's shared `N_RBF=32`
(used by SchNet/PaiNN/EGNN/TorchMD-Net/MACE's simple radial embeddings) —
DimeNet's angular basis size grows as `num_spherical * num_radial`, so
reusing a 32-wide RBF here would blow up both parameter count and
triplet-tensor size for no benefit; 6/7 is what the original paper tuned
and what PyG's own QM9-pretrained checkpoints use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer, SphericalBasisLayer,
    InteractionBlock, InteractionPPBlock,
    OutputBlock, OutputPPBlock,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import CUTOFF
from ...data.features_3d import node_feat_dim_3d


class _EdgeEmbeddingBlock(nn.Module):
    """PyG's `EmbeddingBlock`, adapted to read our own atom embeddings
    (see module docstring, point 2) instead of a bare `nn.Embedding(95, .)`.
    """

    def __init__(self, num_radial: int, hidden_channels: int, act):
        super().__init__()
        self.act = act
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

    def forward(self, h_atom: torch.Tensor, rbf: torch.Tensor,
                i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([h_atom[i], h_atom[j], rbf], dim=-1)))


class DimeNetSigmaModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_emb_channels: int = 128,
        num_blocks: int = 3,
        num_bilinear: int = 8,
        int_emb_size: int = 32,
        basis_emb_size: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = CUTOFF,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 2,
        out_dim: int = 51,
        dropout: float = 0.05,
        pp: bool = True,
    ):
        super().__init__()
        self.pp = pp
        self.cutoff = cutoff
        n_feat = node_feat_dim_3d()
        act = activation_resolver("swish")

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden_channels // 4)
        self.input_proj = nn.Linear(n_feat + hidden_channels // 4, hidden_channels)

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.emb = _EdgeEmbeddingBlock(num_radial, hidden_channels, act)

        if pp:
            self.output_blocks = nn.ModuleList([
                OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                              hidden_channels, num_output_layers, act)
                for _ in range(num_blocks + 1)
            ])
            self.interaction_blocks = nn.ModuleList([
                InteractionPPBlock(hidden_channels, int_emb_size, basis_emb_size,
                                   num_spherical, num_radial,
                                   num_before_skip, num_after_skip, act)
                for _ in range(num_blocks)
            ])
        else:
            self.output_blocks = nn.ModuleList([
                OutputBlock(num_radial, hidden_channels, hidden_channels,
                           num_output_layers, act)
                for _ in range(num_blocks + 1)
            ])
            self.interaction_blocks = nn.ModuleList([
                InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                                 num_radial, num_before_skip, num_after_skip, act)
                for _ in range(num_blocks)
            ])

        self.dropout = nn.Dropout(dropout)
        self.readout = ResidualMLP(hidden_channels, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        pos, z = data.pos, data.z
        i, j = data.edge_index[1], data.edge_index[0]  # i=target(center), j=source(neighbor)
        idx_i, idx_j, idx_k = data.tri_idx_i, data.tri_idx_j, data.tri_idx_k
        idx_kj, idx_ji = data.tri_idx_kj, data.tri_idx_ji
        # Stored as int32 on disk to shrink the cache (see PROVENANCE.md
        # Bug #7); cast to int64 here because torch_geometric's `scatter`
        # (and, on some torch_geometric versions, advanced indexing on
        # CUDA) requires int64 index tensors. This is a cheap per-forward
        # cast over a few thousand triplets, not a per-epoch bottleneck.
        idx_i, idx_j, idx_k = idx_i.long(), idx_j.long(), idx_k.long()
        idx_kj, idx_ji = idx_kj.long(), idx_ji.long()

        z_emb = self.z_embed(z.clamp(max=MAX_Z))
        h_atom = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        dist = data.edge_weight
        # DimeNet++ convention: angle formed by (pos_j - pos_k) and (pos_i - pos_j).
        # DimeNet (original) convention: angle formed by (pos_j - pos_i) and (pos_k - pos_i).
        # Both are rotation/translation invariant; we follow PyG's own
        # DimeNet.forward branch exactly (torch_geometric/nn/models/dimenet.py).
        if self.pp:
            pos_jk = pos[idx_j] - pos[idx_k]
            pos_ij = pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        else:
            pos_ji = pos[idx_j] - pos[idx_i]
            pos_ki = pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        x = self.emb(h_atom, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            x = self.dropout(x)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        return F.softplus(self.readout(P))

