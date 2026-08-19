"""
unimol.py — Uni-Mol (SE(3)-invariant 3D Transformer) for per-atom
sigma-profile prediction.

Paper:    Zhou, Gao, Ding, Zheng, Xu, Wei, Zhang & Ke, "Uni-Mol: A
          Universal 3D Molecular Representation Learning Framework",
          ICLR 2023 (OpenReview submission referenced by the user:
          https://openreview.net/forum?id=6K2RM6wVqKu).
Official: https://github.com/deepmodeling/Uni-Mol

Architecture (paper Sec. 3.1-3.2): unlike every other 3D model in this
benchmark, Uni-Mol is NOT a cutoff-radius message-passing network — it is
a Transformer that attends over EVERY pair of atoms in the molecule (no
neighbour cutoff), with the geometry injected as a PAIR-WISE ATTENTION
BIAS derived from interatomic distances (a Gaussian-kernel pair
representation, paper Sec. 3.1, "SE(3) Transformer Encoder"), and that
pair representation is itself updated, layer by layer, from the
attention maps of the previous layer (paper Fig. 2 / Sec. 3.1, "pair
representation update").

This benchmark therefore does NOT use the shared radius graph
(`edge_index`/`edge_weight` from `features_3d.py`) for this model — dense,
all-pairs attention over one molecule at a time is exactly the point of
this architecture, and is what makes it structurally distinct from every
other entry in this benchmark. Dense per-molecule batching uses
`torch_geometric.utils.to_dense_batch` (a core, dependency-free PyG
utility — no `pyg-lib`/`torch_cluster` needed), padding every molecule in
a batch to the size of its largest member and masking out padding
positions in the attention softmax.

PRETRAINING NOTE (important, read before citing this as "Uni-Mol" in a
paper): Uni-Mol's headline results come from large-scale SELF-SUPERVISED
PRETRAINING (masked atom-type prediction + 3D coordinate denoising on
~209M conformers, paper Sec. 4.1) BEFORE task-specific fine-tuning. This
benchmark does not have access to that pretraining corpus or the official
pretrained checkpoint, and — like every other model here — trains this
architecture FROM SCRATCH, directly on the sigma-profile task. What is
implemented and compared here is the Uni-Mol ENCODER ARCHITECTURE (the
SE(3)-invariant pair-biased Transformer), not the full Uni-Mol
pretrain-then-finetune PIPELINE. Report it as such in any write-up (e.g.
"Uni-Mol architecture, trained from scratch" or "Uni-Mol (no
pretraining)"), not as a reproduction of the paper's pretrained-model
numbers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from ..layers import ResidualMLP
from ...data.constants import MAX_Z
from ...data.constants_3d import CUTOFF
from ...data.features_3d import node_feat_dim_3d



class GaussianPairBias(nn.Module):
    """Expands a pairwise distance matrix into `num_kernels` Gaussian
    radial-basis channels, then linearly projects to one bias value per
    attention head (Zhou et al. 2023, Sec. 3.1, "pair representation")."""

    def __init__(self, num_heads: int, num_kernels: int = 32, max_dist: float = 12.0):
        super().__init__()
        means = torch.linspace(0.0, max_dist, num_kernels)
        self.register_buffer("means", means)
        self.std = max_dist / num_kernels
        self.proj = nn.Linear(num_kernels, num_heads)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (B, L, L) -> (B, L, L, num_heads)
        expanded = torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.means) / self.std) ** 2)
        return self.proj(expanded)


class UniMolLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, dropout: float = 0.05):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.pair_update = nn.Linear(num_heads, num_heads)  # pair-repr feedback (Sec 3.1)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pair_bias, key_padding_mask):
        B, L, F_ = h.shape
        H, D = self.num_heads, self.head_dim

        x = self.norm1(h)
        qkv = self.qkv(x).view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)  # (3,B,H,L,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logits = torch.einsum("bhid,bhjd->bhij", q, k) / (D ** 0.5)
        logits = logits + pair_bias.permute(0, 3, 1, 2)  # (B,H,L,L)

        mask = key_padding_mask[:, None, None, :]  # (B,1,1,L), True = valid
        # dtype-aware "negative infinity" substitute for masked_fill: a
        # hardcoded -1e9 overflows float16 (min representable value
        # ~-65504) under torch.autocast/AMP, since `logits` is float16
        # there -- torch.finfo(dtype).min is always safe for whatever
        # dtype `logits` happens to be (float16 under AMP, float32
        # otherwise), and is large enough in magnitude to zero out the
        # softmax weight for padded positions either way.
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg_inf)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).permute(0, 2, 1, 3).reshape(B, L, F_)
        h = h + self.dropout(self.out_proj(out))

        h = h + self.dropout(self.ffn(self.norm2(h)))

        # Pair-representation feedback from this layer's attention maps
        # (paper Sec. 3.1, "pair representation update"): the next layer's
        # bias is this layer's bias plus a linear re-projection of the
        # (head, i, j) attention probabilities.
        pair_bias = pair_bias + self.pair_update(attn.permute(0, 2, 3, 1))
        return h, pair_bias


class UniMolSigmaModel(nn.Module):
    def __init__(
        self,
        hidden: int = 192,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kernels: int = 32,
        max_dist: float = 4.0 * CUTOFF,
        out_dim: int = 51,
        dropout: float = 0.05,
    ):
        super().__init__()
        n_feat = node_feat_dim_3d()

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden // 4)
        self.input_proj = nn.Linear(n_feat + hidden // 4, hidden)

        self.pair_bias = GaussianPairBias(num_heads, num_kernels, max_dist)
        self.layers = nn.ModuleList([
            UniMolLayer(hidden, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_emb = self.z_embed(data.z.clamp(max=MAX_Z))
        h0 = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(h0.size(0), dtype=torch.long, device=h0.device)

        h, mask = to_dense_batch(h0, batch)             # (B, L, hidden), (B, L)
        pos_dense, _ = to_dense_batch(data.pos, batch)  # (B, L, 3)

        dist = torch.cdist(pos_dense, pos_dense)  # (B, L, L); padded rows/cols are masked out below
        pair_bias = self.pair_bias(dist)          # (B, L, L, num_heads)

        for layer in self.layers:
            h, pair_bias = layer(h, pair_bias, mask)

        h = self.final_norm(h)
        h_flat = h[mask]  # (N_atoms, hidden), restores the original per-atom order

        return F.softplus(self.readout(h_flat))
