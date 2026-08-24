"""
unimol.py - Uni-Mol (SE(3)-invariant 3D Transformer) for per-atom
sigma-profile prediction.

Paper:    Zhou, Gao, Ding, Zheng, Xu, Wei, Zhang & Ke, "Uni-Mol: A
          Universal 3D Molecular Representation Learning Framework",
          ICLR 2023. https://openreview.net/forum?id=6K2RM6wVqKu
Official: https://github.com/deepmodeling/Uni-Mol

REVISION HISTORY:
  v1 -> v2: rewritten to match the paper's Eq. 1/Eq. 2 more precisely
      (the pair-representation update used post-softmax attention
      probabilities instead of the raw pre-softmax score; the Gaussian
      kernel was not pair-type-aware).
  v2 -> v3 (this version): fidelity pass against the OFFICIAL SOURCE
      CODE (unimol/unimol/models/unimol.py's GaussianLayer/NonLinearHead,
      and transformer_encoder_layer.py, confirmed by direct inspection,
      not just the paper text) found and fixed three further
      discrepancies in GaussianPairTypeBias:
        (a) kernel means/stds were FIXED (a `torch.linspace` buffer and
            a plain Python float) -- official GaussianLayer makes both
            LEARNABLE (`nn.Embedding(1,K)`, `uniform_(0,3)` init).
        (b) no Gaussian normalization constant `1/(std*sqrt(2*pi))` was
            applied -- official `gaussian()` includes it.
        (c) the kernel bank -> per-head bias projection was a single
            `nn.Linear` -- official `NonLinearHead` is a 2-layer MLP
            (Linear -> activation -> Linear, hidden dim = input dim).
      Also switched the FFN activation from SiLU to GELU
      (`activation_fn="gelu"` is the official `base_architecture`
      default). See GaussianPairTypeBias's own docstring below for the
      parameter-count impact of (a)-(c) (~+1.1k params with default
      settings, confirmed well inside this benchmark's +/-5%
      parameter-budget tolerance via
      `python -m scripts.count_params --configs-dir configs/3d`).
      This is STILL a from-scratch reimplementation (no official code
      copied; no pretrained weights used), sized to this benchmark's
      shared parameter budget rather than the paper's own 15-layer/
      512-dim/64-head configuration (Table 6) -- see PRETRAINING NOTE
      below.

Architecture (paper Sec. 3.1-3.2): unlike every other 3D model in this
benchmark, Uni-Mol is NOT a cutoff-radius message-passing network -- it
is a Transformer that attends over EVERY pair of atoms in the molecule
(no neighbour cutoff, paper Sec. 2.1: "we choose Transformer ... as it
fully connects the nodes/atoms"), with 3D geometry injected as a
PAIR-LEVEL REPRESENTATION maintained alongside the usual per-atom
(token) representation, communicating with it in both directions inside
every self-attention layer:

  1. Pair representation is INITIALIZED from a pair-type-aware Gaussian
     kernel (GKPT) of the interatomic distances (paper Sec. 2.1,
     "Encode 3D positions" + Appendix D.1's GKPT ablation, which the
     paper found to outperform a plain, non-pair-type-aware Gaussian
     kernel). See `GaussianPairTypeBias` below.
  2. Pair-to-atom communication: the pair representation is added as an
     additive BIAS inside the softmax of every attention layer (paper
     Eq. 2):
         Attention(Q,K,V) = softmax( QK^T/sqrt(d) + q_ij^{l-1,h} ) V
  3. Atom-to-pair communication: after that same layer's raw
     (pre-softmax) attention score is computed, it is ACCUMULATED into
     the pair representation for the NEXT layer (paper Eq. 1):
         q_ij^{l+1} = q_ij^l + { Q_i^{l,h}(K_j^{l,h})^T/sqrt(d) | h }
     Note this is the RAW score (no bias added, no softmax applied) --
     not the post-softmax attention probability. Getting this right
     matters: accumulating post-softmax probabilities (as this file's
     first version did) discards the sign and unbounded magnitude
     information the raw score carries, which is exactly what lets the
     pair representation act as a genuinely learned, evolving spatial
     bias rather than a smoothed running average of attention patterns.

This benchmark does NOT use the shared radius graph
(`edge_index`/`edge_weight` from `features_3d.py`) for this model --
dense, all-pairs attention over one molecule at a time is exactly the
point of this architecture. Dense per-molecule batching uses
`torch_geometric.utils.to_dense_batch` (a core, dependency-free PyG
utility -- no `pyg-lib`/`torch_cluster` needed), padding every molecule
in a batch to the size of its largest member and masking out padding
positions in the attention softmax.

WHAT STILL DIFFERS FROM THE PAPER'S OWN CONFIGURATION, AND WHY (these are
scope/budget choices, not correctness bugs):
  - Depth/width: paper uses 15 layers, 512-dim embeddings, 64 attention
    heads (Table 6) for its 47M-parameter pretraining backbone. This
    benchmark targets 700k parameters (shared budget across all 9
    architectures, see configs/3d/base.yaml) -- see configs/3d/unimol.yaml
    for the actual (auto-tuned) depth/width used here.
  - No [CLS] token / no BOS-EOS: the paper adds a special whole-molecule
    [CLS] atom (coordinate = centroid) for its finetuning heads, and its
    tokenizer additionally wraps every sequence in BOS/EOS tokens that
    participate in the pair-type indexing alongside real atoms. This
    benchmark's task is PER-ATOM regression, not per-molecule, so every
    real atom's own final representation is used directly -- there is no
    per-molecule pooling step to attach a [CLS] token to, and
    `pair_type` here is computed purely from real atomic numbers (see
    GaussianPairTypeBias.forward), not from a vocabulary that also
    contains BOS/EOS/PAD/MASK entries.
  - The SE(3)-equivariant coordinate-prediction head (paper Eq. 3, used
    for the 3D-position-recovery PRETRAINING task) is not implemented:
    this benchmark only ever trains supervised, directly on the
    sigma-profile task, and never needs to predict/denoise coordinates.

PRETRAINING NOTE (important, read before citing this as "Uni-Mol" in a
paper): Uni-Mol's headline results come from large-scale SELF-SUPERVISED
PRETRAINING (masked atom-type prediction + 3D coordinate denoising on
~209M conformers, paper Sec. 2.2) BEFORE task-specific finetuning. This
file/config trains the encoder architecture FROM SCRATCH, directly on
the sigma-profile task -- no pretrained weights are used or available in
this pipeline. Report it as "Uni-Mol architecture, trained from scratch"
/ "Uni-Mol (no pretraining)", not as a reproduction of the paper's
pretrained-model numbers. For an experiment that DOES use the real
pretrained checkpoint, see `src/models/models_3d/unimol_pretrained.py`
and `src/train_unimol_pretrained.py` -- a SEPARATE, second experiment,
deliberately kept out of this shared-parameter-budget benchmark (see
that file's own docstring for why).
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


class GaussianPairTypeBias(nn.Module):
    """Pair-TYPE-aware Gaussian kernel (GKPT), paper Sec. 2.1 + Appendix
    D.1. Unlike a plain Gaussian kernel (which the paper's own ablation,
    Fig. 5, shows converges slower), the raw distance is first put
    through an AFFINE transform (mul, bias) that is looked up per PAIR OF
    ATOM TYPES (z_i, z_j) -- i.e. a C-C pair and an O-H pair get their own
    learned scale/offset on the distance -- before the (shared) bank of
    Gaussian kernels is applied. mul/bias are initialized to (1, 0) so
    training starts at the plain-Gaussian-kernel behaviour and learns the
    pair-type-specific correction from there (matches the standard init
    used for this exact layer in the official codebase's public
    description of the technique).

    REVISION NOTE (fidelity pass against the official GaussianLayer /
    NonLinearHead, unimol/unimol/models/unimol.py, confirmed via the
    official source): this class previously (a) used FIXED, non-learnable
    Gaussian kernel centers/width (`means` as a buffer via
    `torch.linspace`, `std` as a plain Python float) and (b) applied NO
    normalization constant, and (c) projected kernels -> per-head bias
    through a single `nn.Linear`. The official implementation makes both
    the kernel means AND stds LEARNABLE parameters (`nn.Embedding(1, K)`
    each, initialized `uniform_(0, 3)`), includes the standard Gaussian
    normalization `1/(std*sqrt(2*pi))`, and projects through a 2-layer
    MLP with an activation in between (`NonLinearHead`: Linear -> act ->
    Linear, hidden dim defaulting to the input dim, i.e. `num_kernels`).
    All three are fixed below. Net effect on parameter count: means/stds
    go from 0 trainable params (buffer + python float) to 2*num_kernels
    (64 with the default num_kernels=32); proj goes from
    num_kernels*num_heads+num_heads to a 2-layer MLP roughly
    num_kernels^2 + num_kernels + num_kernels*num_heads + num_heads --
    an increase of about +1.1k params total with default settings, well
    inside this benchmark's +/-5% parameter-budget tolerance (see
    tests/test_param_budget_3d.py) -- re-run
    `python -m scripts.count_params --configs-dir configs/3d` to confirm
    the exact number for your configs/3d/unimol.yaml before finalizing
    `hidden`.
    """

    def __init__(self, num_heads: int, num_kernels: int = 32,
                max_dist: float = 12.0, num_atom_types: int = MAX_Z + 1):
        super().__init__()
        n_pair_types = num_atom_types * num_atom_types
        self.num_atom_types = num_atom_types
        self.num_kernels = num_kernels

        self.mul = nn.Embedding(n_pair_types, 1)
        self.bias = nn.Embedding(n_pair_types, 1)
        nn.init.constant_(self.mul.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        # LEARNABLE kernel means/stds (official GaussianLayer:
        # nn.Embedding(1, K), uniform_(0, 3) init) -- shared across all
        # pair types (only mul/bias above are pair-type-specific),
        # matching the official design.
        self.means = nn.Parameter(torch.empty(num_kernels))
        self.log_stds = nn.Parameter(torch.empty(num_kernels))
        nn.init.uniform_(self.means, 0.0, 3.0)
        with torch.no_grad():
            init_stds = torch.empty(num_kernels).uniform_(0.0, 3.0).clamp_min(1e-3)
            self.log_stds.copy_(init_stds.log())

        # 2-layer MLP (official NonLinearHead: Linear -> act -> Linear,
        # hidden dim = input dim when not otherwise specified).
        self.proj = nn.Sequential(
            nn.Linear(num_kernels, num_kernels),
            nn.GELU(),
            nn.Linear(num_kernels, num_heads),
        )

    def forward(self, dist: torch.Tensor, z_dense: torch.Tensor) -> torch.Tensor:
        """dist: (B, L, L) padded pairwise distances.
        z_dense: (B, L) padded atomic numbers (clamped to MAX_Z already).
        Returns: (B, L, L, num_heads) pair bias."""
        pair_type = z_dense.unsqueeze(2) * self.num_atom_types + z_dense.unsqueeze(1)  # (B,L,L)
        mul = self.mul(pair_type).squeeze(-1)   # (B,L,L)
        bias = self.bias(pair_type).squeeze(-1)  # (B,L,L)
        d = mul * dist + bias

        stds = self.log_stds.exp() + 1e-5  # positivity, matches official ".abs() + 1e-5" in spirit
        d = d.unsqueeze(-1)  # (B,L,L,1)
        normalization = 1.0 / (stds * (2.0 * torch.pi) ** 0.5)
        expanded = normalization * torch.exp(-0.5 * ((d - self.means) / stds) ** 2)
        return self.proj(expanded)  # (B,L,L,num_heads)


class UniMolLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, dropout: float = 0.05):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        # REVISION NOTE: official Uni-Mol's default activation_fn is
        # "gelu" (unimol/unimol/models/unimol.py, base_architecture);
        # this was nn.SiLU() here, now fixed to nn.GELU() to match.
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pair_bias, key_padding_mask):
        """pair_bias here is q_ij^{l-1} (the PREVIOUS layer's accumulated
        pair representation) -- used as an additive attention bias
        (paper Eq. 2). Returns the updated hidden state AND this layer's
        raw (pre-softmax, pre-bias) attention score, which the CALLER
        accumulates into the pair representation per paper Eq. 1 (kept
        as a separate step here, rather than done inside this layer, so
        the accumulation itself lives in exactly one place -- see
        UniMolSigmaModel.forward)."""
        B, L, F_ = h.shape
        H, D = self.num_heads, self.head_dim

        x = self.norm1(h)
        qkv = self.qkv(x).view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)  # (3,B,H,L,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        raw_score = torch.einsum("bhid,bhjd->bhij", q, k) / (D ** 0.5)  # paper Eq. 1's accumulated term

        logits = raw_score + pair_bias.permute(0, 3, 1, 2)  # paper Eq. 2

        mask = key_padding_mask[:, None, None, :]  # (B,1,1,L), True = valid
        # dtype-aware "negative infinity" substitute for masked_fill: a
        # hardcoded -1e9 overflows float16 (min representable value
        # ~-65504) under torch.autocast/AMP, since `logits` is float16
        # there -- torch.finfo(dtype).min is always safe for whatever
        # dtype `logits` happens to be.
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg_inf)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).permute(0, 2, 1, 3).reshape(B, L, F_)
        h = h + self.dropout(self.out_proj(out))

        h = h + self.dropout(self.ffn(self.norm2(h)))

        return h, raw_score.permute(0, 2, 3, 1)  # (B,L,L,H), for the caller to accumulate


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

        self.pair_bias_init = GaussianPairTypeBias(num_heads, num_kernels, max_dist)
        self.layers = nn.ModuleList([
            UniMolLayer(hidden, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        z_clamped = data.z.clamp(max=self.z_embed.num_embeddings - 1)
        z_emb = self.z_embed(z_clamped)
        h0 = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(h0.size(0), dtype=torch.long, device=h0.device)

        h, mask = to_dense_batch(h0, batch)                    # (B, L, hidden), (B, L)
        pos_dense, _ = to_dense_batch(data.pos, batch)         # (B, L, 3)
        z_dense, _ = to_dense_batch(z_clamped, batch, fill_value=0)  # (B, L)

        dist = torch.cdist(pos_dense, pos_dense)  # (B, L, L); padded rows/cols masked out in attention
        pair_bias = self.pair_bias_init(dist, z_dense)  # q_ij^0 (paper Sec. 2.1)

        for layer in self.layers:
            h, raw_score = layer(h, pair_bias, mask)
            pair_bias = pair_bias + raw_score  # paper Eq. 1: accumulate into q_ij^{l+1}

        h = self.final_norm(h)
        h_flat = h[mask]  # (N_atoms, hidden), restores the original per-atom order

        return F.softplus(self.readout(h_flat))
