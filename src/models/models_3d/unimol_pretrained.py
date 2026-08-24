"""
unimol_pretrained.py - SECOND, separate Uni-Mol experiment: a small
trainable head on top of REAL pretrained per-atom Uni-Mol representations
(from `unimol_tools`, see `src/data/dataset_unimol_pretrained.py`).

NOT PART OF THE MAIN 700k-PARAMETER-BUDGET BENCHMARK, deliberately: the
frozen backbone producing these representations has ~47M pretrained
parameters (paper Table 6) trained on 209M conformations -- there is no
way to make this "budget-matched" against the other 9 from-scratch
architectures without either (a) crippling the pretrained model in a way
that defeats the point of using it, or (b) inflating every other
architecture to tens of millions of parameters, which this benchmark's
whole controlled-comparison design (Dwivedi et al. 2022) exists to avoid.
Report this experiment's results in a SEPARATE table/section, e.g.
"Uni-Mol (pretrained, frozen backbone)" vs. the main "9 architectures,
700k params, trained from scratch" table -- see PROVENANCE.md.

The pretrained backbone is FROZEN (not finetuned): `dataset_unimol_pretrained.py`
precomputes and caches every molecule's per-atom representations once, as
a data-preprocessing step, not as part of the trainable computation graph.
This is a "frozen feature extraction" baseline, not full end-to-end
finetuning of the pretrained transformer -- a legitimate and common way to
evaluate a foundation model's representations, but weaker than finetuning
would likely be. If you want full finetuning instead, you would need to
run this through unimol_tools' own `MolTrain` finetuning interface
directly (a fundamentally different, non-PyTorch-Geometric pipeline) --
out of scope for this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..layers import ResidualMLP


class UniMolPretrainedSigmaModel(nn.Module):
    def __init__(
        self,
        repr_dim: int = 512,     # unimolv1 default embedding dim (paper Table 6); auto-checked at runtime, see forward()
        hidden: int = 128,
        out_dim: int = 51,
        dropout: float = 0.1,
        num_head_layers: int = 2,
    ):
        super().__init__()
        self.repr_dim = repr_dim

        layers = [nn.Linear(repr_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(max(num_head_layers - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.proj = nn.Sequential(*layers)

        self.readout = ResidualMLP(hidden, out_dim, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        repr_ = data.unimol_repr.float()  # stored as fp16 on disk (see dataset docstring), upcast for the head
        if repr_.size(-1) != self.repr_dim:
            raise ValueError(
                f"unimol_pretrained repr_dim mismatch: model configured for "
                f"{self.repr_dim}, but data has {repr_.size(-1)}. Set "
                f"`repr_dim` in configs/3d/unimol_pretrained.yaml to match "
                f"what UniMolRepr actually returned (printed by "
                f"dataset_unimol_pretrained.py's logger as 'repr_dim=...')."
            )
        h = self.proj(repr_)
        return F.softplus(self.readout(h))
