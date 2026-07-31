"""simple.py — plain MSE loss.

This repository uses a single training objective (MSE) for every
architecture in the controlled comparison, so that 'loss' in the
saved metrics is directly comparable across models (see evaluate.py).
This intentionally drops the earlier repo's optional "combined loss /
ablation" path (multi-term physically-motivated loss) to keep this
2D-only rewrite focused on one thing: a clean, correctly-controlled
architecture comparison. Add it back here later if you need the
ablation study.
"""
import torch.nn as nn


class MSELoss(nn.MSELoss):
    """Thin wrapper so the loss shares the same import path as any future
    additions (e.g. a combined/physically-motivated loss)."""
    name = "mse"
