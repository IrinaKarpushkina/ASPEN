"""simple.py — plain MSE loss, used as the fair-comparison objective."""
import torch.nn as nn

class MSELoss(nn.MSELoss):
    """Thin wrapper so all losses share the same import path."""
    name = "mse"
