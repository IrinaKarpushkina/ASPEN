"""
layers.py — общие слои, используемые всеми архитектурами.

ResidualMLP — двухслойный MLP с одним residual skip и LayerNorm.
  Используется как output head во всех моделях.
  Параметры: dim → dim → dim → out_dim, с SiLU-активацией.
"""
import torch.nn as nn


class ResidualMLP(nn.Module):
    """
    Output head: LayerNorm → FC1 → SiLU → Dropout → FC2 + residual → FC3

    Идентичен для GCN, GAT, GATv2, GINE, SchNet и финальной модели.
    Замена его во ВСЕХ моделях сразу — достаточно изменить этот файл.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1  = nn.Linear(in_dim, in_dim)
        self.fc2  = nn.Linear(in_dim, in_dim)
        self.fc3  = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        r = x
        x = self.drop(self.act(self.fc1(x)))
        x = self.act(self.fc2(x)) + r
        return self.fc3(x)
