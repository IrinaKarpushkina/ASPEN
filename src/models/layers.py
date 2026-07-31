"""
layers.py — общие вспомогательные слои.

ResidualMLP — двухслойный MLP-readout head с одним residual skip и
LayerNorm, применяемый к уже посчитанным per-atom эмбеддингам КАЖДОЙ
модели в самом конце (после того, как вся message-passing/attention
специфичная для архитектуры уже отработала).

Это НЕ то же самое, что "shared global pooling block", который был
найден и удалён в этой версии репозитория (см. PROVENANCE.md): ResidualMLP
не смешивает информацию между атомами и не даёт молекулярный контекст —
это обычный per-atom prediction head, применяемый независимо к каждому
узлу. Использование одного и того же по форме output head во всех
моделях — стандартная и не вызывающая вопросов практика (аналогично
тому, как разные архитектуры в литературе обычно завершаются одним и тем
же типом linear/MLP head перед лоссом).
"""
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Output head: LayerNorm -> FC1 -> SiLU -> Dropout -> FC2 + residual -> FC3."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.fc3 = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        r = x
        x = self.drop(self.act(self.fc1(x)))
        x = self.act(self.fc2(x)) + r
        return self.fc3(x)
