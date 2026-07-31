"""
metrics.py — единый модуль метрик для всех архитектур.

Один источник правды: любое изменение метрики отражается во всех моделях
одновременно. Ported unchanged from the original ASPEN benchmark — this is
pure evaluation math with no dependency on model architecture, so it was
not part of the pooling/graph-construction bug fixed in this rewrite.

Атомарные метрики (per-atom sigma-profiles):
  weighted_r2, weighted_mae, polar_mae, cosine_similarity,
  emd_raw, emd_normalized

Молекулярные метрики (суммированные per-molecule sigma-profiles):
  molecular_mae, molecular_emd, molecular_cosine, molecular_polar_mae
"""

from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance

from .data.constants import SIGMA_BINS, POLAR_MASK, EPS

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ─────────────────────────────────────────────────────────────────────────────
# Атомарные метрики
# ─────────────────────────────────────────────────────────────────────────────
def weighted_r2(targets: np.ndarray, preds: np.ndarray, weights: np.ndarray) -> float:
    """Variance-weighted R^2 по бинам sigma-профиля."""
    w = weights / (weights.sum() + EPS)
    ss_res = np.sum(((targets - preds) ** 2) * w[None, :], axis=1).mean()
    mean_t = np.sum(targets * w[None, :], axis=1, keepdims=True)
    ss_tot = np.sum(((targets - mean_t) ** 2) * w[None, :], axis=1).mean()
    return float(1.0 - ss_res / (ss_tot + EPS))


def weighted_mae(targets: np.ndarray, preds: np.ndarray, weights: np.ndarray) -> float:
    """Variance-weighted MAE по бинам sigma-профиля."""
    w = weights / (weights.sum() + EPS)
    return float(np.mean(np.sum(np.abs(targets - preds) * w[None, :], axis=1)))


def polar_mae(targets: np.ndarray, preds: np.ndarray) -> float:
    """MAE только на полярных бинах (|sigma| >= 0.01 e/A^2)."""
    return float(np.mean(np.abs(targets[:, POLAR_MASK] - preds[:, POLAR_MASK])))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def cosine_similarity(targets: np.ndarray, preds: np.ndarray) -> float:
    """Среднее косинусное сходство по атомам."""
    return float(np.mean([_cosine(t, p) for t, p in zip(targets, preds)]))


def emd_raw(targets: np.ndarray, preds: np.ndarray) -> float:
    """Earth Mover's Distance (Wasserstein-1) между ненормированными
    атомными sigma-профилями. Чувствителен к абсолютным значениям."""
    vals = []
    for t, p in zip(targets, preds):
        t_s, p_s = np.maximum(t, 0.0) + EPS, np.maximum(p, 0.0) + EPS
        try:
            vals.append(wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_s, p_s))
        except Exception:
            pass
    return float(np.nanmean(vals)) if vals else float("nan")


def emd_normalized(targets: np.ndarray, preds: np.ndarray) -> float:
    """EMD между нормированными (единичная площадь) sigma-профилями."""
    vals = []
    for t, p in zip(targets, preds):
        t_s = np.maximum(t, 0.0) + EPS
        p_s = np.maximum(p, 0.0) + EPS
        t_n = t_s / (_trapz(t_s, SIGMA_BINS) + EPS)
        p_n = p_s / (_trapz(p_s, SIGMA_BINS) + EPS)
        try:
            vals.append(wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_n, p_n))
        except Exception:
            pass
    return float(np.nanmean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Молекулярные метрики
# ─────────────────────────────────────────────────────────────────────────────
def molecular_metrics(targets: np.ndarray, preds: np.ndarray, mol_sizes: list) -> dict:
    """Суммирует атомные sigma-профили в молекулярные и считает метрики.

    Args:
        targets:   (N_atoms, 51) array
        preds:     (N_atoms, 51) array
        mol_sizes: list of ints — число атомов в каждой молекуле батча
                   (от torch.bincount(data.batch))
    """
    mol_mae, mol_emd, mol_cos, mol_polar = [], [], [], []
    idx = 0
    for size in mol_sizes:
        t_mol = targets[idx: idx + size].sum(axis=0)
        p_mol = preds[idx: idx + size].sum(axis=0)

        mol_mae.append(float(np.mean(np.abs(t_mol - p_mol))))
        mol_polar.append(float(np.mean(np.abs(t_mol[POLAR_MASK] - p_mol[POLAR_MASK]))))
        mol_cos.append(_cosine(
            t_mol / (np.linalg.norm(t_mol) + EPS),
            p_mol / (np.linalg.norm(p_mol) + EPS),
        ))
        try:
            mol_emd.append(wasserstein_distance(
                SIGMA_BINS, SIGMA_BINS,
                np.maximum(t_mol, 0.0) + EPS,
                np.maximum(p_mol, 0.0) + EPS,
            ))
        except Exception:
            pass
        idx += size

    return {
        "molecular_mae": float(np.mean(mol_mae)) if mol_mae else float("nan"),
        "molecular_emd": float(np.mean(mol_emd)) if mol_emd else float("nan"),
        "molecular_cosine": float(np.mean(mol_cos)) if mol_cos else float("nan"),
        "molecular_polar_mae": float(np.mean(mol_polar)) if mol_polar else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Сводная функция — вызывается из evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    mol_sizes: list,
    bin_weights: np.ndarray,
    loss_value: float = float("nan"),
) -> dict:
    """Вычисляет полный набор метрик из предсказаний и целей.

    Примечание: 'loss' в возвращаемом dict — training objective данной
    архитектуры (MSE). В этом репозитории все архитектуры используют
    ОДИНАКОВЫЙ loss (MSE) для честного сравнения (Level 1), поэтому
    'loss' здесь сравним между моделями напрямую.
    """
    p = np.maximum(preds, 0.0)
    t = np.maximum(targets, 0.0)

    metrics = {
        "loss": loss_value,
        "weighted_r2": weighted_r2(t, p, bin_weights),
        "weighted_mae": weighted_mae(t, p, bin_weights),
        "polar_mae": polar_mae(t, p),
        "cosine_similarity": cosine_similarity(t, p),
        "emd_raw": emd_raw(t, p),
        "emd_normalized": emd_normalized(t, p),
    }
    metrics.update(molecular_metrics(t, p, mol_sizes))
    return metrics
