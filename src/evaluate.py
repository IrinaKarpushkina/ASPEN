"""
evaluate.py — единый evaluation loop для всех архитектур.

evaluate() возвращает полный набор метрик из metrics.compute_all_metrics,
плюс 'loss' (MSE — единственный training objective в этом репозитории,
поэтому 'loss' напрямую сравним между архитектурами).

checkpoint selection criterion (best model) — weighted_mae на валидации,
ОДИНАКОВЫЙ для всех моделей. Это устраняет любую возможность, что разные
модели выбираются по разным критериям.
"""
from __future__ import annotations
import logging
import traceback

import numpy as np
import torch

from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Метрика, используемая для отбора лучшего чекпоинта (early stopping /
# model selection). Одна и та же для ВСЕХ моделей -> честное сравнение.
CHECKPOINT_METRIC = "weighted_mae"   # lower is better


@torch.no_grad()
def evaluate(model, loader, criterion, bin_weights_np, device, use_amp=False) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets, mol_sizes = [], [], []

    for data in loader:
        try:
            data = data.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                preds = model(data)
                loss = criterion(preds, data.y)

            total_loss += float(loss.item())
            n_batches += 1

            all_preds.append(preds.float().cpu().numpy())
            all_targets.append(data.y.float().cpu().numpy())
            mol_sizes.extend(torch.bincount(data.batch.cpu()).numpy().tolist())
        except Exception as e:
            logger.error(f"Eval error: {e}\n{traceback.format_exc()}")

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    loss_value = total_loss / max(n_batches, 1)

    return compute_all_metrics(targets_np, preds_np, mol_sizes, bin_weights_np, loss_value)
