"""
train.py — единый training entrypoint для ВСЕХ архитектур.

Запуск:
    python -m src.train --config configs/gatv2.yaml --seed 0
    python -m src.train --config configs/final_model.yaml --seed 0 --loss combined

Все архитектурно-независимые настройки (batch size, lr, scheduler, max
epochs, patience, cutoff, extended features) живут в configs/base.yaml.
Каждый configs/<model>.yaml переопределяет только model.* (hidden, heads,
...) и, при необходимости, n_params_target для авто-подбора hidden.

Выход:
    results/checkpoints/<model_name>_seed<seed>_<loss>.pt
    results/metrics/<model_name>_seed<seed>_<loss>.json
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .config import load_config
from .data.dataset import ChaosParquetDataset
from .evaluate import evaluate, CHECKPOINT_METRIC
from .losses import MSELoss, CombinedLoss
from .models import MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, use_extended: bool, mode: str = "3d"):
    import inspect
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    model_cfg.pop("n_params_target", None)  # only used by count_params.py
    model_cls = MODEL_REGISTRY[name]
    # Передаём mode только если модель явно его поддерживает.
    # SchNet — исключительно 3D, параметра mode не имеет.
    sig = inspect.signature(model_cls.__init__)
    if "mode" in sig.parameters:
        model_cfg["mode"] = mode
    return model_cls(use_extended=use_extended, **model_cfg), name


def build_loss(loss_name: str, bin_weights: torch.Tensor, device):
    if loss_name == "mse":
        return MSELoss()
    elif loss_name == "combined":
        return CombinedLoss(bin_weights.to(device))
    raise ValueError(f"Unknown loss: {loss_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", default=None,
                         help="Override cfg['loss'] (e.g. 'mse' or 'combined') — for ablation.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--force-recompute-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_name = args.loss or cfg.get("loss", "mse")
    use_extended = data_cfg.get("use_extended", True)

    # ── Datasets (precomputed + cached, see data/dataset.py) ────────────────
    dataset_mode = data_cfg.get("mode", "3d")   # "3d" или "2d_pure"
    common_ds_kwargs = dict(
        mode=dataset_mode,
        use_extended=use_extended,
        cutoff=data_cfg.get("cutoff", 12.0),
        max_num_neighbors=data_cfg.get("max_num_neighbors", 32),
        n_rbf=data_cfg.get("n_rbf", 32),
        cache_dir=data_cfg.get("cache_dir"),
        force_recompute=args.force_recompute_cache,
    )
    train_dataset = ChaosParquetDataset(data_cfg["train_path"], **common_ds_kwargs)
    val_dataset   = ChaosParquetDataset(data_cfg["val_path"],   **common_ds_kwargs)
    test_dataset  = ChaosParquetDataset(data_cfg["test_path"],  **common_ds_kwargs)

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    bin_weights_np = train_dataset.bin_weights_numpy()
    bin_weights_t  = torch.tensor(bin_weights_np, dtype=torch.float)

    # ── Model / loss / optimizer / scheduler ────────────────────────────────
    model, model_name = build_model(cfg, use_extended, mode=dataset_mode)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = cfg["model"].get("n_params_target", cfg.get("target_params"))
    if target:
        diff_pct = abs(n_params - target) / target * 100
        logger.info(f"{model_name}: {n_params:,} params (target {target:,}, diff {diff_pct:.1f}%)")
    else:
        logger.info(f"{model_name}: {n_params:,} params")

    criterion = build_loss(loss_name, bin_weights_t, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=train_cfg.get("scheduler_T0", 20),
        T_mult=train_cfg.get("scheduler_Tmult", 2),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    max_epochs = train_cfg["max_epochs"]
    patience   = train_cfg["patience"]
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    run_id = f"{model_name}_seed{args.seed}_{loss_name}"
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_id}.pt")

    best_metric = np.inf
    patience_ctr = 0
    history = []

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                preds = model(data)
                loss = criterion(preds, data.y) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step(epoch + 1)

        val_metrics = evaluate(model, val_loader, criterion, bin_weights_np, device, use_amp)
        epoch_time = time.time() - t0

        logger.info(
            f"[{run_id}] Epoch {epoch+1:3d}/{max_epochs} | "
            f"Train={train_loss:.5f} | Val={val_metrics['loss']:.5f} | "
            f"wMAE={val_metrics['weighted_mae']:.6f} | "
            f"PolarMAE={val_metrics['polar_mae']:.6f} | "
            f"EMD={val_metrics['emd_raw']:.6f} | "
            f"MolEMD={val_metrics['molecular_emd']:.6f} | "
            f"R2={val_metrics['weighted_r2']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        current_metric = val_metrics[CHECKPOINT_METRIC]
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": cfg,
                "n_params": n_params,
            }, ckpt_path)
            logger.info(f"  -> new best ({CHECKPOINT_METRIC}={best_metric:.6f}), saved {ckpt_path}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    # ── Final test evaluation with best checkpoint ──────────────────────────
    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, bin_weights_np, device, use_amp)
    logger.info(f"[{run_id}] TEST METRICS: {test_metrics}")

    result = {
        "run_id": run_id, "model": model_name, "loss": loss_name, "seed": args.seed,
        "n_params": n_params, "best_epoch": ckpt["epoch"],
        "val_metrics_at_best": ckpt["val_metrics"], "test_metrics": test_metrics,
        "history": history,
    }
    out_path = os.path.join(metrics_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
