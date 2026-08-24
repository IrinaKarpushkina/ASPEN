"""
train_unimol_pretrained.py - training entrypoint for the SECOND, separate
Uni-Mol experiment: a small trainable head on top of REAL pretrained
per-atom representations from `unimol_tools`.

Deliberately a SEPARATE script from src/train_3d.py (not just a config
flag) for the same reason this experiment lives in its own config
directory (configs/unimol_pretrained/, not configs/3d/): the dataset
class, the "model" (a frozen-backbone-features + trainable head, not a
from-scratch architecture), and the parameter-budget bookkeeping are all
fundamentally different from the rest of this benchmark, and mixing them
into the same entrypoint invites exactly the kind of silent mode-mixing
this repo's PROVENANCE.md documents as a past real bug (see "Bug #1").

Usage:
    python -m src.train_unimol_pretrained \
        --config configs/unimol_pretrained/unimol_pretrained.yaml --seed 0

Output:
    results/checkpoints/unimol_pretrained_seed<seed>_mse.pt
    results/metrics/unimol_pretrained_seed<seed>_mse.json   (mode: "unimol_pretrained")

The "mode" field is deliberately different from "2d_pure"/"3d_pure" so
scripts/aggregate_results.py's mixed-mode guard (see PROVENANCE.md Bug #1)
keeps this out of the main comparison table by default -- exactly as
intended, see this experiment's own docstrings for why.
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
from .data.dataset_unimol_pretrained import UniMolPretrainedDataset
from .evaluate import evaluate, CHECKPOINT_METRIC
from .losses import MSELoss
from .models.models_3d.unimol_pretrained import UniMolPretrainedSigmaModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODE = "unimol_pretrained"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--force-recompute-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    um_cfg = cfg["unimol_tools"]

    common_ds_kwargs = dict(
        cache_dir=data_cfg.get("cache_dir"),
        force_recompute=args.force_recompute_cache,
        model_name=um_cfg["model_name"], model_size=um_cfg["model_size"],
        remove_hs=um_cfg["remove_hs"], use_own_coords=um_cfg["use_own_coords"],
        repr_batch_size=um_cfg.get("repr_batch_size", 32),
    )
    train_dataset = UniMolPretrainedDataset(data_cfg["train_path"], **common_ds_kwargs)
    val_dataset = UniMolPretrainedDataset(data_cfg["val_path"], **common_ds_kwargs)
    test_dataset = UniMolPretrainedDataset(data_cfg["test_path"], **common_ds_kwargs)

    if train_dataset.repr_dim != cfg["model"]["repr_dim"]:
        logger.warning(
            f"configs/unimol_pretrained/*.yaml has model.repr_dim="
            f"{cfg['model']['repr_dim']}, but the dataset actually produced "
            f"repr_dim={train_dataset.repr_dim}. Overriding the config value "
            f"with the dataset's actual dim -- fix the yaml to silence this."
        )
        cfg["model"]["repr_dim"] = train_dataset.repr_dim

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    bin_weights_np = train_dataset.bin_weights_numpy()

    model_cfg = dict(cfg["model"])
    model_cfg.pop("name")
    model = UniMolPretrainedSigmaModel(**model_cfg).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"unimol_pretrained: {n_trainable:,} TRAINABLE params (head only -- "
        f"the pretrained backbone that produced the input features is "
        f"FROZEN and not counted here; see this file's module docstring)."
    )

    criterion = MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=train_cfg.get("scheduler_T0", 15), T_mult=train_cfg.get("scheduler_Tmult", 2),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    max_epochs = train_cfg["max_epochs"]
    patience = train_cfg["patience"]
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    run_id = f"unimol_pretrained_seed{args.seed}_mse"
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_id}.pt")

    best_metric = np.inf
    patience_ctr = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
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
            f"[{run_id}] Epoch {epoch + 1:3d}/{max_epochs} | Train={train_loss:.5f} | "
            f"Val={val_metrics['loss']:.5f} | wMAE={val_metrics['weighted_mae']:.6f} | "
            f"R2={val_metrics['weighted_r2']:.4f} | LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        current_metric = val_metrics[CHECKPOINT_METRIC]
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
                "val_metrics": val_metrics, "config": cfg, "n_trainable_params": n_trainable,
            }, ckpt_path)
            logger.info(f"  -> new best ({CHECKPOINT_METRIC}={best_metric:.6f}), saved {ckpt_path}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, bin_weights_np, device, use_amp)
    logger.info(f"[{run_id}] TEST METRICS: {test_metrics}")

    result = {
        "run_id": run_id, "model": "unimol_pretrained", "loss": "mse", "seed": args.seed,
        "mode": MODE,   # deliberately distinct from "2d_pure"/"3d_pure" -- see module docstring
        "config_path": os.path.abspath(args.config),
        "n_trainable_params": n_trainable, "best_epoch": ckpt["epoch"],
        "val_metrics_at_best": ckpt["val_metrics"], "test_metrics": test_metrics,
        "history": history,
    }
    out_path = os.path.join(metrics_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
