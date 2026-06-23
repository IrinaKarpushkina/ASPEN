#!/usr/bin/env python3
"""
study_cutoff.py — исследует влияние радиуса обрезания графа (cutoff) на
скорость обучения, потребление памяти и качество модели.

Запуск:
    python -m scripts.study_cutoff

Опции:
    --cutoffs 5,8,10,12        : список значений cutoff (по умолчанию: 5,8,10,12)
    --model gcn                : имя модели (должна быть в MODEL_REGISTRY)
    --epochs 20                : количество эпох для каждого cutoff
    --subset 1000              : использовать только первые N молекул из train (ускоряет)
    --output results/cutoff_study.csv

Результаты сохраняются в CSV и выводятся на экран.
"""

import argparse
import copy
import csv
import os
import sys
import time
import yaml
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.config import load_config
from src.models import MODEL_REGISTRY
from src.data import SigmaDataset, radius_graph_pure
from src.trainer import Trainer
from src.evaluate import weighted_mae


def distance_histogram(cutoff_max=15.0, bins=50, cache_path=None):
    """
    Строит гистограмму межатомных расстояний в датасете.
    Использует уже закешированные графы (если есть) или вычисляет на лету.
    """
    cfg = load_config("configs/base.yaml")
    dataset_cfg = cfg["data"]
    dataset = SigmaDataset(
        path=dataset_cfg["train_path"],
        cache_dir=dataset_cfg["cache_dir"],
        use_extended=dataset_cfg["use_extended"],
        cutoff=cutoff_max,
        max_num_neighbors=dataset_cfg["max_num_neighbors"],
        n_rbf=dataset_cfg["n_rbf"],
    )
    all_dists = []
    for i, data in enumerate(dataset):
        # edge_index: [2, E], pos: [N, 3]
        pos = data.pos
        edge_index = data.edge_index
        if edge_index.size(1) == 0:
            continue
        # Вычисляем евклидовы расстояния для каждого ребра
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1)
        all_dists.extend(dist.cpu().numpy())
    all_dists = np.array(all_dists)
    counts, edges = np.histogram(all_dists, bins=bins, range=(0, cutoff_max))
    print("\n=== Распределение межатомных расстояний ===")
    print(f"Всего рёбер: {len(all_dists):,}")
    print(f"Среднее расстояние: {all_dists.mean():.2f} Å")
    print(f"Медиана: {np.median(all_dists):.2f} Å")
    print(f"95-й перцентиль: {np.percentile(all_dists, 95):.2f} Å")
    print(f"99-й перцентиль: {np.percentile(all_dists, 99):.2f} Å")
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if counts[i] > 0:
            print(f"{low:.1f}-{high:.1f} Å: {counts[i]:,} рёбер ({100*counts[i]/len(all_dists):.1f}%)")
    return all_dists


def test_cutoff(cutoff, model_name, config_path, base_config, max_epochs, subset_size, device="cuda"):
    """
    Запускает обучение модели на заданном cutoff и возвращает:
    - best_val_mae
    - total_time (сек)
    - max_memory (MiB)
    """
    # Создаём временный конфиг с новым cutoff
    test_cfg = copy.deepcopy(base_config)
    test_cfg["data"]["cutoff"] = cutoff
    # Если dataset уже закеширован с другим cutoff, временно меняем cache_dir
    # (чтобы не ломать основной кеш)
    cache_subdir = f"tmp_cache_cutoff_{cutoff}"
    test_cfg["data"]["cache_dir"] = os.path.join(test_cfg["data"]["cache_dir"], cache_subdir)
    os.makedirs(test_cfg["data"]["cache_dir"], exist_ok=True)

    # Создаём датасет (построит графы заново)
    dataset = SigmaDataset(
        path=test_cfg["data"]["train_path"],
        cache_dir=test_cfg["data"]["cache_dir"],
        use_extended=test_cfg["data"]["use_extended"],
        cutoff=cutoff,
        max_num_neighbors=test_cfg["data"]["max_num_neighbors"],
        n_rbf=test_cfg["data"]["n_rbf"],
    )
    if subset_size and subset_size < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(subset_size))

    val_dataset = SigmaDataset(
        path=test_cfg["data"]["val_path"],
        cache_dir=test_cfg["data"]["cache_dir"],
        use_extended=test_cfg["data"]["use_extended"],
        cutoff=cutoff,
        max_num_neighbors=test_cfg["data"]["max_num_neighbors"],
        n_rbf=test_cfg["data"]["n_rbf"],
    )

    loader = DataLoader(dataset, batch_size=test_cfg["training"]["batch_size"],
                        shuffle=True, num_workers=0)  # num_workers=0 для простоты
    val_loader = DataLoader(val_dataset, batch_size=test_cfg["training"]["batch_size"],
                            shuffle=False, num_workers=0)

    # Инициализируем модель
    model_cfg = test_cfg["model"]
    model_cfg["name"] = model_name
    # Удаляем ключи, не нужные конструктору
    model_cfg.pop("n_params_target", None)
    model = MODEL_REGISTRY[model_name](
        use_extended=test_cfg["data"]["use_extended"],
        **model_cfg
    ).to(device)

    # Тренер
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        config=test_cfg,
        device=device,
    )

    # Замер времени и памяти
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    best_val_mae = trainer.train(max_epochs=max_epochs)  # возвращает лучшую val weighted_mae
    total_time = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MiB

    # Очистка временного кеша
    shutil.rmtree(test_cfg["data"]["cache_dir"], ignore_errors=True)

    return best_val_mae, total_time, max_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", type=str, default="5,8,10,12",
                        help="comma-separated list of cutoff values (Angstrom)")
    parser.add_argument("--model", type=str, default="gcn",
                        help="model architecture to test (must be in MODEL_REGISTRY)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs per cutoff")
    parser.add_argument("--subset", type=int, default=500,
                        help="number of molecules to use from train set (None = all)")
    parser.add_argument("--output", type=str, default="results/cutoff_study.csv")
    parser.add_argument("--no-histogram", action="store_true",
                        help="skip distance histogram")
    args = parser.parse_args()

    cutoffs = [float(x) for x in args.cutoffs.split(",")]

    # Загружаем базовый конфиг один раз
    base_cfg = load_config("configs/base.yaml")
    # Убедимся, что модель в конфиге соответствует выбранной
    base_cfg["model"] = {"name": args.model}
    # Если в конфиге модели есть параметры, нужно подгрузить их из специфичного файла
    model_config_path = f"configs/{args.model}.yaml"
    if os.path.exists(model_config_path):
        model_cfg_full = load_config(model_config_path)
        base_cfg["model"].update(model_cfg_full.get("model", {}))

    # 1. Гистограмма расстояний (опционально)
    if not args.no_histogram:
        print("\n=== Анализ расстояний в вашем датасете ===")
        distance_histogram(cutoff_max=max(cutoffs), cache_path=base_cfg["data"]["cache_dir"])

    # 2. Тестируем каждый cutoff
    print("\n=== Тестирование cutoff ===")
    results = []
    for cutoff in cutoffs:
        print(f"\nCutoff = {cutoff} Å")
        best_mae, total_time, mem = test_cutoff(
            cutoff=cutoff,
            model_name=args.model,
            config_path=None,  # не используется, передаём base_cfg
            base_config=base_cfg,
            max_epochs=args.epochs,
            subset_size=args.subset,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  Best val MAE: {best_mae:.4f}")
        print(f"  Time: {total_time:.1f} s")
        print(f"  Peak GPU memory: {mem:.1f} MiB")
        results.append({
            "cutoff": cutoff,
            "best_val_mae": best_mae,
            "total_time_s": total_time,
            "peak_memory_mib": mem,
        })

    # Сохраняем в CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cutoff", "best_val_mae", "total_time_s", "peak_memory_mib"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
