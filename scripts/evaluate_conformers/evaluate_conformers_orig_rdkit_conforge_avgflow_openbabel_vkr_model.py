#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка предсказаний сигма-профилей на тестовой выборке
для оригинальных DFT-координат и конформеров (Conforge, RDKit, AvgFlow, OpenBabel)
Используется обученная модель SigmaSchNet (с весами schnet_with_moments_in_loss_my_vkr.pt)
Метрики соответствуют описанным в литобзоре (зелёные и ключевые)
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS (ИЗМЕНИТЕ ПОД СВОЮ СИСТЕМУ)
# =============================================================================
BASE_DATA_DIR = "/mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df"
TEST_ORIG_PATH = os.path.join(BASE_DATA_DIR, "chaos_atomic_test_with_coordinates.parquet")
TRAIN_PATH = os.path.join(BASE_DATA_DIR, "chaos_atomic_train_with_coordinates.parquet")  # для весов

GENERATED_DIR = "/mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/generated_conformers_df"
MODEL_PATH = "/mnt/tank/scratch/ikarpushkina/sigma/ASPEN/models/schnet_with_moments_in_loss_my_vkr.pt"

OUTPUT_JSON = "conformers_evaluation_metrics_full.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# КОНСТАНТЫ СЕТКИ σ
# =============================================================================
SIGMA_BINS = np.linspace(-0.025, 0.025, 51)
DELTA_SIGMA = float(SIGMA_BINS[1] - SIGMA_BINS[0])
SIGMA_RANGE = float(SIGMA_BINS[-1] - SIGMA_BINS[0])
POLAR_THRESHOLD = 0.01
POLAR_MASK = ((SIGMA_BINS <= -POLAR_THRESHOLD) | (SIGMA_BINS >= POLAR_THRESHOLD))
EPS = 1e-12

# =============================================================================
# СЛОВАРИ АТОМОВ И ЭЛЕКТРООТРИЦАТЕЛЬНОСТИ (как в обучении)
# =============================================================================
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
}
ELECTRONEGATIVITY = {
    1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    10: 0.00, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    18: 0.00, 19: 0.82, 20: 1.00, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 46: 2.20, 47: 1.93, 48: 1.69, 50: 1.96, 52: 2.10,
    53: 2.66, 78: 2.28, 79: 2.54, 80: 2.00, 82: 2.33,
}
ENEG_DEFAULT = 2.0
CUTOFF = 12.0

# =============================================================================
# ДАТАСЕТ ДЛЯ ТЕСТОВЫХ ДАННЫХ (ОРИГИНАЛ)
# =============================================================================
class TestParquetDataset(Dataset):
    """Загружает тестовые данные из parquet (с координатами и профилями)."""
    def __init__(self, parquet_path):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.mol_ids = self.df["mol_id"].unique().tolist()
        self.sigma_cols = [c for c in self.df.columns if c.startswith("sigma_")]
        logger.info(f"Загружено молекул: {len(self.mol_ids)}, атомных записей: {len(self.df)}")

    def len(self):
        return len(self.mol_ids)

    def get(self, idx):
        mol_id = self.mol_ids[idx]
        mol_df = self.df[self.df["mol_id"] == mol_id].sort_values("atom_index")
        # Атомные номера
        z_list = [ELEMENT_TO_Z.get(e, 6) for e in mol_df["element"]]
        z = torch.tensor(z_list, dtype=torch.long)
        # Координаты (оригинальные DFT)
        pos = torch.tensor(mol_df[["coord_x", "coord_y", "coord_z"]].values, dtype=torch.float)
        # Электроотрицательность
        eneg = torch.tensor([ELECTRONEGATIVITY.get(zi, ENEG_DEFAULT) for zi in z_list], dtype=torch.float).unsqueeze(1)
        # Координационное число (число соседей в радиусе CUTOFF)
        pos_np = pos.numpy()
        diffs = pos_np[:, None, :] - pos_np[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        degree = ((dists < CUTOFF) & (dists > 0.0)).sum(axis=1)
        degree = torch.tensor(degree, dtype=torch.float).unsqueeze(1)
        # Целевой сигма-профиль (DFT)
        y = torch.tensor(mol_df[self.sigma_cols].values, dtype=torch.float)
        y = torch.nan_to_num(y, nan=0.0)
        return Data(z=z, pos=pos, y=y, eneg=eneg, degree=degree)


# =============================================================================
# ДАТАСЕТ С ПОДСТАНОВКОЙ КООРДИНАТ КОНФОРМЕРОВ
# =============================================================================
class ConformerSubstituteDataset(Dataset):
    """Берёт данные из test_parquet_dataset, но заменяет pos на координаты из внешнего словаря."""
    def __init__(self, base_dataset, coords_dict, name=""):
        super().__init__()
        self.base = base_dataset
        self.coords_dict = coords_dict
        self.name = name
        self.data_list = []
        # Предварительно клонируем данные и подменяем pos
        for idx in range(len(base_dataset)):
            orig_data = base_dataset.get(idx)
            mol_id = base_dataset.mol_ids[idx]
            if mol_id in coords_dict:
                new_coords = coords_dict[mol_id]
                if new_coords.shape[0] == orig_data.z.shape[0]:
                    new_data = orig_data.clone()
                    new_data.pos = torch.tensor(new_coords, dtype=torch.float)
                    self.data_list.append(new_data)
                else:
                    logger.warning(f"{name}: несовпадение числа атомов для {mol_id}, пропуск")
            else:
                logger.warning(f"{name}: молекула {mol_id} не найдена в словаре, пропуск")
        logger.info(f"{name}: загружено {len(self.data_list)} молекул")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# =============================================================================
# АРХИТЕКТУРА МОДЕЛИ (SigmaSchNet) – ДОЛЖНА СОВПАДАТЬ С ОБУЧЕНИЕМ
# =============================================================================
class ResidualMLP(nn.Module):
    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        r = x
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x)) + r
        return self.fc3(x)


class SigmaSchNet(nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=8,
                 num_gaussians=50, cutoff=CUTOFF, out_dim=51):
        super().__init__()
        from torch_geometric.nn import SchNet
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=32,
        )
        self.atom_proj = nn.Linear(hidden_channels + 2, hidden_channels)
        self.readout = ResidualMLP(hidden_channels, out_dim)

    def forward(self, data):
        h = self.schnet.embedding(data.z)
        edge_index, edge_weight = self.schnet.interaction_graph(data.pos, data.batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        extra = torch.cat([data.eneg, data.degree], dim=1)
        h = self.atom_proj(torch.cat([h, extra], dim=1))
        out = self.readout(h)
        return torch.clamp(out, min=0.0)


# =============================================================================
# ВЫЧИСЛЕНИЕ ВЕСОВ БИНОВ (ПО ОБУЧАЮЩЕМУ ДАТАСЕТУ)
# =============================================================================
def compute_bin_weights_from_parquet(train_parquet_path):
    logger.info("Вычисление весов бинов на обучающем датасете...")
    df_train = pd.read_parquet(train_parquet_path)
    sigma_cols = [c for c in df_train.columns if c.startswith("sigma_")]
    # Собираем все атомные профили
    y_list = []
    for mol_id, group in tqdm(df_train.groupby("mol_id"), desc="Сборка профилей"):
        group = group.sort_values("atom_index")
        y = group[sigma_cols].values
        y_list.append(y)
    ys = np.concatenate(y_list, axis=0)
    var = np.var(ys, axis=0)
    var = np.maximum(var, EPS)
    weights = var / var.sum()
    return torch.tensor(weights, dtype=torch.float)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ МЕТРИК
# =============================================================================
def weighted_r2(targets, preds, weights_np):
    """Взвешенный R² (variance-weighted)"""
    weights_np = weights_np / weights_np.sum()
    ss_res = np.sum(((targets - preds) ** 2) * weights_np[None, :], axis=1).mean()
    mean_t = np.sum(targets * weights_np[None, :], axis=1, keepdims=True)
    ss_tot = np.sum(((targets - mean_t) ** 2) * weights_np[None, :], axis=1).mean()
    return 1.0 - ss_res / (ss_tot + EPS)


def weighted_mae(targets, preds, weights_np):
    """Взвешенная средняя абсолютная ошибка"""
    err = np.abs(targets - preds)
    return np.mean(np.sum(err * weights_np[None, :], axis=1))


def std_mae(targets, preds):
    return np.mean(np.abs(targets - preds))


def polar_mae(targets, preds):
    return np.mean(np.abs(targets[:, POLAR_MASK] - preds[:, POLAR_MASK]))


def sobolev_metric(targets, preds):
    dt = (targets[:, 1:] - targets[:, :-1]) / DELTA_SIGMA
    dp = (preds[:, 1:] - preds[:, :-1]) / DELTA_SIGMA
    return np.mean(np.abs(dt - dp))


def cosine_fast(a, b):
    a_norm = np.linalg.norm(a) + EPS
    b_norm = np.linalg.norm(b) + EPS
    return np.dot(a, b) / (a_norm * b_norm)


def cosine_mean(targets, preds):
    vals = [cosine_fast(t, p) for t, p in zip(targets, preds)]
    return float(np.mean(vals))


def wasserstein_raw(targets, preds):
    vals = []
    for t, p in zip(targets, preds):
        t_safe = np.maximum(t, 0.0) + EPS
        p_safe = np.maximum(p, 0.0) + EPS
        try:
            d = wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_safe, p_safe)
            vals.append(d)
        except:
            pass
    return float(np.nanmean(vals))


def wasserstein_normalized(targets, preds):
    vals = []
    for t, p in zip(targets, preds):
        t_safe = np.maximum(t, 0.0) + EPS
        p_safe = np.maximum(p, 0.0) + EPS
        t_area = np.trapezoid(t_safe, SIGMA_BINS)
        p_area = np.trapezoid(p_safe, SIGMA_BINS)
        t_norm = t_safe / (t_area + EPS)
        p_norm = p_safe / (p_area + EPS)
        try:
            d = wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_norm, p_norm)
            vals.append(d)
        except:
            pass
    return float(np.nanmean(vals))


def area_mae(targets, preds):
    """Интегральная ошибка площади (AreaMAE) по формуле из литобзора"""
    # Сумма Римана: сумма |t-p| * Δσ
    return np.mean(np.sum(np.abs(targets - preds), axis=1) * DELTA_SIGMA)


def moment_error(targets, preds, moment=1):
    """Ошибка MAE для момента n (нормированного профиля)"""
    vals = []
    for t, p in zip(targets, preds):
        t_safe = np.maximum(t, 0.0) + EPS
        p_safe = np.maximum(p, 0.0) + EPS
        t_norm = t_safe / (np.trapezoid(t_safe, SIGMA_BINS) + EPS)
        p_norm = p_safe / (np.trapezoid(p_safe, SIGMA_BINS) + EPS)
        m_t = np.sum((SIGMA_BINS ** moment) * t_norm * DELTA_SIGMA)
        m_p = np.sum((SIGMA_BINS ** moment) * p_norm * DELTA_SIGMA)
        vals.append(abs(m_t - m_p))
    return float(np.mean(vals))


def molecular_metrics(targets, preds, mol_sizes):
    """Молекулярные метрики (сумма атомов в молекуле)"""
    mol_mae, mol_emd, mol_cos, mol_polar, mol_area = [], [], [], [], []
    idx = 0
    for size in mol_sizes:
        t = targets[idx:idx + size]
        p = preds[idx:idx + size]
        t_mol = t.sum(axis=0)
        p_mol = p.sum(axis=0)

        mol_mae.append(np.mean(np.abs(t_mol - p_mol)))
        mol_polar.append(np.mean(np.abs(t_mol[POLAR_MASK] - p_mol[POLAR_MASK])))
        mol_area.append(np.sum(np.abs(t_mol - p_mol)) * DELTA_SIGMA)

        t_cos = t_mol / (np.linalg.norm(t_mol) + EPS)
        p_cos = p_mol / (np.linalg.norm(p_mol) + EPS)
        mol_cos.append(cosine_fast(t_cos, p_cos))

        t_safe = np.maximum(t_mol, 0.0) + EPS
        p_safe = np.maximum(p_mol, 0.0) + EPS
        try:
            emd = wasserstein_distance(SIGMA_BINS, SIGMA_BINS, t_safe, p_safe)
            mol_emd.append(emd)
        except:
            pass
        idx += size
    return {
        "molecular_mae": float(np.mean(mol_mae)),
        "molecular_emd": float(np.mean(mol_emd)) if mol_emd else 0.0,
        "molecular_cosine": float(np.mean(mol_cos)),
        "molecular_polar_mae": float(np.mean(mol_polar)),
        "molecular_area_mae": float(np.mean(mol_area)),
    }


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# =============================================================================
def evaluate_model(model, loader, bin_weights_np):
    model.eval()
    all_preds = []
    all_targets = []
    mol_sizes = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Предсказание"):
            data = data.to(DEVICE)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            mol_sizes.extend(torch.bincount(data.batch.cpu()).numpy().tolist())

    preds_np = np.maximum(np.concatenate(all_preds, axis=0), 0.0)
    targets_np = np.maximum(np.concatenate(all_targets, axis=0), 0.0)

    metrics = {
        # Взвешенные метрики
        "weighted_r2": weighted_r2(targets_np, preds_np, bin_weights_np),
        "weighted_mae": weighted_mae(targets_np, preds_np, bin_weights_np),
        # Стандартные
        "mae": std_mae(targets_np, preds_np),
        "polar_mae": polar_mae(targets_np, preds_np),
        "sobolev": sobolev_metric(targets_np, preds_np),
        "cosine_similarity": cosine_mean(targets_np, preds_np),
        "emd_raw": wasserstein_raw(targets_np, preds_np),
        "emd_norm": wasserstein_normalized(targets_np, preds_np),
        "area_mae": area_mae(targets_np, preds_np),
        # Ошибки моментов 1-4
        "moment1_mae": moment_error(targets_np, preds_np, 1),
        "moment2_mae": moment_error(targets_np, preds_np, 2),
        "moment3_mae": moment_error(targets_np, preds_np, 3),
        "moment4_mae": moment_error(targets_np, preds_np, 4),
    }
    metrics.update(molecular_metrics(targets_np, preds_np, mol_sizes))
    return metrics


# =============================================================================
# ЗАГРУЗКА КООРДИНАТ КОНФОРМЕРОВ ИЗ PARQUET
# =============================================================================
def load_conformer_coords(parquet_path, method_name):
    logger.info(f"Загрузка координат {method_name} из {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    # Фильтруем только тестовые молекулы (по имени файла предполагается, что там уже test)
    # Проверяем наличие колонок с координатами
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        coord_cols = ['x', 'y', 'z']
    elif 'coord_x' in df.columns:
        coord_cols = ['coord_x', 'coord_y', 'coord_z']
    else:
        raise ValueError(f"Неизвестные имена колонок координат в {parquet_path}")
    # Если есть признак not_converged – отфильтруем
    if 'not_converged' in df.columns:
        valid = ~df['not_converged']
    else:
        valid = True
    df_valid = df[valid].sort_values(['mol_id', 'atom_index'])
    coords_dict = {}
    for mol_id, group in tqdm(df_valid.groupby('mol_id'), desc=f"Сборка {method_name}"):
        coords_dict[mol_id] = group[coord_cols].values
    logger.info(f"Загружено координат для {len(coords_dict)} молекул")
    return coords_dict


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ ДЛЯ РАЗНЫХ КОНФОРМЕРОВ")
    logger.info("=" * 80)

    # 1. Веса бинов на обучающем наборе (для weighted метрик)
    bin_weights = compute_bin_weights_from_parquet(TRAIN_PATH).to(DEVICE)
    bin_weights_np = bin_weights.cpu().numpy()

    # 2. Загрузка модели
    logger.info(f"Загрузка модели из {MODEL_PATH}")
    model = SigmaSchNet(
        hidden_channels=128, num_filters=128, num_interactions=8,
        num_gaussians=50, cutoff=CUTOFF, out_dim=51
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info("Модель загружена")

    # 3. Оригинальный тестовый датасет (DFT координаты)
    logger.info("Загрузка оригинального тестового датасета...")
    test_orig_dataset = TestParquetDataset(TEST_ORIG_PATH)
    # Создаём словарь методов: имя -> путь к parquet с координатами (для методов, кроме Original)
    methods = {
        "Original": None,
        "Conforge": os.path.join(GENERATED_DIR, "chaos_atomic_test_3d_conforge.parquet"),
        "RDKit": os.path.join(GENERATED_DIR, "chaos_atomic_test_3d_rdkit.parquet"),
        "AvgFlow": os.path.join(GENERATED_DIR, "chaos_atomic_test_3d_avgflow.parquet"),
        "OpenBabel": os.path.join(GENERATED_DIR, "chaos_atomic_test_3d_openbabel.parquet"),
    }

    # Загружаем координаты для всех методов (кроме Original)
    coords_dicts = {}
    for name, path in methods.items():
        if path is not None and os.path.exists(path):
            coords_dicts[name] = load_conformer_coords(path, name)
        elif path is not None:
            logger.warning(f"Файл {path} не найден, метод {name} будет пропущен")
            methods[name] = None

    # 4. Создаём датасеты для каждого метода
    datasets = {"Original": test_orig_dataset}
    for name in coords_dicts:
        datasets[name] = ConformerSubstituteDataset(test_orig_dataset, coords_dicts[name], name)

    # 5. Оценка
    batch_size = 32
    all_metrics = {}
    for name, ds in datasets.items():
        logger.info(f"\n{'='*60}\nОЦЕНКА НА {name.upper()}\n{'='*60}")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = evaluate_model(model, loader, bin_weights_np)
        all_metrics[name.lower()] = metrics

    # 6. Вывод таблицы
    logger.info("\n" + "=" * 100)
    headers = list(all_metrics.keys())
    # Получаем все названия метрик из первого метода
    metric_names = list(all_metrics[headers[0]].keys())
    # Формируем строку заголовка
    header_str = f"{'Metric':<30} | " + " | ".join(f"{h:<15}" for h in headers)
    logger.info(header_str)
    logger.info("-" * len(header_str))
    for mname in metric_names:
        row = f"{mname:<30} | "
        row += " | ".join(f"{all_metrics[h][mname]:<15.6f}" for h in headers)
        logger.info(row)
    logger.info("=" * 100)

    # 7. Сохраняем JSON
    final_results = {
        "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
        "test_dataset": TEST_ORIG_PATH,
        "metrics": all_metrics,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"\n✅ Результаты сохранены в {OUTPUT_JSON}")
