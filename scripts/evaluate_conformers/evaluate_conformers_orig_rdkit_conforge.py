import os
import sys

# ─── ИСПРАВЛЕНИЕ ПУТЕЙ (РАБОТАЕТ ПРИ ЗАПУСКЕ ОТКУДА УГОДНО) ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))

sys.path.append(SCRIPTS_DIR)

from architectures.schnet import ClassicSchNet, ChaosDataset, DEVICE

import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import logging
import traceback
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from datetime import datetime

# ─── НАСТРОЙКА ЛОГИРОВАНИЯ ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── ПУТИ К ДАННЫМ ──────────────────────────────────────────────────────────────
CSV_PATH = "/mnt/tank/scratch/ikarpushkina/sigma/chaos/atomic_profiles/"
JSON_PATH = "/mnt/tank/scratch/ikarpushkina/sigma/chaos/"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classic_schnet_random_split.pt")
CONFORGE_PATH = os.path.join(PROJECT_ROOT, "data", "generated_conformers_df", "chaos_atomic_sigma_3d_conforge.parquet")
RDKIT_PATH = os.path.join(PROJECT_ROOT, "data", "generated_conformers_df", "chaos_atomic_sigma_3d_rdkit.parquet")
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "scripts/evaluate_conformers", "conformers_evaluation_metrics.json")

# ─── КОНСТАНТЫ СЕТКИ СИГМА-ПРОФИЛЯ ──────────────────────────────────────────────
SIGMA_BINS = np.linspace(-0.025, 0.025, 51)
DELTA_SIGMA = float(SIGMA_BINS[1] - SIGMA_BINS[0])
POLAR_THRESHOLD = 0.01
POLAR_MASK = ((SIGMA_BINS <= -POLAR_THRESHOLD) | (SIGMA_BINS >= POLAR_THRESHOLD))
EPS = 1e-12

# ─── ФИЗИЧЕСКИЕ МЕТРИКИ ─────────────────────────────────────────────────────────
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
        # Заменили trapz на trapezoid для совместимости с новыми версиями numpy
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

def molecular_metrics(targets, preds, mol_sizes):
    mol_mae, mol_emd, mol_cos, mol_polar = [], [], [], []
    idx = 0
    for size in mol_sizes:
        t = targets[idx:idx + size]
        p = preds[idx:idx + size]
        t_mol = t.sum(axis=0)
        p_mol = p.sum(axis=0)

        mol_mae.append(np.mean(np.abs(t_mol - p_mol)))
        mol_polar.append(np.mean(np.abs(t_mol[POLAR_MASK] - p_mol[POLAR_MASK])))
        
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

    # Значения уже обернуты в float()
    return {
        "molecular_mae": float(np.mean(mol_mae)),
        "molecular_emd": float(np.mean(mol_emd)),
        "molecular_cosine": float(np.mean(mol_cos)),
        "molecular_polar_mae": float(np.mean(mol_polar))
    }

# ─── ФУНКЦИЯ ОЦЕНКИ ─────────────────────────────────────────────────────────────
def evaluate_metrics(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, mol_sizes = [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Оценка батчей"):
            try:
                data = data.to(DEVICE)
                preds = model(data)
                
                loss = criterion(preds, data.y)
                total_loss += loss.item()

                all_preds.append(preds.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())

                bincount = torch.bincount(data.batch.cpu()).numpy()
                mol_sizes.extend(bincount.tolist())
            except Exception as e:
                logger.error(f"Ошибка: {e}\n{traceback.format_exc()}")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    preds = np.maximum(preds, 0.0)
    targets = np.maximum(targets, 0.0)

    mask = ~np.isnan(targets).any(axis=1)
    if mask.sum() > 0:
        r2 = r2_score(targets[mask], preds[mask], multioutput='variance_weighted')
    else:
        r2 = 0.0

    # Явное оборачивание всех метрик в float для сериализации JSON
    metrics = {
        "huber_loss": float(total_loss / len(loader)),
        "r2_score": float(r2),
        "mae": float(std_mae(targets, preds)),
        "polar_mae": float(polar_mae(targets, preds)),
        "sobolev": float(sobolev_metric(targets, preds)),
        "cosine_similarity": float(cosine_mean(targets, preds)),
        "emd_raw": float(wasserstein_raw(targets, preds)),
        "emd_norm": float(wasserstein_normalized(targets, preds))
    }

    metrics.update(molecular_metrics(targets, preds, mol_sizes))
    return metrics

# ─── ПОДМЕНА КООРДИНАТ ──────────────────────────────────────────────────────────
def load_generated_coords(parquet_path, name):
    logger.info(f"Загрузка координат {name} из {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    valid_mask = (~df['not_converged']) & (df[['x', 'y', 'z']].notna().all(axis=1))
    df_valid = df[valid_mask].sort_values(by=['mol_id', 'atom_index'])
    
    coords_dict = {}
    for mol_id, group in tqdm(df_valid.groupby('mol_id'), desc=f"Сборка словаря {name}"):
        coords_dict[mol_id] = group[['x', 'y', 'z']].values
        
    return coords_dict

class SubstituteCoordsDataset(Dataset):
    def __init__(self, base_dataset, test_indices, coords_dict, name=""):
        super().__init__()
        self.data_list = []
        
        for idx in test_indices:
            csv_file = base_dataset.csv_files[idx]
            mol_id = csv_file.split('_')[0]
            orig_data = base_dataset.get(idx)
            
            if mol_id in coords_dict:
                coords = coords_dict[mol_id]
                if coords.shape[0] == orig_data.z.shape[0]:
                    new_data = orig_data.clone()
                    new_data.pos = torch.tensor(coords, dtype=torch.float)
                    self.data_list.append(new_data)

    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

# ─── ОСНОВНОЙ КОД ЗАПУСКА ───────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("1. Инициализация датасета и разбиения (use_cache=False)...")
    full_dataset = ChaosDataset(CSV_PATH, JSON_PATH, use_cache=False)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    test_indices = test_ds.indices
    
    logger.info("2. Загрузка сгенерированных координат...")
    conforge_coords = load_generated_coords(CONFORGE_PATH, "Conforge")
    rdkit_coords = load_generated_coords(RDKIT_PATH, "RDKit")
    
    logger.info("3. Формирование тестовых датасетов (с подменой pos)...")
    test_ds_orig = test_ds
    test_ds_conforge = SubstituteCoordsDataset(full_dataset, test_indices, conforge_coords, "Conforge")
    test_ds_rdkit = SubstituteCoordsDataset(full_dataset, test_indices, rdkit_coords, "RDKit")
    
    batch_size = 32
    loader_orig = DataLoader(test_ds_orig, batch_size=batch_size, shuffle=False)
    loader_conforge = DataLoader(test_ds_conforge, batch_size=batch_size, shuffle=False)
    loader_rdkit = DataLoader(test_ds_rdkit, batch_size=batch_size, shuffle=False)
    
    logger.info("4. Инициализация модели и загрузка весов...")
    model = ClassicSchNet(
        hidden_channels=128, num_filters=128, num_interactions=6,
        num_gaussians=50, cutoff=10.0, out_dim=51
    ).to(DEVICE)
    
    # weights_only=False чтобы не было Warning'ов при загрузке словарей из файла
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    criterion = nn.HuberLoss(delta=1.0)
    
    logger.info("\n" + "="*60 + "\nОЦЕНКА НА ОРИГИНАЛЬНЫХ КООРДИНАТАХ\n" + "="*60)
    metrics_orig = evaluate_metrics(model, loader_orig, criterion)
    
    logger.info("\n" + "="*60 + "\nОЦЕНКА НА КООРДИНАТАХ CONFORGE\n" + "="*60)
    metrics_conf = evaluate_metrics(model, loader_conforge, criterion)
    
    logger.info("\n" + "="*60 + "\nОЦЕНКА НА КООРДИНАТАХ RDKIT\n" + "="*60)
    metrics_rdkit = evaluate_metrics(model, loader_rdkit, criterion)
    
    # ─── КРАСИВЫЙ ВЫВОД ───────────────────────────────────────────────────────────
    logger.info("\n" + "="*85)
    logger.info(f"{'Метрика':<25} | {'Original':<15} | {'Conforge':<15} | {'RDKit':<15}")
    logger.info("-" * 85)
    for key in metrics_orig.keys():
        logger.info(f"{key:<25} | {metrics_orig[key]:<15.5f} | {metrics_conf[key]:<15.5f} | {metrics_rdkit[key]:<15.5f}")
    logger.info("=" * 85)

    final_results = {
        "original_json": metrics_orig,
        "conforge": metrics_conf,
        "rdkit": metrics_rdkit,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Сохраняем в папку data/
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_results, f, indent=4)
    logger.info(f"✅ Результаты сохранены в {OUTPUT_JSON_PATH}")
