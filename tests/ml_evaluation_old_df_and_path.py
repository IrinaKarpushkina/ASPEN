import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

# Модели и инструменты
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Отключаем лишние предупреждения
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# Пути
BASE_DIR = "/mnt/tank/scratch/ikarpushkina/sigma/"
sys.path.append(os.path.join(BASE_DIR, 'dekan'))

from schnet import ClassicSchNet, ChaosDataset, PAULING_EN, VDW_RADII
from torch_geometric.loader import DataLoader

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
CHAOS_JSON_DIR = os.path.join(BASE_DIR, "chaos/")
TRUE_PROFILES_DIR = os.path.join(BASE_DIR, "chaos/atomic_profiles/") 
MODEL_WEIGHTS = os.path.join(BASE_DIR, "dekan/best_schnet_classic_20260309_125151.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_results_final/")

TARGETS = {
    "HOMO": ("electronic", "HOMOEnergy"),
    "LUMO": ("electronic", "LUMOEnergy"),
    "HLG": ("electronic", "HLG"),
    "DipoleMoment": ("electronic", "DipoleMoment")
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def pool_atomic_profiles(matrix):
    """Агрегация (N_atoms, 51) -> (204,)"""
    if matrix is None or matrix.shape[0] == 0: return np.zeros(204)
    return np.concatenate([
        np.sum(matrix, axis=0), 
        np.mean(matrix, axis=0), 
        np.max(matrix, axis=0), 
        np.min(matrix, axis=0)
    ])

def get_classical_descriptors(atomic_nums, coords):
    """
    Генерирует только НЕКОРРЕЛИРОВАННЫЕ классические дескрипторы (4 признака).
    Выбраны на основе отчета: N_atoms, Mean_EN, Max_EN, Mean_VDW.
    """
    n = len(atomic_nums)
    if n == 0: return np.zeros(4)
    
    en_vals = [PAULING_EN.get(z, 2.1) for z in atomic_nums]
    vdw_vals = [VDW_RADII.get(z, 1.5) for z in atomic_nums]
    
    return np.array([
        n,              # N_atoms (Размер)
        np.mean(en_vals), # Mean_EN (Химическая природа)
        np.max(en_vals),  # Max_EN (Наличие активного центра)
        np.mean(vdw_vals) # Mean_VDW (Средний объем атома)
    ])

# ==========================================
# 3. ЗАГРУЗКА И ИНФЕРЕНС
# ==========================================

def load_data_and_predict():
    csv_list = glob.glob(os.path.join(TRUE_PROFILES_DIR, "*_atomic_sigma.csv"))
    csv_map = {os.path.basename(f).replace('_atomic_sigma.csv', ''): f for f in csv_list}
    
    raw_data = []
    json_files = glob.glob(os.path.join(CHAOS_JSON_DIR, "*.json"))
    
    for jf in tqdm(json_files, desc="Loading Data"):
        mid = os.path.basename(jf).replace('.json', '')
        if mid not in csv_map: continue
        try:
            with open(jf, 'r') as f: jd = json.load(f)
            nums = [a['atomic_number'] for a in jd['general']['AtomList']]
            pos = np.array(jd['structural'].get('Coordinates') or jd['structural'].get('Coordinates_Input'))
            
            # Дескрипторы
            row = {
                "id": mid, 
                "classical": get_classical_descriptors(nums, pos)
            }
            # Таргеты
            for t_name, t_path in TARGETS.items(): 
                row[t_name] = jd[t_path[0]][t_path[1]]
            
            # True Sigma из CSV
            csv_data = pd.read_csv(csv_map[mid]).iloc[:, 3:54].values
            row["true_sigma"] = pool_atomic_profiles(csv_data)
            raw_data.append(row)
        except: continue
    
    df = pd.DataFrame(raw_data)
    
    # Инференс нейросети
    model = ClassicSchNet(out_dim=51).to(device)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = ChaosDataset(TRUE_PROFILES_DIR, CHAOS_JSON_DIR, use_cache=False)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    gnn_res = {}
    curr = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="GNN Inference"):
            batch = batch.to(device)
            out = model(batch).cpu().numpy()
            b_idx = batch.batch.cpu().numpy()
            for i in range(batch.num_graphs):
                # Извлекаем ID из списка файлов датасета
                mid = dataset.csv_files[curr].replace('_atomic_sigma.csv', '')
                gnn_res[mid] = pool_atomic_profiles(out[b_idx == i])
                curr += 1
    
    df['pred_sigma'] = df['id'].map(gnn_res)
    df = df.dropna().copy()
    
    # Комбинированные наборы (4 классики + 204 сигмы = 208 признаков)
    df['combined_true'] = [np.concatenate([c, t]) for c, t in zip(df['classical'], df['true_sigma'])]
    df['combined_pred'] = [np.concatenate([c, p]) for c, p in zip(df['classical'], df['pred_sigma'])]
    
    return df

# ==========================================
# 4. НАСТРОЙКА МОДЕЛЕЙ
# ==========================================

def get_model_pipeline(name):
    if name == "Ridge":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1.0))
        ])
    elif name == "HGBR":
        return HistGradientBoostingRegressor(max_iter=300, random_state=42)
    elif name == "XGBoost":
        return XGBRegressor(n_estimators=200, learning_rate=0.1, n_jobs=-1, random_state=42)
    elif name == "RandomForest":
        return RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)

# ==========================================
# 5. ГЛАВНЫЙ ЦИКЛ ОЦЕНКИ
# ==========================================

if __name__ == "__main__":
    # Загружаем всё
    df = load_data_and_predict()
    
    feature_sets = {
        "Classical": "classical",
        "True_Sigma": "true_sigma",
        "Pred_Sigma": "pred_sigma",
        "Combined_True": "combined_true",
        "Combined_Pred": "combined_pred"
    }
    
    colors = {
        "Classical": "#95a5a6", 
        "True_Sigma": "#2980b9", 
        "Pred_Sigma": "#27ae60", 
        "Combined_True": "#8e44ad", 
        "Combined_Pred": "#d35400"
    }

    all_results = {}
    model_names = ["Ridge", "HGBR", "XGBoost", "RandomForest"]

    for target in TARGETS.keys():
        print(f"\n--- Processing Target: {target} ---")
        y = df[target].values
        all_results[target] = {}

        for m_name in model_names:
            print(f"  Training model: {m_name}")
            
            # Создаем общее полотно для сравнения фич
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(f"Target: {target} | Model: {m_name} (Features Comparison)", fontsize=16)
            
            for i, (fs_name, fs_col) in enumerate(feature_sets.items()):
                X = np.nan_to_num(np.vstack(df[fs_col].values))
                
                # Кросс-валидация 5-fold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_p = np.zeros_like(y)
                
                for train_idx, test_idx in kf.split(X):
                    model = get_model_pipeline(m_name)
                    model.fit(X[train_idx], y[train_idx])
                    y_p[test_idx] = model.predict(X[test_idx])
                
                # Считаем метрики
                met = {
                    "R2": float(r2_score(y, y_p)), 
                    "MAE": float(mean_absolute_error(y, y_p)),
                    "RMSE": float(np.sqrt(mean_squared_error(y, y_p)))
                }
                all_results[target][f"{m_name}_{fs_name}"] = met
                
                # Отрисовка на общем полотне
                axes[i].scatter(y, y_p, alpha=0.2, color=colors[fs_name], s=10)
                axes[i].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                axes[i].set_title(f"{fs_name}\n$R^2$={met['R2']:.3f}")
                
                # Сохраняем отдельный детальный график
                plt.figure(figsize=(6,6))
                plt.scatter(y, y_p, alpha=0.3, color=colors[fs_name], edgecolor='k', s=15)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                plt.title(f"{target} | {m_name} | {fs_name}\n$R^2$={met['R2']:.3f}")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(OUTPUT_DIR, f"detail_{target}_{m_name}_{fs_name}.png"), dpi=150)
                plt.close()

            # Сохраняем суммарный график по модели
            plt.figure(fig.number)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_DIR, f"summary_{target}_{m_name}.png"), dpi=200)
            plt.close()

    # Сохраняем все метрики в один JSON
    with open(os.path.join(OUTPUT_DIR, "full_comparison_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✅ Done! All results, metrics and plots are in {OUTPUT_DIR}")
