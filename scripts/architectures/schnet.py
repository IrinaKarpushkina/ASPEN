import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import math
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.utils.data import random_split
import logging
from datetime import datetime
import traceback

# ─── НАСТРОЙКА ЛОГИРОВАНИЯ ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("schnet_classic_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── ПУТИ ────────────────────────────────────────────────────────────────────────
CSV_PATH = "/mnt/tank/scratch/ikarpushkina/sigma/chaos/atomic_profiles/"
JSON_PATH = "/mnt/tank/scratch/ikarpushkina/sigma/chaos/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── ФИЗИЧЕСКИЕ СПРАВОЧНИКИ ─────────────────────────────────────────────────────
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06,
    19: 2.03, 20: 1.76, 35: 1.20, 53: 1.39
}
PAULING_EN = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 35: 2.96, 53: 2.66
}
VDW_RADII = {
    1: 1.20, 2: 1.40, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 10: 1.54,
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88,
    19: 2.75, 20: 2.31, 35: 1.85, 53: 1.98
}

# ─── МОДЕЛЬ: чистый SchNet + простой MLP readout ────────────────────────────────
class ClassicSchNet(nn.Module):
    def __init__(self,
                 hidden_channels=128,
                 num_filters=128,
                 num_interactions=6,
                 num_gaussians=50,
                 cutoff=10.0,
                 out_dim=51):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=32,
            readout='add'  # игнорируется
        )

        # Простой MLP readout — без KAN
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_dim)
        )

        self.softplus = nn.Softplus()  # p(σ) ≥ 0

    def forward(self, data):
        # Получаем атомные эмбеддинги после всех interaction блоков
        h = self.schnet.embedding(data.z)

        # Строим граф и edge features (как в оригинальном SchNet)
        edge_index, edge_weight = self.schnet.interaction_graph(data.pos, data.batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        # Проходим через interaction блоки
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Защита от одиночных атомов
        if h.dim() == 1:
            h = h.unsqueeze(0)

        out = self.readout(h)
        return self.softplus(out)

# ─── ДАТАСЕТ (без изменений) ────────────────────────────────────────────────────
class ChaosDataset(Dataset):
    def __init__(self, csv_dir, json_dir, use_cache=True):
        super().__init__()
        self.csv_dir = csv_dir
        self.json_dir = json_dir
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        self.use_cache = use_cache
        self.cache = [] if use_cache else None
        mol_ids = [f.split('_')[0] for f in self.csv_files]
        if len(set(mol_ids)) != len(mol_ids):
            logger.warning("Обнаружены дублирующиеся mol_id! Возможна утечка данных.")
        if use_cache:
            logger.info("Загрузка данных в кэш...")
            for idx in tqdm(range(len(self.csv_files)), desc="Caching data"):
                self.cache.append(self._load_data(idx))
            logger.info("Кэширование завершено.")

    def len(self):
        return len(self.csv_files)

    def _load_data(self, idx):
        csv_file = self.csv_files[idx]
        try:
            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))
            df = df.sort_values(by='atom_index')
            y = torch.tensor(df.iloc[:, 3:54].values, dtype=torch.float)
            y = torch.nan_to_num(y, nan=0.0)

            mol_id = csv_file.split('_')[0]
            with open(os.path.join(self.json_dir, f"{mol_id}.json"), 'r') as f:
                jdata = json.load(f)

            atom_list = sorted(jdata['general']['AtomList'], key=lambda a: a['index'])
            atomic_nums = [a['atomic_number'] for a in atom_list]
            n_atoms = len(atomic_nums)

            struct = jdata.get('structural', {})
            coords_list = struct.get('Coordinates') or struct.get('Coordinates_Input')
            coords = np.array(coords_list) if coords_list else np.zeros((n_atoms, 3))

            if n_atoms == 0:
                logger.warning(f"Молекула {mol_id} без атомов → dummy C")
                atomic_nums = [6]
                coords = np.zeros((1, 3))
                y = torch.zeros((1, 51))
                n_atoms = 1

            z = torch.tensor(atomic_nums, dtype=torch.long)
            pos = torch.tensor(coords, dtype=torch.float)

            if y.shape[0] != n_atoms:
                logger.warning(f"y {y.shape[0]} != атомов {n_atoms} → корректировка")
                if y.shape[0] > n_atoms:
                    y = y[:n_atoms]
                else:
                    pad = torch.zeros((n_atoms - y.shape[0], 51))
                    y = torch.cat([y, pad], dim=0)

            global_feats = self._compute_global_features(atomic_nums, coords)

            return Data(
                z=z,
                pos=pos,
                y=y,
                global_feats=global_feats
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке {csv_file}: {e}")
            return Data(
                z=torch.tensor([6], dtype=torch.long),
                pos=torch.zeros((1, 3), dtype=torch.float),
                y=torch.zeros((1, 51), dtype=torch.float),
                global_feats=torch.zeros((1, 2), dtype=torch.float)
            )

    def _compute_global_features(self, atomic_nums, coords):
        n = len(atomic_nums)
        en_vals = np.array([PAULING_EN.get(z, 2.5) for z in atomic_nums])
        mean_en_norm = np.mean(en_vals) / 4.0 if n > 0 else 0.0
        diameter = 0.0
        if n > 1:
            dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            diameter = np.max(dist_matrix) / 20.0
        return torch.tensor([[diameter, mean_en_norm]], dtype=torch.float)

    def get(self, idx):
        if self.use_cache:
            return self.cache[idx]
        return self._load_data(idx)

# ─── ФУНКЦИИ ОЦЕНКИ ──────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    num_batches = 0
    with torch.no_grad():
        for data in loader:
            try:
                data = data.to(DEVICE)
                preds = torch.nan_to_num(model(data), nan=0.0, posinf=1.0, neginf=0.0)
                loss = criterion(preds, data.y)
                total_loss += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())
                num_batches += 1
            except Exception as e:
                logger.error(f"Ошибка при валидации батча: {e}\n{traceback.format_exc()}")
                continue
    if num_batches == 0:
        logger.warning("Нет данных для валидации")
        return 0.0, 0.0
    try:
        all_targets_np = np.concatenate(all_targets, axis=0)
        all_preds_np = np.concatenate(all_preds, axis=0)
        mask = ~np.isnan(all_targets_np).any(axis=1)
        if mask.sum() > 0:
            r2 = r2_score(all_targets_np[mask], all_preds_np[mask], multioutput='variance_weighted')
        else:
            r2 = 0.0
    except Exception as e:
        logger.error(f"Ошибка при вычислении R²: {e}\n{traceback.format_exc()}")
        r2 = 0.0
    return total_loss / num_batches, r2

def test_model(model, loader, criterion):
    logger.info("="*60)
    logger.info("ТЕСТИРОВАНИЕ МОДЕЛИ")
    logger.info("="*60)
    test_loss, test_r2 = evaluate(model, loader, criterion)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    return test_loss, test_r2

# ─── ЗАПУСК ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("Инициализация датасета...")
    full_dataset = ChaosDataset(CSV_PATH, JSON_PATH, use_cache=True)
    logger.info(f"Всего образцов: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    sample = full_dataset[0]
    out_dim = sample.y.shape[1]  # 51

    model = ClassicSchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        out_dim=out_dim
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.HuberLoss(delta=1.0)

    logger.info(f"Classic SchNet модель: hidden=128, interactions=6, cutoff=10.0 Å (без KAN)")
    logger.info(f"Data split: {len(train_dataset)} Train | {len(val_dataset)} Validation | {len(test_dataset)} Test")

    best_val_r2 = -float('inf')
    best_epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patience_counter = 0
    patience = 10

    for epoch in range(100):
        model.train()
        train_loss = 0
        num_batches = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            try:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                preds = torch.nan_to_num(model(data), nan=0.0, posinf=1.0, neginf=0.0)
                loss = criterion(preds, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.error(f"Ошибка при обучении на батче: {e}\n{traceback.format_exc()}")
                continue

        avg_train_loss = train_loss / max(num_batches, 1)
        val_loss, val_r2 = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val R²: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            model_path = f"best_schnet_classic_{timestamp}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'config': {
                    'hidden_channels': 128,
                    'num_interactions': 6,
                    'out_dim': out_dim,
                    'cutoff': 10.0
                }
            }, model_path)
            logger.info(f"✨ Новая лучшая модель сохранена: {model_path} с R²={val_r2:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping на эпохе {epoch+1}")
            break

    logger.info(f"Обучение завершено. Лучшая эпоха: {best_epoch}, Val R²: {best_val_r2:.4f}")

    logger.info("\n" + "="*60)
    logger.info("ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ ДЛЯ ТЕСТИРОВАНИЯ")
    logger.info("="*60)

    checkpoint = torch.load(f"best_schnet_classic_{timestamp}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_r2 = test_model(model, test_loader, criterion)

    final_metrics = {
        'best_val_r2': best_val_r2,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_r2': test_r2,
        'timestamp': timestamp
    }

    with open(f'training_metrics_schnet_classic_{timestamp}.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"✅ Обучение завершено. Метрики сохранены в training_metrics_schnet_classic_{timestamp}.json")
