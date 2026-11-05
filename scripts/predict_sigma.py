"""predict_sigma.py - предсказание атомных sigma-профилей обученной моделью (sigma_dn_geom и др. 3D-модели бенчмарка).

Работает для любого чекпойнта из src/train_3d_physics.py: конфиг берётся из самого чекпойнта (ckpt["config"]),
поэтому отдельный yaml не нужен. Несколько --checkpoint = ансамбль (среднее предсказаний).

Вход: parquet, одна строка = один атом, колонки
    mol_id, atom_index, element, smiles, coord_x, coord_y, coord_z   (обязательно)
    sigma_0 .. sigma_50                                              (необязательно: если есть - считаются метрики)
Выход:
    <out>                   атомные предсказания: mol_id, atom_index, element, pred_sigma_0..pred_sigma_50
    <out без .parquet>_mol.parquet   молекулярные профили (сумма по атомам): mol_id, n_atoms, pred_sigma_0..50

Пример:
  python -m scripts.predict_sigma \
     --checkpoint results/all_pred_ablation_experiments_results/checkpoints/sigma_dn_geom_seed0_physics.pt \
     --parquet /path/to/my_molecules.parquet --out /path/to/pred/my_molecules_pred.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict_sigma")

REQUIRED = ["mol_id", "atom_index", "element", "smiles", "coord_x", "coord_y", "coord_z"]
SIGMA_COLS = [f"sigma_{i:02d}" for i in range(51)]   # имена, если таргетов во входе нет
PRED_COLS = [f"pred_sigma_{i:02d}" for i in range(51)]


def sigma_cols_of(columns) -> list:
    """Колонки sigma_* в порядке номера бина (sigma_0 и sigma_00 - оба варианта), как в src/data/dataset_3d.py."""
    return sorted([c for c in columns if c.startswith("sigma_")], key=lambda c: int(c.split("_")[1]))


def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:10]


def prepare_input(parquet: str, work_dir: str) -> tuple[str, pd.DataFrame, bool]:
    """Проверяет схему; если нет sigma_* - пишет копию с нулевыми sigma_* (датасет требует 51 колонку).
    Возвращает (путь для датасета, df, есть_ли_таргеты)."""
    df = pd.read_parquet(parquet)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"в {parquet} нет колонок {miss}; нужны {REQUIRED}")
    if df[["coord_x", "coord_y", "coord_z"]].isna().any().any():
        raise ValueError("есть пустые координаты: модель обучалась только на заданной 3D-геометрии")
    from src.data.constants import ELEMENT_TO_Z
    unknown = sorted(set(df["element"]) - set(ELEMENT_TO_Z))
    if unknown:
        raise ValueError(f"неизвестные элементы {unknown}: датасет молча заменил бы их на C")
    found = sigma_cols_of(df.columns)
    if found and len(found) != 51:
        raise ValueError(f"найдено {len(found)} колонок sigma_*, нужно 51 (или ни одной)")
    has_y = len(found) == 51
    os.makedirs(work_dir, exist_ok=True)
    if has_y:
        return parquet, df, True
    tmp = os.path.join(work_dir, f"input_{_file_md5(parquet)}.parquet")
    if not os.path.exists(tmp):
        d2 = df.copy()
        for c in SIGMA_COLS:
            d2[c] = 0.0
        d2.to_parquet(tmp, index=False)
    return tmp, df, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, help="один или несколько .pt (несколько = ансамбль)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True, help="путь к выходному .parquet с атомными предсказаниями")
    ap.add_argument("--cache-dir", default=None,
                    help="папка кэша графов/SASA; по умолчанию <папка out>/cache_<md5 входа>. "
                         "Кэш привязан к ИМЕНИ файла, поэтому для каждого нового входа нужна своя папка")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-parquet", default=None,
                    help="train-сплит для весов бинов в метриках (по умолчанию data.train_path из чекпойнта)")
    ap.add_argument("--cpu", action="store_true", help="считать на CPU даже при наличии GPU")
    args = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    from scripts._model_factory import build_for_eval

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    cache_dir = args.cache_dir or os.path.join(out_dir, f"cache_{_file_md5(args.parquet)}")
    path, df, has_y = prepare_input(args.parquet, cache_dir)

    preds_sum, dataset, sizes = None, None, None
    for k, ck_path in enumerate(args.checkpoint):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model, DatasetCls, kw = build_for_eval(cfg, "3d")
        kw["cache_dir"] = cache_dir
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
        logger.info("модель %s (%s), эпоха %d, %d параметров", cfg["model"]["name"], os.path.basename(ck_path),
                    ck.get("epoch", -1) + 1, sum(p.numel() for p in model.parameters()))
        if dataset is None:
            dataset = DatasetCls(path, **kw)
        preds, sz = [], []
        with torch.no_grad():
            for data in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
                data = data.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    pr = model(data)
                preds.append(pr.float().cpu().numpy())
                sz.extend(torch.bincount(data.batch.cpu()).numpy().tolist())
        p = np.maximum(np.concatenate(preds), 0.0)  # как в src/evaluate.py: отрицательные бины обрезаются нулём
        preds_sum = p if preds_sum is None else preds_sum + p
        sizes = sz
    pred = preds_sum / len(args.checkpoint)

    # порядок атомов = порядок датасета: молекулы в порядке первого появления mol_id, атомы по atom_index
    groups = {m: g.sort_values("atom_index") for m, g in df.groupby("mol_id", sort=False)}
    order = pd.concat([groups[m] for m in dataset.mol_ids], ignore_index=True)
    assert len(order) == len(pred) == sum(sizes), "рассогласование числа атомов"
    assert list(dataset.mol_ids) == list(groups.keys()), "рассогласование порядка mol_id"

    atom = order[["mol_id", "atom_index", "element"]].copy()
    pcols = PRED_COLS
    atom[pcols] = pred
    os.makedirs(out_dir, exist_ok=True)
    atom.to_parquet(args.out, index=False)

    off = np.concatenate([[0], np.cumsum(sizes)])
    mol = pd.DataFrame(np.stack([pred[a:b].sum(0) for a, b in zip(off[:-1], off[1:])]), columns=pcols)
    mol.insert(0, "n_atoms", sizes)
    mol.insert(0, "mol_id", list(dataset.mol_ids))
    mol_out = (args.out[:-8] if args.out.endswith(".parquet") else args.out) + "_mol.parquet"
    mol.to_parquet(mol_out, index=False)
    logger.info("атомы: %s (%d строк); молекулы: %s (%d)", args.out, len(atom), mol_out, len(mol))

    if has_y:
        from src.metrics import weighted_r2, weighted_mae, polar_mae
        ycols = sigma_cols_of(df.columns)
        t = np.maximum(np.nan_to_num(order[ycols].to_numpy(dtype=np.float64)), 0.0)
        # веса бинов бенчмарка = дисперсия бинов на TRAIN-сплите (src/data/dataset.py::bin_weights_numpy)
        train_path = args.train_parquet or cfg.get("data", {}).get("train_path")
        try:
            import pyarrow.parquet as pq
            tcols = sigma_cols_of(pq.read_schema(train_path).names)
            y = np.nan_to_num(pd.read_parquet(train_path, columns=tcols).to_numpy(dtype=np.float64))
            src_w = f"train ({train_path})"
        except Exception as e:  # нет доступа к train - приближённо по самому входу
            logger.warning("не удалось прочитать train для весов бинов (%s); веса по входу, R2/wMAE приблизительные", e)
            y, src_w = np.nan_to_num(order[ycols].to_numpy(dtype=np.float64)), "этот вход"
        var = np.maximum(y.var(0), 1e-12); w = var / var.sum()
        logger.info("метрики (веса бинов: %s): weighted_r2=%.5f weighted_mae=%.5f polar_mae=%.5f", src_w,
                    weighted_r2(t, pred, w), weighted_mae(t, pred, w), polar_mae(t, pred))

if __name__ == "__main__":
    main()
