#!/usr/bin/env python3
"""
Добавляет 3D координаты из JSON-файлов CHAOS в parquet-датасеты.
Использует Coordinates или Coordinates_Input.
Индексация атомов: в parquet atom_index начинается с 0, в JSON список координат тоже с 0.
Логирует все ошибки, продолжает работу.
"""

import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# ==================== НАСТРОЙКИ ====================
PROJECT_ROOT = Path(__file__).parent.resolve()
PARQUET_DIR = PROJECT_ROOT / "data" / "train_test_val_df"
JSON_DIR = PROJECT_ROOT.parent / "chaos"
OUTPUT_DIR = PARQUET_DIR
COORD_TOLERANCE = 1e-6
N_RANDOM_MOLECULES = 10

INPUT_FILES = [
    "chaos_atomic_train.parquet",
    "chaos_atomic_test.parquet",
    "chaos_atomic_val.parquet",
]

ERROR_LOG = PROJECT_ROOT / "error_log.txt"

# ==================== ФУНКЦИИ ====================

def load_json_coordinates(mol_id: str, json_dir: Path) -> Optional[List[List[float]]]:
    """Возвращает список координат [[x,y,z], ...] или None при ошибке."""
    json_path = json_dir / f"{mol_id}.json"
    if not json_path.exists():
        _log_error(mol_id, f"JSON файл не найден: {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        _log_error(mol_id, f"Ошибка чтения JSON: {e}")
        return None

    structural = data.get("structural", {})
    coords = structural.get("Coordinates")
    if coords is None:
        coords = structural.get("Coordinates_Input")

    if coords is None:
        _log_error(mol_id, "Нет ни Coordinates, ни Coordinates_Input")
        return None

    if not isinstance(coords, list) or len(coords) == 0:
        _log_error(mol_id, f"Координаты не список или пусты: {type(coords)}")
        return None

    # Проверяем, что каждый элемент — список из 3 чисел
    for i, atom in enumerate(coords[:5]):
        if not isinstance(atom, (list, tuple)) or len(atom) != 3:
            _log_error(mol_id, f"Неверный формат атома {i}: {atom}")
            return None

    return coords

def _log_error(mol_id: str, message: str):
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{mol_id}: {message}\n")
        f.flush()
    print(f"  [ОШИБКА] {mol_id}: {message}")

def add_coordinates_to_df(df: pd.DataFrame, json_dir: Path) -> pd.DataFrame:
    unique_mol_ids = df['mol_id'].unique()
    print(f"  Загрузка координат для {len(unique_mol_ids)} уникальных молекул...")
    
    mol_coords: Dict[str, Optional[List[List[float]]]] = {}
    for i, mol_id in enumerate(unique_mol_ids):
        if (i + 1) % 5000 == 0:
            print(f"    Обработано {i+1} / {len(unique_mol_ids)} молекул")
        mol_coords[mol_id] = load_json_coordinates(mol_id, json_dir)
    
    print(f"  Добавление координат в DataFrame...")
    
    def get_coords(row):
        coords_list = mol_coords.get(row['mol_id'])
        if coords_list is None:
            return pd.Series([None, None, None], index=['coord_x', 'coord_y', 'coord_z'])
        idx = row['atom_index']          # Исправлено: теперь без -1, т.к. atom_index начинается с 0
        if idx < 0 or idx >= len(coords_list):
            _log_error(row['mol_id'], f"atom_index={idx} вне диапазона (0..{len(coords_list)-1})")
            return pd.Series([None, None, None], index=['coord_x', 'coord_y', 'coord_z'])
        x, y, z = coords_list[idx]
        return pd.Series([x, y, z], index=['coord_x', 'coord_y', 'coord_z'])

    df[['coord_x', 'coord_y', 'coord_z']] = df.apply(get_coords, axis=1)
    return df

def verify_coordinates(df: pd.DataFrame, json_dir: Path, num_samples: int = N_RANDOM_MOLECULES):
    """Проверяет случайные молекулы, у которых координаты успешно добавлены."""
    # Молекулы, у которых все координаты не NaN
    good_mols = df.groupby('mol_id')['coord_x'].apply(lambda x: not x.isna().any())
    good_mol_ids = good_mols[good_mols].index.tolist()
    
    if not good_mol_ids:
        print("  Нет молекул с валидными координатами для проверки.")
        return
    
    sample_ids = random.sample(good_mol_ids, min(num_samples, len(good_mol_ids)))
    print(f"\n  Проверка {len(sample_ids)} случайных молекул...")
    
    all_ok = True
    for mol_id in sample_ids:
        json_coords = load_json_coordinates(mol_id, json_dir)
        if json_coords is None:
            print(f"    ❌ {mol_id}: не удалось загрузить координаты из JSON")
            all_ok = False
            continue
        
        mol_df = df[df['mol_id'] == mol_id].sort_values('atom_index')
        if len(mol_df) != len(json_coords):
            print(f"    ❌ {mol_id}: несовпадение числа атомов (df={len(mol_df)}, JSON={len(json_coords)})")
            all_ok = False
            continue
        
        mismatch = False
        for idx, (_, row) in enumerate(mol_df.iterrows()):
            px, py, pz = row['coord_x'], row['coord_y'], row['coord_z']
            jx, jy, jz = json_coords[idx]
            if (abs(px - jx) > COORD_TOLERANCE or
                abs(py - jy) > COORD_TOLERANCE or
                abs(pz - jz) > COORD_TOLERANCE):
                print(f"    ❌ {mol_id}: атом {row['atom_index']} координаты не совпадают:\n"
                      f"       parquet: ({px:.6f}, {py:.6f}, {pz:.6f})\n"
                      f"       JSON:    ({jx:.6f}, {jy:.6f}, {jz:.6f})")
                mismatch = True
                all_ok = False
                break
        if not mismatch:
            print(f"    ✅ {mol_id}: OK (атомов {len(mol_df)})")
    
    if all_ok:
        print("  Все проверенные молекулы прошли успешно.")
    else:
        print("  Внимание: обнаружены несоответствия (см. выше).")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Очищаем лог
    with open(ERROR_LOG, 'w', encoding='utf-8') as f:
        f.write("# Лог ошибок при загрузке координат\n# mol_id : причина\n")
    
    for filename in INPUT_FILES:
        input_path = PARQUET_DIR / filename
        if not input_path.exists():
            print(f"Предупреждение: {input_path} не найден, пропускаем.")
            continue
        
        print(f"\n=== Обработка {filename} ===")
        df = pd.read_parquet(input_path)
        print(f"  Загружено строк: {len(df)}, уникальных молекул: {df['mol_id'].nunique()}")
        
        df_with_coords = add_coordinates_to_df(df, JSON_DIR)
        
        missing = df_with_coords['coord_x'].isna().sum()
        print(f"  Строк с NaN координатами: {missing} из {len(df_with_coords)} ({100*missing/len(df_with_coords):.2f}%)")
        
        output_name = filename.replace(".parquet", "_with_coordinates.parquet")
        output_path = OUTPUT_DIR / output_name
        df_with_coords.to_parquet(output_path, index=False)
        print(f"  Сохранён: {output_path}")
        
        if missing < len(df_with_coords):
            verify_coordinates(df_with_coords, JSON_DIR)
        else:
            print("  Нет валидных координат для проверки (все строки NaN).")
    
    print(f"\n✅ Готово! Лог ошибок: {ERROR_LOG}")

if __name__ == "__main__":
    main()
