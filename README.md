# Sigma-profile GNN benchmark — controlled comparison

Структура для честного (controlled, fixed-parameter-budget) сравнения
GNN-архитектур при предсказании атомарных σ-профилей, следуя протоколу
Dwivedi et al. (JMLR 2022, "Benchmarking Graph Neural Networks").

## Принципы

1. **Level 1 — controlled comparison**: GCN, GAT, GATv2, GINE, SchNet и
   финальная модель обучаются с ОДИНАКОВЫМ: loss (MSE), scheduler
   (CosineAnnealingWarmRestarts), batch size (24, grad_accum=2 → eff. 48),
   cutoff (12 Å), max_num_neighbors (32), node/edge признаками, и
   параметрическим бюджетом (~695k, ±5%). Любая разница в метриках —
   архитектурный эффект, а не следствие разных гиперпараметров.

2. **Level 2 — loss ablation**: финальная модель обучается с MSE
   (как все baseline) и с CombinedLoss (6-компонентный, см.
   `src/losses/combined.py`). Это изолирует вклад вашего loss-дизайна
   от вклада архитектуры.

3. **Единый источник правды**: все константы (`src/data/constants.py`),
   построение признаков (`src/data/features.py`), метрики
   (`src/metrics.py`), output head (`src/models/layers.py`) и training
   loop (`src/train.py`) используются ВСЕМИ моделями. Изменить что-то
   для одной модели = изменить для всех.

## Структура

```
ASPEN/
├── configs/
│   ├── base.yaml          # ОБЩИЕ настройки (пути, batch, lr, scheduler...)
│   ├── gcn.yaml            # model.* — только архитектурно-специфичное
│   ├── gat.yaml
│   ├── gatv2.yaml
│   ├── gine.yaml
│   ├── schnet.yaml
│   └── final_model.yaml    # ЗАМЕНИТЕ src/models/final_model.py на вашу архитектуру
├── src/
│   ├── config.py           # загрузка/слияние base.yaml + model.yaml
│   ├── train.py             # единый training entrypoint
│   ├── evaluate.py          # единый evaluation loop
│   ├── metrics.py            # все метрики (weighted_mae, EMD, molecular_*, ...)
│   ├── data/
│   │   ├── constants.py     # ELEMENT_TO_Z, SIGMA_BINS, CUTOFF, ...
│   │   ├── features.py      # node/edge features, pure-PyTorch radius_graph
│   │   └── dataset.py        # ChaosParquetDataset (precompute + disk cache)
│   ├── losses/
│   │   ├── simple.py         # MSELoss
│   │   └── combined.py        # CombinedLoss (для ablation)
│   └── models/
│       ├── layers.py          # ResidualMLP (общий output head)
│       ├── gcn.py / gat.py / gatv2.py / gine.py / schnet.py
│       └── final_model.py      # ЗАМЕНИТЬ на актуальную архитектуру
├── scripts/
│   ├── count_params.py        # таблица параметров + авто-подбор hidden
│   ├── run_benchmark.sh        # Level 1: все baseline, N seed-ов
│   ├── run_ablation.sh         # Level 2: final model, MSE vs Combined
│   └── aggregate_results.py    # results/metrics/*.json -> сводная таблица
├── requirements.txt
└── results/                    # создаётся автоматически
    ├── checkpoints/<model>_seed<N>_<loss>.pt
    └── metrics/<model>_seed<N>_<loss>.json
```

## Установка на сервере

```bash
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN

# Распакуйте архив сюда (он принесёт src/, configs/, scripts/, requirements.txt,
# README.md — поверх существующих data/, models/, scripts/*, tests/).
# ВНИМАНИЕ: в архиве scripts/ содержит НОВЫЕ файлы (count_params.py,
# run_benchmark.sh, run_ablation.sh, aggregate_results.py, __init__.py).
# Если в вашем scripts/ уже есть подпапки architectures/, data_processing/
# и т.д. — они не затрутся, новые файлы лягут рядом.

pip install --break-system-packages -r requirements.txt

# (опционально, но рекомендуется) — RDKit нужен для extended-признаков
# (гибридизация, формальный заряд, ароматичность, Gasteiger charges, типы связей).
python -c "import rdkit; print(rdkit.__version__)"
```

torch_scatter / torch_cluster / pyg-lib **не требуются** — radius_graph
реализован на чистом PyTorch в `src/data/features.py::radius_graph_pure`,
чтобы не зависеть от компилируемых расширений, привязанных к конкретной
версии CUDA/torch.

## Шаг 1 — проверка параметрических бюджетов

```bash
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN
python -m scripts.count_params --auto-tune --csv results/param_budget_table.csv
```

Текущие значения (с extended-признаками, target=695,667):

| model  | hidden          | params  | diff   |
|--------|-----------------|---------|--------|
| gcn    | 256             | 699,635 | +0.57% |
| gat    | 256             | 701,683 | +0.86% |
| gatv2  | 216             | 690,981 | -0.67% |
| gine   | 213             | 706,288 | +1.53% |
| schnet | hidden_channels=150 | 693,202 | -0.35% |

Всё в пределах ±2% — для статьи можно использовать как есть, либо
применить значения из колонки `-> suggest ...` для более точного попадания.
Таблица `param_budget_table.csv` идёт в Supplementary напрямую.

## Шаг 2 — первый запуск (построит и закеширует датасет)

При первом запуске любой модели датасет строится с нуля: для каждой
молекулы строится radius graph (CPU, pure PyTorch), RDKit Mol из SMILES
(с валидацией порядка атомов — см. лог `RDKit mol valid: X/Y`), node/edge
признаки. Результат кешируется в
`data/train_test_val_df/cache/*.pt` — повторные запуски (другая модель,
другой seed) читают кеш мгновенно.

```bash
# Быстрая проверка на одной модели (построит кеш для train/val/test)
python -m src.train --config configs/gcn.yaml --seed 0
```

Проверьте в логе строку вида:
```
chaos_atomic_train_with_coordinates.parquet: NNNNN molecules, MMMMMMM atoms,
extended=True, RDKit mol valid: XXXXX/NNNNN (YY.Y%)
```

**Если процент `RDKit mol valid` низкий** (скажем, <80%) — это означает,
что порядок атомов после `Chem.MolFromSmiles(...).AddHs()` не совпадает с
`atom_index` в parquet для многих молекул, и extended-признаки для них
будут нулями (безопасный fallback, но менее информативный). Это стоит
упомянуть в Methods/Limitations статьи. Если процент близок к 0% —
вероятно, исходный pipeline генерации 3D-структур переставляет атомы
(например, через xtb/CREST) иначе, чем RDKit; в этом случае стоит either
(a) сохранить mapping atom_index ↔ RDKit-индекс на этапе генерации данных,
либо (b) временно работать с `use_extended: false` в `configs/base.yaml`
(только 7 base-признаков, без RDKit).

## Шаг 3 — Level 1: controlled comparison (все baseline)

```bash
# 3 seed-а на модель, ~12-20 часов на GPU в зависимости от размера train
SEEDS="0 1 2" bash scripts/run_benchmark.sh gcn gat gatv2 gine schnet

# Сводная таблица (mean ± std по seed-ам)
python -m scripts.aggregate_results --loss mse --csv results/comparison_table_level1.csv
```

Используйте для сравнения архитектур: `weighted_mae`, `weighted_r2`,
`polar_mae`, `emd_raw`, `emd_normalized`, `molecular_mae`, `molecular_emd`,
`molecular_cosine`, `molecular_polar_mae`. **Не сравнивайте** `loss`
между MSE-моделями и CombinedLoss-моделями — разные шкалы.

Чекпоинт лучшей эпохи выбирается по `weighted_mae` на валидации —
одинаковый критерий для всех архитектур (см. `src/evaluate.py::CHECKPOINT_METRIC`).

## Шаг 4 — замена final_model.py на вашу архитектуру

`src/models/final_model.py` сейчас — временный GINE-based stub (5 слоёв).
Когда определитесь с финальной архитектурой:

1. Перепишите класс `FinalSigmaModel` в `final_model.py`, сохранив контракт:
   - `__init__(self, ..., out_dim=51, dropout=0.05, use_extended=True)`
   - `forward(self, data) -> Tensor (N_atoms, 51)`, неотрицательный (softplus)
   - использует `data.x`, `data.edge_index`, `data.edge_attr`, `data.z`,
     `data.pos`, `data.batch` — НИЧЕГО не пересчитывает внутри forward
     (всё уже в dataset.py)
2. Обновите `configs/final_model.yaml` под новые гиперпараметры
3. `python -m scripts.count_params --auto-tune` — подберите hidden под бюджет
4. `bash scripts/run_benchmark.sh final` — Level 1 (MSE, та же таблица что у baseline)
5. `bash scripts/run_ablation.sh` — Level 2 (MSE vs CombinedLoss)

## Шаг 5 — Level 2: loss ablation

```bash
SEEDS="0 1 2" bash scripts/run_ablation.sh

python -m scripts.aggregate_results --loss mse      --csv results/final_mse.csv
python -m scripts.aggregate_results --loss combined --csv results/final_combined.csv
```

## Использование GPU "по максимуму"

- `training.use_amp: true` в `base.yaml` — mixed precision (autocast + GradScaler)
- `training.num_workers: 4` — параллельная подгрузка батчей
- `training.grad_accum_steps: 2` с `batch_size: 24` → effective batch 48;
  если на GPU достаточно памяти, можно поднять `batch_size` и/или убрать
  grad_accum — но делайте это **для всех конфигов одинаково**, иначе
  нарушится fair comparison.
- Кеш датасета (`data/train_test_val_df/cache/*.pt`) убирает CPU-bottleneck
  (RDKit/radius_graph) из тренировочного цикла — после первого запуска GPU
  не ждёт CPU.

## Что проверено (smoke test)

Все 6 моделей (gcn/gat/gatv2/gine/schnet/final) прогнаны end-to-end на
синтетических данных (3 молекулы: этанол, бензол, уксусная кислота,
случайные σ-профили) — обучение, чекпоинтинг, evaluate, aggregate_results,
ablation (mse/combined) работают без ошибок. Параметрические бюджеты
посчитаны и находятся в пределах ±2% от 695,667.

**Не проверено** (нужно сделать на реальных данных на сервере):
- Процент `RDKit mol valid` на полном train/val/test (см. Шаг 2)
- Время на эпоху / общее время Level 1 на реальном объёме данных
  (~1.08M атомов в train)
- Память GPU при batch_size=24 + extended features (39-dim edge_attr) для
  GINE/final на больших молекулах
