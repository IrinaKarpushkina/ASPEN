# xTB-признаки + surface-field: что добавлено и как запустить

Все файлы **новые**, существующие не перезаписываются. Единственная правка существующего файла —
`src/train_3d_physics.py` (6 вставок, делает `scripts/patch_xtb.py`, с бэкапом).

## 0. Что внутри

| файл | назначение |
|---|---|
| `scripts/xtb_features.py` | считает GFN2-xTB single-point **по вашим координатам** → `<папка parquet>/xtb/*.xtb.npz` |
| `src/data/features_xtb.py`, `src/data/dataset_3d_xtb.py` | присоединяют 7 xTB-признаков к `d.x` (+ `d.xtb_q`); features `xtb` и `xtb_sasa` |
| `src/physics_extension/surface_field.py` | модель `SigmaSurfaceField` (поверхностное поле + дифференцируемая гистограмма) |
| `scripts/patch_xtb.py` | вставки в `train_3d_physics.py` |
| `configs/3d/*xtb*.yaml`, `sigma_surface_geom.yaml`, `dimenet_pp_delta_l1.yaml` | 8 конфигов |
| `tests/test_surface_field.py` | 5 проверок (вращение, SASA, потенциал, батчинг, чанкование) |
| `run_xtb_features_slurm.sh`, `run_xtb_experiments_slurm.sh` | запуск на кластере |

7 xTB-признаков атома: заряд Малликена, сумма и максимум порядков связей, полный заряд молекулы,
|дипольный момент|, щель HOMO–LUMO, флаг «расчёт сошёлся».

## 1. Установка xTB (одна команда, без компиляции)

```bash
conda activate sigma
pip install tblite            # GFN2-xTB как python-пакет
# если pip не находит колесо под вашу версию python:
# conda install -c conda-forge tblite tblite-python
python -m scripts.xtb_features --selftest     # должно напечатать 4 строки с ok=1
```

## 2. Посчитать xTB для всех трёх файлов

```bash
D=/mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df
python -m scripts.xtb_features --n-jobs 16 --parquet \
  $D/chaos_atomic_train_with_coordinates.parquet \
  $D/chaos_atomic_val_with_coordinates.parquet \
  $D/chaos_atomic_test_with_coordinates.parquet
# или:  sbatch run_xtb_features_slurm.sh
```

Скрипт печатает минимальное межатомное расстояние (должно быть ≈0.9–1.1 Å; если нет — координаты не в ангстремах)
и число молекул, где SCF не сошёлся (для них признаки = 0, флаг ok=0). Файл `.npz` пропускается,
если уже есть (`--force` — пересчитать). Папку можно сменить: `--out-dir` или `export ASPEN_XTB_DIR=...`
(тогда задайте её и при обучении).

**Время.** У меня на 1 ядре: 0.003 с (9 атомов), 0.05 с (18–33 атома), 0.12 с (67 атомов). Для ~50 тыс. молекул по ~28 атомов
это ≈ 40–60 мин на одном ядре, около 3–5 мин на 16 ядрах (оценка, на ваших данных не мерил).

## 3. Проверка до обучения

```bash
python -m pytest tests/test_surface_field.py -q       # 5 passed
```

## 4. Патч trainer'а

```bash
python scripts/patch_xtb.py --check    # покажет, применимы ли все 6 замен
python scripts/patch_xtb.py            # применить (бэкап src/train_3d_physics.py.bak_pre_xtb)
```

Если ваш файл отличается и скрипт сообщит о неприменимой замене — вставьте вручную (см. список `EDITS` в `scripts/patch_xtb.py`):
импорты `XtbChaosParquet3DDataset, XtbSasaChaosParquet3DDataset, SigmaSurfaceField`; имена
`dimenet_pp_enhanced_xtb/delta_xtb/delta_l1` в `_MODELS_NEEDING_TRIPLETS` и в `registry`;
ветка `if name.startswith("sigma_surface"): return SigmaSurfaceField(**mc), name` перед `sigma_kitchen_sink`;
`"xtb"`, `"xtb_sasa"` в допустимых features и в словаре `ds_cls`; `or name.startswith("sigma_surface")` в `compute_triplets`.

## 5. Эксперименты

Сначала smoke (одна эпоха, покажет время эпохи и проверит данные):
```bash
python -m src.train_3d_physics --config configs/3d/sigma_surface_xtb.yaml --seed 0 --smoke --output-dir results/xtb
```

| # | конфиг | что изолирует | сравнивать с |
|---|---|---|---|
| 1 | `dimenet_pp_enhanced_xtb` (MSE) | эффект xTB-признаков | `dimenet_pp_enhanced_sched` (R² 0.9170) |
| 2 | `dimenet_pp_delta_l1` | эффект L1-loss (без xTB) | `dimenet_pp_delta_fixed` (0.9152) |
| 3 | `dimenet_pp_delta_xtb` (L1) | эффект xTB при L1 | № 2 |
| 4 | `sigma_dn_geom_xtb` | xTB с SASA-головой | `sigma_dn_geom` (0.9159) |
| 5 | `sigma_surface_geom` | surface-голова без xTB | `sigma_dn_geom` |
| 6 | `sigma_surface_xtb` (~0.72M) | surface + xTB | № 4 |
| 7 | `sigma_surface_xtb_big` (~1.31M) | то же при большем размере | № 8 |
| 8 | `sigma_dn_geom_xtb_big` (~1.30M) | парный контроль для № 7 | — |

**Порядок запуска (с точками остановки):**
1. Smoke каждого конфига (`--smoke`): время эпохи и отсутствие ошибок. Время эпохи surface-модели я не мерил.
2. Ворота: только № 1, seed 0. Если R² не выше 0.9170 хотя бы на 0.002 — остальное не запускать, сначала разобраться (см. «Как читать»).
3. Если ворота пройдены: № 2 и 3 (delta-пара), затем № 4.
4. Surface-семейство № 5–6, и только если 5 или 6 не хуже 4 — большие № 7–8.
5. Для финала — 3 сида (0,1,2) у победителей, как в вашей таблице.

Минимальный набор, если время ограничено: **1, 3, 6** (+ 2 как контроль к 3). Запуск: `sbatch run_xtb_experiments_slurm.sh`
(список конфигов и сидов задаётся переменными `CONFIGS` и `SEEDS`). Результаты — `results/xtb/metrics/`;
агрегация: `python -m scripts.aggregate_results --metrics-dir results/xtb/metrics --csv results/comparison_table_xtb.csv`.

В логе у `*_big` будет «diff=87%» относительно target_params — это просто сравнение с бюджетом 700k из base.yaml, ожидаемо.

**Как читать.** Std по сидам в ваших таблицах 1e-4…3e-4 по R², поэтому Δ ≥ 0.001 уже статистически заметна, а практически
интересна Δ ≳ 0.005. Если № 1 даёт < 0.001 — заряды/порядки связей xTB информации не добавляют, и потолок лежит не здесь
(значит, шум таргета или геометрия таргета ≠ геометрия модели).

## 6. Ограничения (важно знать)

* Surface-модель не проверена на ваших данных; на синтетике проверены только формы, конечность, градиенты и геометрические тесты.
  Время эпохи на GPU я не мерил: плотные тензоры (B, Nmax, 48, Nmax) тяжелее DimeNet++, при необходимости уменьшите `n_points`
  (32) — защита от OOM для очень больших молекул встроена (`max_elems`).
* tblite не отдаёт атомные диполи, поэтому анизотропия (σ-hole у Cl/Br/I) описывается только геометрическими признаками
  (`c_back` — косинус между направлением на точку и продолжением связи), а не мультиполями xTB.
* Поверхность — сферы 1.17·vdW (как ваш `features_3d_sasa.py`), а не реальная полость COSMO; насколько это близко, показывает
  `scripts/ceiling_analysis.py` (часть A, R² площадь–SASA). Запустите его до больших прогонов.
* `inference_time_s_per_molecule` в json-результатах НЕ включает xTB (считается отдельно, до модели): не сравнивайте его с «полным» временем без добавки ~0.05 с/молекулу.
* Скорость инференса: xTB single-point (десятки миллисекунд на молекулу на CPU) станет доминирующей частью времени.
