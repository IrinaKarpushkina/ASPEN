# ASPEN: Atomic Sigma Profile Prediction via 3D Graph Neural Networks

**ASPEN** (Atomic Sigma Profiles Embedding for Neural Networks) – это инструмент для предсказания атомных σ-профилей на основе трехмерной геометрии молекулы. Проект реализует 3D-инвариантные графовые нейронные сети (SchNet, GAT, Graph Transformer и др.) с физически-информированными функциями потерь для точной декомпозиции поверхностного распределения заряда по атомам.

---

## 📂 Структура репозитория

```
ASPEN/
├── data/                                 # Папка для данных 
│   ├── train_test_val_df/                # Разделение по кластерам (70/15/15)
│   └── generated_conformers_df/          # σ-профили со структурами молекул, полученными с использованием разных методов генерации конформеров
│       # (RDKit, OpenBabel, ConForge, AVGFlow)
├── models/                               # Сохранённые веса обученных моделей (.pt – не в репозитории)
├── scripts/                              # Вспомогательные скрипты
│   ├── architectures/                    # Реализации GCN, GIN, GAT, Graph Transformer, SchNet
│   ├── data_processing/                  # Разделение датасета, построение атомных σ-профилей
│   ├── visualisations/                   # UMAP-проекции, тепловые карты корреляций
│   └── evaluate_conformers/              # Оценка влияния геометрии конформеров на качество
│       └── results/                      # JSON-логи, .out файлы экспериментов
├── src/                                  # Финальный код для предсказания атомных σ-профилей (основной инструмент)
├── tests/                                # Валидационные тесты: предсказание ЯМР спектров, свойств
├── .gitignore                            # Игнорирование .parquet, .pt, __pycache__
└── README.md
```

**Примечание о данных**: все `.parquet` файлы (обучающие/тестовые наборы, сгенерированные профили) не хранятся в Git из-за большого размера. Если Вы хотите получить доступ к этим данным, обращайтесь по адресу в разделе "Контакты".

---

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/IrinaKarpushkina/ASPEN.git
cd ASPEN
```

### 2. Установка окружения

Проект использует Python 3.10. Рекомендуется создать отдельное conda-окружение:

```bash
conda create -n aspen python=3.10
conda activate aspen
pip install -r requirements.txt
```

Основные библиотеки: `torch`, `torch-geometric`, `rdkit`, `numpy`, `pandas`, `scikit-learn`, `umap-learn`, `hdbscan`, `xgboost`.

### 3. Получение данных

Исходные данные доступны на **[Zenodo](https://zenodo.org/record/xxxxx)** (ссылка будет добавлена после публикации).  
Загрузите архив и распакуйте его в папку `data/`:

```bash
wget https://zenodo.org/.../ASPEN_data.zip
unzip ASPEN_data.zip -d data/
```

После этого структура `data/train_test_val_df/` и `data/generated_conformers_df/` будет заполнена.

### 4. Запуск предсказания 

```python
from src.predict import predict_atomic_sigma_profiles

# Вход: молекула в формате XYZ или SMILES + 3D координаты
profiles = predict_atomic_sigma_profiles(
    "example.xyz",
    model_path="models/schnet_best.pt"
)
# profiles – словарь {atom_index: numpy.array длины 51}
```

Более подробные примеры см. в `examples/` (папка будет добавлена).

---

## Воспроизведение экспериментов

### Обучение модели SchNet с физическими потерями

```bash
python scripts/architectures/train_schnet.py \
  --data_dir data/generated_conformers_df/rdkit/ \
  --split_file data/train_test_val_df/split_clusters.csv \
  --epochs 200 \
  --loss_weight_phys 0.05
```

### Оценка влияния метода генерации конформеров

```bash
python scripts/evaluate_conformers/evaluate_conformers_orig_rdkit_conforge_avgflow_openbabel_vkr_model.py
```

### Предсказание квантово-химических свойств на основе σ-профилей

```bash
python tests/predict_properties.py \
  --profile_dir data/generated_conformers_df/rdkit/ \
  --targets homo lumo hlg dipole \
  --model xgboost
```

---

## Основные результаты

| Модель | R² (атомный σ-профиль) | Косинусное сходство | EMD |
|--------|------------------------|---------------------|-----|
| GCN (2D, поатомная)     | 0.573   | 0.841 | 0.00150 |
| GIN (2D + реберные признаки) | 0.754 | 0.902 | 0.00101 |
| Graph Transformer (2D + расстояния) | 0.862 | 0.941 | 0.00072 |
| **SchNet (3D, физические потери)** | **0.894** | **0.924** | **0.00066** |

**Применение для прогноза свойств** (XGBoost, R²):

| Набор признаков | HOMO | LUMO | HLG | Диполь |
|----------------|------|------|-----|--------|
| Базовые дескрипторы | 0.702 | 0.755 | 0.814 | 0.454 |
| Истинные σ-профили (DFT) | 0.769 | 0.797 | 0.786 | 0.631 |
| **Предсказанные σ-профили (SchNet)** | **0.839** | **0.858** | **0.855** | **0.699** |
| Предсказанные + базовые | **0.877** | **0.896** | **0.903** | **0.715** |

> Предсказанные профили дают **более высокую точность**, чем исходные DFT-профили, что объясняется регуляризующим эффектом нейросетевого представления.

---

## Зависимости

- Python 3.10
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- RDKit 2023.09
- UMAP, HDBSCAN
- XGBoost 2.0
- NumPy, Pandas, SciPy, Matplotlib

Полный список – в `requirements.txt`.

---

## Цитирование

Если вы используете этот код или данные в своей работе, пожалуйста, цитируйте:

```
Авторы, Название статьи.
[ Цитата ]. 2026.
```

BibTeX:

```bibtex
@misc{aspen2026,
  author = {},
  title = {},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/IrinaKarpushkina/ASPEN}
}
```

---

## Лицензия

Проект распространяется под лицензией [] (см. файл `LICENSE`). Данные CHAOS и COSMO-RS результаты принадлежат их авторам; используйте их в соответствии с их лицензиями.

---

## Контакты

Ирина Карпушкина – `ikarpushkina@niuitmo.ru`  
Лаборатория компьютерного дизайна материалов, Университет ИТМО, Санкт-Петербург, Россия.

По вопросам, предложениям и сообщениям об ошибках открывайте **Issue** на GitHub.
```
