#!/bin/bash
#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=aichem        # Очередь для химических расчетов
#SBATCH --nodelist=aihub
#SBATCH --nodes=1                 # Используем один узел
#SBATCH --ntasks-per-node=8       # Количество ядер (процессов)
#SBATCH --mem=32G                 # Память (4Гб на ядро — оптимально для ORCA)
#SBATCH --time=48:00:00           # Максимальное время (до 2 суток)
#SBATCH --output=logs/baseline_%j.out  # Файл с логами (папка logs должна существов>
#SBATCH --error=logs/baseline_%j.err   # Файл с ошибками

# 0. Создаем папку для логов, если её нет
mkdir -p logs

# 1. Активация окружения Conda
# Используем полный путь к conda.sh для надежности
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh

conda activate sigma

# run_benchmark.sh — запускает Level-1 (controlled comparison, MSE loss)
# для всех baseline-архитектур, несколько seed-ов на модель.
#
# Использование:
#   bash scripts/run_benchmark.sh                       # все модели, seeds 0 1 2
#   bash scripts/run_benchmark.sh gcn gat gatv2         # только указанные
#   SEEDS="0 1 2 3 4" bash scripts/run_benchmark.sh     # другой набор seed-ов
#
# Каждый запуск пишет:
#   results/checkpoints/<model>_seed<seed>_mse.pt
#   results/metrics/<model>_seed<seed>_mse.json

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${*:-gcn gat gatv2 gine schnet}"
SEEDS="${SEEDS:-0 1 2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"

for model in $MODELS; do
  config="configs/${model}.yaml"
  if [ ! -f "$config" ]; then
    echo "WARNING: $config not found, skipping $model"
    continue
  fi
  for seed in $SEEDS; do
    echo ""
    echo "=== $model | seed $seed | loss mse ==="
    python -m src.train --config "$config" --seed "$seed" --loss mse
  done
done

echo ""
echo "Done. Aggregate results with:"
echo "  python -m scripts.aggregate_results --csv results/comparison_table.csv"
