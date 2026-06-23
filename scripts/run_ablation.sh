#!/bin/bash
# run_ablation.sh — Level-2: финальная модель с MSE vs CombinedLoss.
#
# Изолирует вклад multi-component loss от вклада архитектуры (Level-1
# уже показал архитектурный эффект при одинаковом MSE для всех моделей).
#
# Использование:
#   bash scripts/run_ablation.sh
#   SEEDS="0 1 2" bash scripts/run_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"

for seed in $SEEDS; do
  echo ""
  echo "=== final | seed $seed | loss mse (baseline objective) ==="
  python -m src.train --config configs/final_model.yaml --seed "$seed" --loss mse

  echo ""
  echo "=== final | seed $seed | loss combined (ablation) ==="
  python -m src.train --config configs/final_model.yaml --seed "$seed" --loss combined
done

echo ""
echo "Done. Compare with:"
echo "  python -m scripts.aggregate_results --loss mse      --csv results/final_mse.csv"
echo "  python -m scripts.aggregate_results --loss combined --csv results/final_combined.csv"
