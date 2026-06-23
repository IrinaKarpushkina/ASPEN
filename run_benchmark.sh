#!/bin/bash
#SBATCH --job-name=sigma_benchmark
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00           # 7 суток — 24 запуска (8 моделей × 3 seed) по ~8-20ч
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

mkdir -p logs results/checkpoints results/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN_benchmark/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# ── Список моделей и seed-ов ───────────────────────────────────────────────
# Можно переопределить при запуске:
#   sbatch --export=MODELS="gcn gat",SEEDS="0 1" run_benchmark.sh
MODELS="${MODELS:-gcn gat gatv2 gine schnet attentive_fp dmpnn gps}"
SEEDS="${SEEDS:-0 1 2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo ""

# ── Основной цикл ─────────────────────────────────────────────────────────
for model in $MODELS; do
    config="configs/${model}.yaml"

    if [ ! -f "$config" ]; then
        echo "WARNING: $config not found, skipping $model"
        continue
    fi

    for seed in $SEEDS; do
        # Пропускаем, если результат уже есть (удобно при перезапуске после сбоя)
        result_file="results/metrics/${model}_seed${seed}_mse.json"
        if [ -f "$result_file" ]; then
            echo "SKIP: $result_file already exists"
            continue
        fi

        echo "========================================="
        echo "  $model | seed $seed | $(date)"
        echo "========================================="

        python -m src.train \
            --config "$config" \
            --seed   "$seed"   \
            --loss   mse       \
            --output-dir results

        echo "  Done: $model seed $seed"
        echo ""
    done
done

# ── Агрегация результатов ─────────────────────────────────────────────────
echo "All runs finished: $(date)"
echo "Aggregating results..."

python -m scripts.aggregate_results \
    --loss mse \
    --sort emd_raw \
    --csv results/comparison_table_level1.csv

echo "Done. Results: results/comparison_table_level1.csv"
