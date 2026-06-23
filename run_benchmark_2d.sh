#!/bin/bash
#SBATCH --job-name=sigma_benchmark_2d
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=logs/benchmark_2d_%j.out
#SBATCH --error=logs/benchmark_2d_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# 2D бенчмарк: граф из химических связей SMILES/RDKit, никаких координат.
# Node features: 14-dim | Edge features: 7-dim (topology only, no RBF)
#
# SchNet исключён намеренно — он принципиально 3D-архитектура.
# Все остальные архитектуры идентичны 3D-бенчмарку (гиперпараметры,
# бюджет параметров, seed-ы) — только source of geometry убран.
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/2d/checkpoints results/2d/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN_benchmark/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# SchNet исключён (нет configs/2d/schnet.yaml — это намеренно)
#MODELS="${MODELS:-gcn gat gatv2 gine attentive_fp dmpnn gps}"
MODELS="${MODELS:-dmpnn}"
SEEDS="${SEEDS:-0 1 2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo ""

for model in $MODELS; do
    config="configs/2d/${model}.yaml"

    if [ ! -f "$config" ]; then
        echo "WARNING: $config not found, skipping $model"
        continue
    fi

    for seed in $SEEDS; do
        result_file="results/2d/metrics/${model}_seed${seed}_mse.json"
        if [ -f "$result_file" ]; then
            echo "SKIP: $result_file already exists"
            continue
        fi

        echo "========================================="
        echo "  $model [2D] | seed $seed | $(date)"
        echo "========================================="

        python -m src.train \
            --config   "$config"          \
            --seed     "$seed"            \
            --loss     mse                \
            --output-dir results/2d

        echo "  Done: $model seed $seed"
        echo ""
    done
done

echo "All runs finished: $(date)"
echo "Aggregating results..."

python -m scripts.aggregate_results \
    --loss mse \
    --sort emd_raw \
    --metrics-dir results/2d/metrics \
    --csv results/2d/comparison_table_level1_2d.csv

echo "Done. Results: results/2d/comparison_table_level1_2d.csv"
