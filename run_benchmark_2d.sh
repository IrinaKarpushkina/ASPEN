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
# 2D-only benchmark: graph built from RDKit chemical bonds, no coordinates.
# Node features: 14-dim | Edge features: 7-dim (topology only, no RBF).
#
# EDIT before running:
#   - conda env name / activation path below
#   - cd path below
#   - configs/base.yaml data.{train,val,test}_path / cache_dir
#   - #SBATCH --partition / --nodelist for your cluster
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/checkpoints results/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

MODELS="${MODELS:-gps}"
#MODELS="${MODELS:-gcn gat gatv2 gine dmpnn attentive_fp gps}"
SEEDS="${SEEDS:-0 1 2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo ""

# Sanity check before spending GPU time (fast, CPU-only):
python -m pytest tests/ -q || { echo "Tests failed, aborting."; exit 1; }
python -m scripts.count_params

for model in $MODELS; do
    config="configs/2d/${model}.yaml"

    if [ ! -f "$config" ]; then
        echo "WARNING: $config not found, skipping $model"
        continue
    fi

    for seed in $SEEDS; do
        result_file="results/metrics/${model}_seed${seed}_mse.json"
        if [ -f "$result_file" ]; then
            echo "SKIP: $result_file already exists"
            continue
        fi

        echo "========================================="
        echo "  $model [2D] | seed $seed | $(date)"
        echo "========================================="

        python -m src.train \
            --config "$config" \
            --seed "$seed" \
            --output-dir results

        echo "  Done: $model seed $seed"
        echo ""
    done
done

echo "All runs finished: $(date)"
echo "Aggregating results..."

python -m scripts.aggregate_results \
    --metrics-dir results/metrics \
    --loss mse \
    --sort emd_raw \
    --csv results/comparison_table_2d.csv

echo "Done. Results: results/comparison_table_2d.csv"
