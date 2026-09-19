#!/bin/bash
#SBATCH --job-name=sigma_physics_3d
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=172:00:00
#SBATCH --output=logs/physics_%j.out
#SBATCH --error=logs/physics_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Physics-informed 3D experiments for atomic COSMO-RS sigma-profile prediction.
#
# IMPORTANT:
#   This script uses src/train_3d_physics.py and does NOT run/modify the
#   original benchmark training script src/train_3d.py.
#
# The original benchmark models/results are left untouched.
#
# EDIT before running:
#   - conda environment / activation path below
#   - repository path below
#   - Slurm partition / nodelist if needed
#
# Usage:
#   sbatch run_physics_ablation.sh
#
# Optional environment variables:
#   MODELS="dimenet_pp_enhanced" sbatch run_physics_ablation.sh
#   SEEDS="0 1 2" sbatch run_physics_ablation.sh
#
# Smoke test:
#   SMOKE=1 SEEDS="0" MODELS="dimenet_pp_enhanced" sbatch run_physics_ablation.sh
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/physics/checkpoints results/physics/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true
echo ""

# A-F physics ablation.
# Default order:
# A = enhanced features + MSE
# B = base features + physics loss
# C = enhanced features + physics loss
# D = enhanced + molecular FiLM/global context + physics loss
# E = PaiNN + enhanced features + physics loss
# F = enhanced + element prior in shape/logit space + physics loss
MODELS="${MODELS:-dimenet_pp_enhanced dimenet_pp_physics dimenet_pp_enhanced_physics dimenet_pp_global painn_physics dimenet_pp_delta}"
SEEDS="${SEEDS:-0 1 2}"
#SMOKE="${SMOKE:-0}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo "Smoke:  $SMOKE"
echo ""

for model in $MODELS; do
    config="configs/3d/${model}.yaml"

    if [ ! -f "$config" ]; then
        echo "WARNING: $config not found, skipping $model"
        continue
    fi

    # The training script constructs:
    #   ${model}_seed${seed}_mse.json   for A
    #   ${model}_seed${seed}_physics.json for B-F
    if [ "$model" = "dimenet_pp_enhanced" ]; then
        loss_name="mse"
    else
        loss_name="physics"
    fi

    for seed in $SEEDS; do
        result_file="results/physics/metrics/${model}_seed${seed}_${loss_name}.json"

        if [ -f "$result_file" ]; then
            echo "SKIP: $result_file already exists"
            continue
        fi

        echo "========================================="
        echo "  $model | seed $seed | $(date)"
        echo "========================================="

        if [ "$SMOKE" = "1" ]; then
            python -m src.train_3d_physics \
                --config "$config" \
                --seed "$seed" \
                --output-dir results/physics \
                --smoke
        else
            python -m src.train_3d_physics \
                --config "$config" \
                --seed "$seed" \
                --output-dir results/physics
        fi

        echo "  Done: $model seed $seed"
        echo ""
    done
done

echo "All physics runs finished: $(date)"
echo "Results directory: results/physics/metrics"
echo "Checkpoints directory: results/physics/checkpoints"
