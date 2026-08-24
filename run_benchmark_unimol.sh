#!/bin/bash
#SBATCH --job-name=unimol_training_bench
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=logs/unimol_train_benchmark_3d_%j.out
#SBATCH --error=logs/unimol_train_benchmark_3d_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# 3D-only benchmark: geometric graph built from 3D coordinates (radius graph
# for SchNet/PaiNN/DimeNet(++)/SphereNet/EGNN/TorchMD-Net/MACE, dense all-pairs
# attention with a distance-bias for Uni-Mol). See PROVENANCE.md.
#
# EDIT before running:
#   - conda env name / activation path below
#   - cd path below
#   - configs/3d/base.yaml data.{train,val,test}_path / cache_dir (SEPARATE
#     from the 2D cache_dir — see configs/3d/base.yaml)
#   - #SBATCH --partition / --nodelist for your cluster
#
# Extra dependency: pip install -r requirements-3d.txt (adds e3nn, needed
# only for MACE; every other model works without it).
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/checkpoints results/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

#MODELS="${MODELS:-schnet}"
#MODELS="${MODELS:-schnet painn dimenet dimenet_pp spherenet egnn torchmdnet mace unimol}"
MODELS="unimol"
SEEDS="${SEEDS:-0 1 2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo ""

# Sanity check before spending GPU time (fast, CPU-only):
#python -m pytest tests/ -k "_3d" -q || { echo "3D tests failed, aborting."; exit 1; }
#python -m scripts.count_params --configs-dir configs/3d

for model in $MODELS; do
    config="configs/3d/${model}.yaml"

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
        echo "  $model [3D] | seed $seed | $(date)"
        echo "========================================="

        python -m src.train_3d \
            --config "$config" \
            --seed "$seed" \
            --output-dir results

        echo "  Done: $model seed $seed"
        echo ""
    done
done

echo "All runs finished: $(date)"
echo "Aggregating results..."

# NOTE: if results/metrics/ ALSO contains 2D runs (train.py writes to the
# same directory by default), aggregate_results.py will refuse to mix
# "mode": "2d_pure" and "mode": "3d_pure" files (this is intentional — see
# PROVENANCE.md, Bug #1). Point --metrics-dir at a 3D-only subset, or run
# the 2D and 3D benchmarks with different --output-dir values, if you want
# them aggregated separately by default.

cp results/metrics/unimol_seed* results/metrics/metrics_3d/

python -m scripts.aggregate_results \
    --metrics-dir results/metrics/metrics_3d/ \
    --loss mse \
    --sort emd_raw \
    --csv results/comparison_table_3d.csv

echo "Done. Results: results/comparison_table_3d.csv"
