#!/bin/bash
#SBATCH --job-name=unimol_pretrained_bench
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/unimol_pretrained_%j.out
#SBATCH --error=logs/unimol_pretrained_%j.err
# ─────────────────────────────────────────────────────────────────────────────
# Track B: FROZEN pretrained Uni-Mol backbone (unimol_tools) + small trainable
# head (122,035 trainable params -- the 47M-param backbone is frozen and not
# counted). NOT part of the 700k-parameter-budget comparison (see
# configs/unimol_pretrained/unimol_pretrained.yaml and
# src/models/models_3d/unimol_pretrained.py docstrings) -- deliberately a
# SEPARATE script/entrypoint from run_benchmark_3d.sh / src.train_3d.
#
# Pipeline confirmed working end-to-end interactively before this script was
# written: train/val/test all build + cache correctly, atom-vocabulary-limited
# molecules are skipped and logged (val 26/7941, test 11/7800, train TBD --
# check this run's logs), and training converges (R2=0.62 by epoch 5, seed 0,
# interactive smoke run). See PROVENANCE.md for the atom-vocabulary limitation
# writeup.
#
# EDIT before running (only if your paths/env differ from what was just
# confirmed working interactively):
#   - conda env / cd path below
#   - configs/unimol_pretrained/unimol_pretrained.yaml: data.{train,val,test}_path,
#     data.cache_dir
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

mkdir -p logs results/checkpoints results/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

CONFIG="configs/unimol_pretrained/unimol_pretrained.yaml"
SEEDS="${SEEDS:-0 1 2}"

echo "Config: $CONFIG"
echo "Seeds:  $SEEDS"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG not found."
    exit 1
fi

for seed in $SEEDS; do
    result_file="results/metrics/unimol_pretrained_seed${seed}_mse.json"
    if [ -f "$result_file" ]; then
        echo "SKIP: $result_file already exists"
        continue
    fi

    echo "========================================="
    echo "  unimol_pretrained | seed $seed | $(date)"
    echo "========================================="

    # Seed 0 reuses the train/val/test cache already built during the
    # interactive smoke run (cache_dir in the config) -- subsequent seeds
    # reuse it too, so only seed 0 pays the one-off UniMolRepr forward-pass
    # cost if the cache from the interactive run is still on disk.
    python -m src.train_unimol_pretrained \
        --config "$CONFIG" \
        --seed "$seed" \
        --output-dir results

    echo "  Done: unimol_pretrained seed $seed"
    echo ""
done

echo "All runs finished: $(date)"
echo ""
echo "NOTE: these results write to results/metrics/unimol_pretrained_seed*_mse.json"
echo "with \"mode\": \"unimol_pretrained\" -- scripts/aggregate_results.py's mixed-mode"
echo "guard (PROVENANCE.md Bug #1) will refuse to mix these into the main"
echo "\"3d_pure\" comparison table. Aggregate them separately:"
echo ""
echo "  mkdir -p results/metrics_unimol_pretrained_only"
echo "  cp results/metrics/unimol_pretrained_seed*_mse.json results/metrics_unimol_pretrained_only/"
echo "  python -m scripts.aggregate_results \\"
echo "      --metrics-dir results/metrics_unimol_pretrained_only \\"
echo "      --loss mse --sort emd_raw \\"
echo "      --csv results/comparison_table_unimol_pretrained.csv"
echo ""
echo "REMINDER: record the exact number of atom-vocabulary-limited molecules"
echo "skipped per split (val/test already confirmed: 26/7941, 11/7800; check"
echo "this run's log for the train split count) in PROVENANCE.md before"
echo "reporting these numbers anywhere."
