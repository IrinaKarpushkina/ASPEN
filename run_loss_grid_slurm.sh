#!/bin/bash
#SBATCH --job-name=sigma_lossgrid
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=300:00:00
#SBATCH --output=logs/lossgrid_%j.out
#SBATCH --error=logs/lossgrid_%j.err

# Сетка «архитектура x loss»: 3 архитектуры (geom/global/full) x {mse, strong} x 3 сида = 18 прогонов.
# Слабый physics-loss (3 архитектуры x 3 сида) уже посчитан в results/all_pred_ablation_experiments_results.
# Запуск одной архитектуры (рекомендуется, 6 прогонов ~ 1/3 всего времени):
#   CONFIGS="sigma_dn_geom_mse sigma_dn_geom_strong" sbatch run_loss_grid_slurm.sh
# Все сразу: sbatch run_loss_grid_slurm.sh
mkdir -p logs results/loss_grid/checkpoints results/loss_grid/metrics
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

CONFIGS="${CONFIGS:-sigma_dn_geom_mse sigma_dn_geom_strong sigma_dn_global_mse sigma_dn_global_strong sigma_dn_full_mse sigma_dn_full_strong}"
SEEDS="${SEEDS:-0 1 2}"
OUT=results/loss_grid
echo "Job started: $(date)"; nvidia-smi || true
for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    if compgen -G "$OUT/metrics/${cfg}_seed${seed}_*.json" > /dev/null; then
      echo "SKIP: $cfg seed $seed"; continue
    fi
    echo "=== $cfg | seed $seed | $(date) ==="
    python -m src.train_3d_physics --config configs/3d/${cfg}.yaml --seed "$seed" --output-dir $OUT
  done
done
echo "Done: $(date)"
