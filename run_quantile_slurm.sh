#!/bin/bash
#SBATCH --job-name=sigma_qf
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=102:00:00
#SBATCH --output=logs/sigma_qf_%j.out
#SBATCH --error=logs/sigma_qf_%j.err

# Абляция представления: sigma_dn_geom с квантильной головой, 3 сида (~8 ч на сид).
# Запуск из корня проекта:  sbatch run_quantile_slurm.sh
# Свои сиды:                SEEDS="0" sbatch run_quantile_slurm.sh
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN
mkdir -p logs results/quantile/checkpoints results/quantile/metrics
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

CFG=sigma_dn_geom_qf
SEEDS="${SEEDS:-0 1 2}"
OUT=results/quantile
echo "Job started: $(date)"; nvidia-smi || true
for seed in $SEEDS; do
  if compgen -G "$OUT/metrics/${CFG}_seed${seed}_*.json" > /dev/null; then
    echo "SKIP: $CFG seed $seed (метрики уже есть)"; continue
  fi
  echo "=== $CFG | seed $seed | $(date) ==="
  python -m src.train_3d_physics --config configs/3d/${CFG}.yaml --seed "$seed" --output-dir $OUT
done
echo "Done: $(date)"
