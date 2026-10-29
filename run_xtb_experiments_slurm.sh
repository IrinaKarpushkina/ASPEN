#!/bin/bash
#SBATCH --job-name=sigma_xtb
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=172:00:00
#SBATCH --output=logs/sigma_xtb_%j.out
#SBATCH --error=logs/sigma_xtb_%j.err

# Использование:  sbatch run_xtb_experiments_slurm.sh
#CONFIGS="dimenet_pp_enhanced_xtb dimenet_pp_delta_xtb sigma_surface_xtb" SEEDS="0 1 2" #sbatch run_xtb_experiments_slurm.sh
mkdir -p logs results/xtb/checkpoints results/xtb/metrics
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

CONFIGS="${CONFIGS:-dimenet_pp_delta_l1 sigma_dn_geom_xtb sigma_surface_geom sigma_surface_xtb_big sigma_dn_geom_xtb_big}"
SEEDS="${SEEDS:-0 1 2}"

if python -c "import pytest" 2>/dev/null; then
  python -m pytest tests/test_surface_field.py -q || exit 1
else
  echo "pytest не установлен - тесты пропущены (pip install pytest)"
fi
echo "Job started: $(date)"; nvidia-smi || true

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    if compgen -G "results/xtb/metrics/${cfg}_seed${seed}_*.json" > /dev/null; then
      echo "SKIP: $cfg seed $seed"; continue
    fi
    echo "=== $cfg | seed $seed | $(date) ==="
    python -m src.train_3d_physics --config configs/3d/${cfg}.yaml --seed "$seed" --output-dir results/xtb
  done
done
echo "Done: $(date)"
