#!/bin/bash
#SBATCH --job-name=sigma_delta
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=172:00:00
#SBATCH --output=logs/sigma_delta_%j.out
#SBATCH --error=logs/sigma_delta_%j.err

# dimenet_pp_delta_fixed отдельно от run_sigma_v2_slurm.sh: раньше падала на NaN на
# эпохе 1 (шаг 350), поэтому не должна блокировать очередь остальных моделей, если
# упадёт снова. Перед постановкой в SLURM стоит прогнать smoke вручную:
#   python -m src.train_3d_physics --config configs/3d/dimenet_pp_delta_fixed.yaml --seed 0 --smoke
# и посмотреть в логе "Initial physics loss components": profile должен быть порядка
# единиц, а не тысяч (как было 6958 в первом падении).

mkdir -p logs results/physics_v2/checkpoints results/physics_v2/metrics
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
nvidia-smi || true

SEEDS="${SEEDS:-0 1 2}"
for seed in $SEEDS; do
    if compgen -G "results/physics_v2/metrics/dimenet_pp_delta_seed${seed}_*.json" > /dev/null; then
        echo "SKIP: seed $seed already exists"
        continue
    fi
    echo "=== dimenet_pp_delta_fixed | seed $seed | $(date) ==="
    python -m src.train_3d_physics \
        --config configs/3d/dimenet_pp_delta_fixed.yaml \
        --seed "$seed" \
        --output-dir results/physics_v2
done

echo "Done: $(date)"
