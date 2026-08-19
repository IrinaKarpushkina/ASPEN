#!/bin/bash
#SBATCH --job-name=sigma_test_2d
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=logs/benchmark_2d_%j.out
#SBATCH --error=logs/benchmark_2d_%j.err


source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

for m in gps dmpnn attentive_fp; do
    python -m scripts.molecule_level_eval \
        --config configs/2d/${m}.yaml \
        --checkpoint results/checkpoints/${m}_seed0_mse.pt \
        --mode 2d --split test --out results/per_mol/${m}_seed0.npz
done

python -m scripts.pairwise_significance_molecule --a results/per_mol/gps_seed0.npz --b results/per_mol/dmpnn_seed0.npz --metric mae
python -m scripts.pairwise_significance_molecule --a results/per_mol/gps_seed0.npz --b results/per_mol/attentive_fp_seed0.npz --metric mae
python -m scripts.pairwise_significance_molecule --a results/per_mol/dmpnn_seed0.npz --b results/per_mol/attentive_fp_seed0.npz --metric mae



