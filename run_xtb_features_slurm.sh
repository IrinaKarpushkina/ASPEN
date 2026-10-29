#!/bin/bash
#SBATCH --job-name=xtb_feats
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/xtb_feats_%j.out
#SBATCH --error=logs/xtb_feats_%j.err

mkdir -p logs
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

D=/mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df
python -m scripts.xtb_features --selftest || exit 1
python -m scripts.xtb_features --n-jobs ${SLURM_CPUS_PER_TASK:-16} --parquet \
  $D/chaos_atomic_train_with_coordinates.parquet \
  $D/chaos_atomic_val_with_coordinates.parquet \
  $D/chaos_atomic_test_with_coordinates.parquet
