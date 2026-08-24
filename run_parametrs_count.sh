#!/bin/bash
#SBATCH --job-name=sigma_benchmark_3d
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_3d_%j.out
#SBATCH --error=logs/benchmark_3d_%j.err

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

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

python -m scripts.count_params --configs-dir configs/3d --auto-tune

python -m pytest tests/test_param_budget_3d.py tests/test_forward_shapes_3d.py -v

python -m pytest tests/ -q -k "unimol and not pretrained"


python -m pytest tests/test_unimol_pretrained_smoke.py -q


python -m scripts.check_unimol_tools_api
