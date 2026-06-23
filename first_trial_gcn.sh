#!/bin/bash
#SBATCH --job-name=baseline_gcn
#SBATCH --partition=aichem        # Очередь для химических расчетов
#SBATCH --nodelist=aihub
#SBATCH --nodes=1                 # Используем один узел
#SBATCH --ntasks-per-node=8       # Количество ядер (процессов)
#SBATCH --mem=32G                 # Память (4Гб на ядро — оптимально для ORCA)
#SBATCH --time=48:00:00           # Максимальное время (до 2 суток)
#SBATCH --output=logs/baseline_gcn_%j.out  # Файл с логами (папка logs должна существов>
#SBATCH --error=logs/baseline_gcn_%j.err   # Файл с ошибками

# 0. Создаем папку для логов, если её нет
mkdir -p logs

# 1. Активация окружения Conda
# Используем полный путь к conda.sh для надежности
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh

conda activate sigma

python -m src.train --config configs/gcn.yaml --seed 0

