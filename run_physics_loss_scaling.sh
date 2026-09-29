#!/bin/bash
#SBATCH --job-name=sigma_v2
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=172:00:00
#SBATCH --output=logs/sigma_loss_scaling_%j.out
#SBATCH --error=logs/sigma_loss_scaling_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/physics_loss_scaling/checkpoints results/physics_loss_scaling/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

#cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN
cd /mnt/tank/scratch/mkokorina/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

# По умолчанию — без sigma_dn_neutral (см. пояснение выше) и без delta (запускается
# отдельно, ниже, т.к. может упасть и не должна блокировать остальные модели в очереди).
MODELS="${MODELS:-dimenet_pp_enhanced_0.5x dimenet_pp_enhanced_2x dimenet_pp_enhanced_4x dimenet_pp_delta_0.5x dimenet_pp_delta_2x dimenet_pp_delta_4x dimenet_pp_0.5x dimenet_pp_2x dimenet_pp_4x}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/physics_loss_scaling}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo "Output: $OUTPUT_DIR"
echo ""

for model in $MODELS; do
    config="configs/3d/scaling/${model}.yaml"

    if [ ! -f "$config" ]; then
        echo "WARNING: $config not found, skipping $model"
        continue
    fi

    for seed in $SEEDS; do
        # имя модели внутри json совпадает с model.name из yaml, суффикс loss зависит
        # от конфига (mse / physics / physics_v2) - проверяем по маске, не по точному имени
        if compgen -G "${OUTPUT_DIR}/metrics/${model}_seed${seed}_*.json" > /dev/null; then
            echo "SKIP: ${OUTPUT_DIR}/metrics/${model}_seed${seed}_*.json already exists"
            continue
        fi

        echo "========================================="
        echo "  $model | seed $seed | $(date)"
        echo "========================================="

        python -m src.train_3d_physics \
            --config "$config" \
            --seed "$seed" \
            --output-dir "$OUTPUT_DIR"

        echo "  Done: $model seed $seed"
        echo ""
    done
done

echo "All runs finished: $(date)"
echo "Aggregating results..."

# Конфиги в этом списке используют РАЗНЫЕ loss (dimenet_pp_enhanced_sched -> mse,
# остальные -> physics). aggregate_results фильтрует по --loss, поэтому агрегируйте
# отдельно на каждый loss, иначе получите пустую/непонятную таблицу:
python -m scripts.aggregate_results \
    --metrics-dir "${OUTPUT_DIR}/metrics" \
    --csv "${OUTPUT_DIR}/comparison_table_physics.csv"

echo "Done. Results: ${OUTPUT_DIR}/comparison_table_*.csv"
