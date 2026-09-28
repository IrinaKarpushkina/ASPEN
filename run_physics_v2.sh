#!/bin/bash
#SBATCH --job-name=sigma_v2
#SBATCH --partition=aichem
#SBATCH --nodelist=aichem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=172:00:00
#SBATCH --output=logs/sigma_v2_%j.out
#SBATCH --error=logs/sigma_v2_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Прогон новых архитектур/расписания (sigma_global.py, physics_stable.py, loss_v2.py)
# поверх старого physics-бенчмарка. Требует, чтобы ОБА патча уже были применены:
#   python scripts/patch_train_3d_physics.py
#   python scripts/patch_delta_and_loss_v2.py
#
# sigma_dn_neutral НЕ включён: ceiling_analysis (часть B) показал, что молекулярная
# нейтральность заряда в этих данных не выполняется (actual |Q|/sum|q| БОЛЬШЕ shuffled,
# только 0.9% молекул близки к нейтральности) — этот constraint для ваших данных неверен.
#
# dimenet_pp_enhanced_sched использует то же имя модели "dimenet_pp_enhanced", что и
# старый baseline (loss mse) — поэтому output-dir здесь ОБЯЗАТЕЛЬНО отдельный
# (results/physics_v2), иначе файлы результатов перезапишут/перепутаются со старыми.
#
# EDIT before running:
#   - conda env / cd путь ниже, если отличаются от прошлого запуска
#   - #SBATCH --partition/--nodelist под ваш кластер
#   - при первом запуске каждой модели рекомендуется сначала smoke-прогон вручную:
#       python -m src.train_3d_physics --config configs/3d/sigma_dn_geom.yaml --seed 0 --smoke
#     и только потом ставить в этот sbatch-скрипт
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs results/physics_v2/checkpoints results/physics_v2/metrics

source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

# По умолчанию — без sigma_dn_neutral (см. пояснение выше) и без delta (запускается
# отдельно, ниже, т.к. может упасть и не должна блокировать остальные модели в очереди).
MODELS="${MODELS:-dimenet_pp_enhanced_sched sigma_dn_geom sigma_dn_global sigma_dn_full sigma_pn_full}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/physics_v2}"

echo "Models: $MODELS"
echo "Seeds:  $SEEDS"
echo "Output: $OUTPUT_DIR"
echo ""

for model in $MODELS; do
    config="configs/3d/${model}.yaml"

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
    --loss physics \
    --sort emd_raw \
    --csv "${OUTPUT_DIR}/comparison_table_physics.csv"

python -m scripts.aggregate_results \
    --metrics-dir "${OUTPUT_DIR}/metrics" \
    --loss mse \
    --sort emd_raw \
    --csv "${OUTPUT_DIR}/comparison_table_mse.csv"

echo "Done. Results: ${OUTPUT_DIR}/comparison_table_*.csv"
