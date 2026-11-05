#!/bin/bash
# Установка абляции «квантильная голова». Запуск из корня проекта ПОСЛЕ распаковки архива:
#   bash install_quantile.sh
set -e
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN
source /mnt/tank/scratch/ikarpushkina/miniconda3/etc/profile.d/conda.sh
conda activate sigma

python scripts/patch_quantile.py          # подключает голову в build_model (бэкап .bak_qf)
python scripts/make_qf_config.py          # configs/3d/sigma_dn_geom_qf.yaml из вашего sigma_dn_geom.yaml

G=configs/stats/groups_ablation.yaml
if grep -q "Q10\." "$G"; then
  echo "группы Q10 уже есть в $G"
else
  cp "$G" "$G.bak_qf"; cat configs/stats/groups_quantile_add.yaml >> "$G"; echo "группы Q10 добавлены в $G (бэкап $G.bak_qf)"
fi
echo "OK. Дальше: проверка (scripts/check_quantile_head.py) и smoke-тест, см. инструкцию."
