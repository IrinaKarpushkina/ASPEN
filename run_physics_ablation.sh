#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   SEEDS="0 1 2" bash run_physics_ablation.sh
#   SMOKE=1 SEEDS="0" bash run_physics_ablation.sh

SEEDS=${SEEDS:-"0 1 2"}
SMOKE=${SMOKE:-0}

configs=(
  configs/3d/dimenet_pp_enhanced.yaml
  configs/3d/dimenet_pp_physics.yaml
  configs/3d/dimenet_pp_enhanced_physics.yaml
  configs/3d/dimenet_pp_global.yaml
  configs/3d/painn_physics.yaml
  configs/3d/dimenet_pp_delta.yaml
)

for cfg in "${configs[@]}"; do
  for seed in $SEEDS; do
    args=(python -m src.train_3d_physics --config "$cfg" --seed "$seed")
    if [[ "$SMOKE" == "1" ]]; then
      args+=(--smoke)
    fi
    echo "============================================================"
    echo "Running ${cfg} seed=${seed}"
    echo "============================================================"
    "${args[@]}"
  done
done
