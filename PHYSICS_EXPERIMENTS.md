# ASPEN: physics-informed sigma-profile experiments

## What is kept fixed

The target is **still exactly the 51-bin atomic sigma profile**. The original
`src/train_3d.py`, benchmark dataset and benchmark result files are not
modified.

The experiments add only:
1. richer invariant atom/local-environment features;
2. molecular context after DimeNet++;
3. a physically constrained profile decoder;
4. an element-specific residual prior;
5. the same physical decoder on an equivariant PaiNN backbone.

## Experiment matrix

| Model | Features | Decoder | Loss | Scientific question |
|---|---|---|---|---|
| `dimenet_pp_enhanced` | 21 invariant features | ordinary positive MLP | MSE | Do richer chemistry/local geometry features help? |
| `dimenet_pp_physics` | original 6 | area + first-moment constrained | physics | Does physical parameterization help by itself? |
| `dimenet_pp_global` | 21 | molecular FiLM + physical head | physics | Does non-local molecular context help? |
| `dimenet_pp_delta` | 21 | element prior + residual + physical head | physics | Is learning deviations from element-specific priors easier? |
| `painn_physics` | 21 | physical head | physics | Does an explicitly equivariant backbone improve the same target? |

The benchmark table in the supplied repository actually has `spherenet`
slightly ahead of `dimenet_pp` on the reported mean metrics. Therefore,
`dimenet_pp` should be described as the best **DimeNet-family** model, not
literally the best 3D model. SphereNet is an excellent candidate for the
same physical-head wrapper after these first experiments.

## Setup

Copy the files into the repository preserving their relative paths.

Build the element prior from TRAIN ONLY:

```bash
python scripts/build_element_prior.py \
  --train /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/chaos_atomic_train_with_coordinates.parquet \
  --output /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/element_sigma_prior.npz
```

Then run one smoke test:

```bash
python -m src.train_3d_physics \
  --config configs/3d/dimenet_pp_physics.yaml \
  --seed 0 \
  --output-dir results/physics
```

Then the full first-pass ablation:

```bash
SEEDS="0 1 2" bash run_physics_ablation.sh
```

## Important interpretation

Do **not** compare only R2. The physics head is designed to improve:
- weighted MAE;
- raw EMD;
- polar MAE;
- molecular reconstruction;
- integral consistency;
- first-moment / screening-charge consistency;
- profile smoothness.

For a paper-quality ablation, compare:
1. original DimeNet++ + MSE;
2. DimeNet++ + physics loss only;
3. DimeNet++ + enhanced features + MSE;
4. DimeNet++ + enhanced features + physics loss;
5. DimeNet++ + global context + physics;
6. DimeNet++ + element prior + physics;
7. PaiNN + physics;
8. SphereNet + the same physics head.

Most importantly, keep the train/validation/test split unchanged and use the
same seeds for every architecture.
