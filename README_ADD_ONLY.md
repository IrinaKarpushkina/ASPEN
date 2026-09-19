# ASPEN physics extension — additive-only patch

This archive is deliberately **additive**. It does not contain replacement
versions of the already-benchmarked files:

- `src/models/models_3d/dimenet.py`
- `src/models/models_3d/painn.py`
- `src/train_3d.py`
- `src/evaluate.py`
- `src/metrics.py`
- `src/data/constants.py`
- `src/data/constants_3d.py`
- `src/data/features_3d.py`
- `src/data/dataset_3d.py`

Therefore installing this archive does not require rerunning the completed
benchmark.

## Install

From the repository root:

```bash
cd /mnt/tank/scratch/ikarpushkina/sigma/ASPEN2D/ASPEN
unzip -o /path/to/ASPEN_physics_extension_additive_fixed.zip
```

The archive paths are relative to the repository root. It only adds files.

## Important

The physics experiments contain isolated copies of the DimeNet++ and PaiNN
backbone implementations under:

```text
src/physics_extension/backbones/
```

These copies are used only by `src/train_3d_physics.py` and do not alter the
existing benchmark implementations.

## Setup

```bash
python scripts/check_physics_setup.py \
  --train /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/chaos_atomic_train_with_coordinates.parquet \
  --cache-dir /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/cache_3d
```

For experiment F:

```bash
python scripts/build_element_prior.py \
  --train /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/chaos_atomic_train_with_coordinates.parquet \
  --output /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/element_sigma_prior.npz
```

Then smoke test:

```bash
SMOKE=1 SEEDS="0" bash run_physics_ablation.sh
```

Full runs:

```bash
SEEDS="0 1 2" bash run_physics_ablation.sh
```

Outputs go to `results/physics/`, separate from the original benchmark
results.
