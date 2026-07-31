# ASPEN2D — a clean, 2D-only GNN benchmark for sigma-profile prediction

This is a from-scratch, methodologically-audited rewrite of a previous
sigma-profile GNN benchmark. It compares 7 message-passing architectures
on **2D molecular graphs only** (built from SMILES/RDKit chemical bonds —
no 3D coordinates anywhere in the model code). Every model is either a
`torch_geometric` layer or a direct reimplementation of its paper's
equations; every design decision is written down in `PROVENANCE.md`,
together with the bugs this rewrite fixes relative to the previous
version.

**3D architectures (SchNet, PaiNN, ...) are intentionally not
implemented here** — see `configs/3d/` and `src/models/models_3d/`
(currently empty placeholders) for where they will go later.

## What's compared

| Model | Paper | Official code |
|---|---|---|
| GCN | Kipf & Welling, ICLR 2017 | github.com/tkipf/gcn |
| GAT | Velickovic et al., ICLR 2018 | github.com/PetarV-/GAT |
| GATv2 | Brody et al., ICLR 2022 | github.com/tech-srl/how_attentive_are_gats |
| GINE | Xu et al. 2019 / Hu et al. 2020 | github.com/weihua916/powerful-gnns, github.com/snap-stanford/pretrain-gnns |
| D-MPNN | Yang et al. 2019 (Chemprop) | github.com/chemprop/chemprop |
| AttentiveFP | Xiong et al. 2020 | github.com/OpenDrugAI/AttentiveFP |
| GraphGPS | Rampasek et al., NeurIPS 2022 | github.com/rampasek/GraphGPS |

All 7 are matched to ~700,000 parameters (±5%, see
`results/param_budget_table.csv`), and none of them share the bolt-on
global-pooling block that compressed architecture differences in the
previous benchmark — see `PROVENANCE.md` §1 "Bug #2" for why that
mattered, and §2 for what each model does instead.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate     # or your usual conda env
pip install -r requirements.txt
```

Requires Python ≥3.10, a working RDKit install, and (for real training
runs) a CUDA GPU — the pipeline runs on CPU too, just slowly.

## 2. Put your data in place

Drop your train/val/test parquet files into `data/raw/` (or point
`configs/base.yaml` at wherever they already live). Expected schema —
**one row per atom**:

| column | meaning |
|---|---|
| `mol_id` | molecule identifier (grouping key) |
| `atom_index` | per-molecule atom order, 0..N-1 |
| `element` | element symbol (`'H'`, `'C'`, `'N'`, ...) |
| `smiles` | SMILES string for the whole molecule (same value for every row of a given `mol_id`) |
| `sigma_0` … `sigma_50` | 51 sigma-profile bins (the prediction target) |

`coord_x`/`coord_y`/`coord_z` columns, if present, are simply **ignored**
— this is the 2D-only benchmark on purpose (see `src/data/dataset.py`
docstring).

Then edit the paths in `configs/base.yaml`:
```yaml
data:
  train_path: data/raw/chaos_atomic_train.parquet
  val_path:   data/raw/chaos_atomic_val.parquet
  test_path:  data/raw/chaos_atomic_test.parquet
  cache_dir:  data/cache
```
The first run per split will parse every molecule with RDKit and cache
the resulting graphs to `cache_dir` (a few minutes to tens of minutes
depending on dataset size); subsequent runs on the same split load
instantly from cache.

## 3. Sanity-check before training (fast, CPU-only, ~5 seconds)

```bash
python -m pytest tests/ -q
python -m scripts.count_params          # confirms all 7 models are within +/-5% of target_params
```

`tests/` includes, among others:
- a regression test that no architecture has the old shared
  global-pooling block back (`test_no_shared_global_block.py`)
- a golden test comparing `GCNConv`'s output to Kipf & Welling's actual
  propagation formula on a hand-built graph (`test_gcn_conv_formula.py`)
- a golden test for D-MPNN's reverse-edge masking
  (`test_dmpnn_reverse_edge.py`) — this is how Bug #4 in `PROVENANCE.md`
  was found
- a forward-pass smoke test + batch-invariance check for every model
  (`test_forward_shapes.py`)

If you change any model's architecture, `scripts/count_params.py
--auto-tune` will suggest a new `hidden` to keep the parameter budget
matched — update the corresponding `configs/<model>.yaml` and re-run the
tests above.

## 4. Train one model

```bash
python -m src.train --config configs/gine.yaml --seed 0
```
Writes `results/checkpoints/gine_seed0_mse.pt` and
`results/metrics/gine_seed0_mse.json` (the latter now records `"mode":
"2d_pure"` and the config path used — see `PROVENANCE.md` Bug #1 for why).

## 5. Run the full benchmark (all 7 models x 3 seeds)

Locally:
```bash
MODELS="gcn gat gatv2 gine dmpnn attentive_fp gps" SEEDS="0 1 2" bash run_benchmark_2d.sh
```

On a SLURM cluster, edit the `#SBATCH` header and the conda-activation /
`cd` lines in `run_benchmark_2d.sh` for your cluster, then:
```bash
sbatch run_benchmark_2d.sh
```
This script runs the test suite first and aborts if it fails, then trains
every (model, seed) pair not already present in `results/metrics/`, then
aggregates everything into `results/comparison_table_2d.csv` (+ a
ready-to-paste `results/comparison_table_2d.tex` LaTeX table).

## 6. Aggregate / re-aggregate results manually

```bash
python -m scripts.aggregate_results \
    --metrics-dir results/metrics \
    --loss mse \
    --sort emd_raw \
    --csv results/comparison_table_2d.csv
```
Note `--metrics-dir` has **no default** — this is deliberate (see
`PROVENANCE.md` Bug #1). The script also refuses to mix result files from
different `mode`s in one table unless you pass `--allow-mixed-mode`.

## Repository layout

```
configs/            2D model configs (base.yaml + one per architecture)
configs/3d/          <- empty, placeholder for a future 3D benchmark
data/raw/            <- put your parquet files here
data/cache/           on-disk cache of precomputed graphs (auto-created)
models/               (currently unused; reserved for saved final models)
results/checkpoints/  training checkpoints (*.pt)
results/metrics/      per-run metrics (*.json)
src/data/            featurizer + dataset
src/models/           7 architectures + shared ResidualMLP head
src/models/models_3d/ <- empty, placeholder for SchNet/PaiNN/etc.
src/losses/          MSE loss
src/train.py, evaluate.py, metrics.py, config.py
scripts/              count_params.py, aggregate_results.py
tests/                 unit + regression tests (run before every training job)
PROVENANCE.md          full audit trail: papers, official repos, bugs fixed
```

## Adding the 3D benchmark later

`configs/3d/` and `src/models/models_3d/` are placeholders on purpose.
When you add e.g. SchNet:
1. Read `PROVENANCE.md` §1 "Bug #3" first — it documents a specific
   pitfall (radius-graph density) to avoid for GCN/GAT/GATv2's 3D variants.
2. Keep the 2D and 3D featurizers in separate files (as they are now:
   `src/data/features.py` is 2D-only) rather than a `mode=` flag inside
   one shared function — that separation is what made Bug #2 possible to
   find and fix cleanly here.
3. Re-run `scripts/count_params.py --auto-tune` for the 3D configs; do not
   assume the 2D `hidden` values transfer (input/edge dims differ).
