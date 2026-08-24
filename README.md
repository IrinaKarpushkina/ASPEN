# ASPEN — a clean, methodologically-audited GNN benchmark for sigma-profile prediction (2D + 3D)

This is a from-scratch, methodologically-audited rewrite of a previous
sigma-profile GNN benchmark. It compares **7 message-passing
architectures on 2D molecular graphs** (built from SMILES/RDKit chemical
bonds — no 3D coordinates anywhere in that half of the code) **and 9
architectures on 3D molecular geometry** (radius-graph message passing
and one dense geometric Transformer). Every model is either a
`torch_geometric` layer or a direct reimplementation of its paper's
equations; every design decision is written down in `PROVENANCE.md`,
together with the bugs this rewrite fixes relative to the previous
version.

This README covers the 2D benchmark first (Sections 1-6), then the 3D
benchmark (Section 7). The two share almost all infrastructure
(`src/evaluate.py`, `src/metrics.py`, `scripts/count_params.py`,
`scripts/aggregate_results.py`) but use separate featurizers, datasets,
model directories, configs, and training entrypoints — see
`PROVENANCE.md` for why that separation is deliberate.

## What's compared — 2D

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

## What's compared — 3D

| Model | Paper | Official code |
|---|---|---|
| SchNet | Schutt et al., NeurIPS 2017 | github.com/atomistic-machine-learning/SchNet |
| PaiNN | Schutt et al., ICML 2021 | github.com/atomistic-machine-learning/schnetpack |
| DimeNet | Klicpera et al., ICLR 2020 | github.com/gasteigerjo/dimenet |
| DimeNet++ | Klicpera et al., NeurIPS-W 2020 | github.com/gasteigerjo/dimenet |
| SphereNet | Liu et al., ICLR 2022 | github.com/divelab/DIG |
| EGNN | Satorras et al., ICML 2021 | github.com/vgsatorras/egnn |
| TorchMD-Net | Tholke & De Fabritiis, ICLR 2022 | github.com/torchmd/torchmd-net |
| MACE | Batatia et al., NeurIPS 2022 | github.com/ACEsuit/mace |
| Uni-Mol | Zhou et al., ICLR 2023 | github.com/deepmodeling/Uni-Mol |

All 9 are matched to the SAME ~700,000-parameter budget as the 2D models
(±5%, see `results/param_budget_table_3d.csv`), so 2D and 3D
architectures are comparable to each other too, not just within each
regime. **SphereNet and MACE are explicitly-documented, reduced-fidelity
reimplementations** (see `PROVENANCE.md` §2.5) — read that before citing
either as a byte-for-byte reproduction of its paper. **Uni-Mol is trained
from scratch here** (no pretrained checkpoint is used or available in
this pipeline — see `PROVENANCE.md` §2.5, "Uni-Mol").

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

## 7. The 3D benchmark

### 7.1 Setup
```bash
pip install -r requirements.txt -r requirements-3d.txt
```
`requirements-3d.txt` adds `e3nn` (needed only by MACE — every other 3D
model works without it, see `src/models/models_3d/__init__.py`) and
`sympy` (used by DimeNet/DimeNet++/SphereNet's basis functions, usually
already pulled in transitively by `torch_geometric`).

### 7.2 Data
Same parquet schema as the 2D benchmark (Section 2), **plus optional**
`coord_x`/`coord_y`/`coord_z` columns (one 3D conformer's coordinates per
atom). If present, they're used directly; if absent, one conformer per
molecule is generated once with RDKit (ETKDGv3 + MMFF94) and cached — see
`src/data/features_3d.py` docstring. Edit `configs/3d/base.yaml`'s
`data:` block (note: use a **separate** `cache_dir` from the 2D one).

### 7.3 Sanity-check before training
```bash
python -m pytest tests/ -q -k "_3d"          # DimeNet/DimeNet++/SphereNet tests are slow
                                              # (tens of seconds, sympy basis-function setup
                                              # — not a bug, see PROVENANCE.md §3.5)
python -m scripts.count_params --configs-dir configs/3d
```
`tests/` includes, among others (all 3D-specific tests are suffixed `_3d`):
- `test_equivariance_3d.py` — the central correctness check for this half
  of the benchmark: every model's prediction must be unchanged when the
  input molecule is rotated and/or translated.
- `test_geometry_3d.py` — hand-checkable golden tests for the radius
  graph / triplet / torsion construction, and for the batching-offset fix
  (`Data3D.__inc__`) that makes DimeNet/DimeNet++/SphereNet's triplet
  indices survive being batched with other molecules.
- `test_forward_shapes_3d.py` — smoke test + batch-invariance check for
  every 3D model (mirrors the 2D `test_forward_shapes.py`).
- `test_param_budget_3d.py` — same ±5% budget check as the 2D benchmark.

### 7.4 Train one model
```bash
python -m src.train_3d --config configs/3d/schnet.yaml --seed 0
```
Writes to the SAME `results/checkpoints/` / `results/metrics/`
directories as the 2D benchmark by default (model names don't collide;
each result JSON is tagged `"mode": "3d_pure"` vs `"2d_pure"`) — use
`--output-dir` to keep them fully separate if you prefer.

### 7.5 Run the full 3D benchmark
```bash
MODELS="schnet painn dimenet dimenet_pp spherenet egnn torchmdnet mace unimol" \
SEEDS="0 1 2" bash run_benchmark_3d.sh
```
On SLURM: edit the `#SBATCH` header / conda-activation / `cd` lines in
`run_benchmark_3d.sh`, then `sbatch run_benchmark_3d.sh`.

### 7.6 Aggregate results
```bash
python -m scripts.aggregate_results \
    --metrics-dir results/metrics \
    --loss mse --sort emd_raw \
    --csv results/comparison_table_3d.csv
```
If `results/metrics/` contains BOTH 2D and 3D runs, this will (correctly)
refuse to mix them — pass `--allow-mixed-mode` only if you specifically
want a combined 2D+3D table (they share the same parameter budget, so
this is a legitimate thing to want; see `PROVENANCE.md` §3.5).

### 7.7 Second experiment: real pretrained Uni-Mol (separate from the main benchmark)

Deliberately NOT part of the 9-architecture, 700k-parameter comparison
table above — this uses the real Uni-Mol checkpoint (~47M pretrained
parameters, trained on 209M conformations) via the `unimol_tools` pip
package, frozen, with only a small head trained on top. See
`PROVENANCE.md` §2.5 "Uni-Mol" for why this can't be budget-matched
against the rest of the benchmark, and report its results in a separate
table/section.

```bash
pip install -r requirements-unimol-pretrained.txt

# Run this FIRST -- confirms your unimol_tools version's API and that you
# have network access to download the pretrained checkpoint from
# Hugging Face (huggingface.co) before trying it on the full dataset:
python -m scripts.check_unimol_tools_api

python -m src.train_unimol_pretrained \
    --config configs/unimol_pretrained/unimol_pretrained.yaml --seed 0
```
Writes to `results/checkpoints/unimol_pretrained_seed<seed>_mse.pt` /
`results/metrics/unimol_pretrained_seed<seed>_mse.json` (tagged
`"mode": "unimol_pretrained"`, distinct from `"2d_pure"`/`"3d_pure"`, so
`scripts/aggregate_results.py`'s mixed-mode guard keeps it out of the
main table by default).

## Repository layout

```
configs/2d/             2D model configs (base.yaml + one per architecture)
configs/3d/             3D model configs (base.yaml + one per architecture)
configs/unimol_pretrained/  config for the SEPARATE pretrained-Uni-Mol experiment (§7.7)
data/raw/               <- put your parquet files here
data/cache/              on-disk cache of precomputed 2D graphs (auto-created)
models/                  (currently unused; reserved for saved final models)
results/checkpoints/     training checkpoints (*.pt)
results/metrics/         per-run metrics (*.json; "mode": "2d_pure" or "3d_pure")
src/data/               2D featurizer + dataset (features.py, dataset.py)
src/data/*_3d.py         3D featurizer + dataset + geometry helpers (geometry_3d.py,
                         constants_3d.py, features_3d.py, dataset_3d.py)
src/models/models_2d/    7 2D architectures + shared ResidualMLP head
src/models/models_3d/    9 3D architectures (SchNet, PaiNN, DimeNet, DimeNet++,
                         SphereNet, EGNN, TorchMD-Net, MACE, Uni-Mol)
src/losses/              MSE loss (shared by both)
src/train.py             2D training entrypoint
src/train_3d.py          3D training entrypoint
src/evaluate.py, metrics.py, config.py    shared by both (architecture-agnostic)
scripts/                 count_params.py, aggregate_results.py (shared by both)
tests/                   unit + regression tests; 3D-specific ones suffixed `_3d`
requirements.txt          2D + shared dependencies
requirements-3d.txt       ADDITIONAL 3D-only dependencies (e3nn, sympy)
requirements-unimol-pretrained.txt  ADDITIONAL deps for §7.7 only (unimol_tools, huggingface_hub)
PROVENANCE.md             full audit trail: papers, official repos, bugs fixed,
                         2D §1-3, 3D §2.5/3.5
```
