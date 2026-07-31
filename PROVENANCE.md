# PROVENANCE.md

This document records, for every architecture in this repository: the
paper, the official reference implementation, which PyG layer we use, and
what (if anything) had to change relative to the previous version of this
benchmark. Its purpose is to make every piece of code in `src/models/`
auditable against a citable source — no unexplained design choices.

This repository is **2D-only**. 3D architectures (SchNet, PaiNN, ...)
are intentionally not implemented — see `configs/3d/` and
`src/models/models_3d/` (placeholders) for where they will go.

---

## 1. Bugs found in the previous benchmark, and how this rewrite fixes them

### Bug #1 — duplicated result table (found first)
`results/comparison_table_level1_2d.csv` at the repo root was byte-identical
to the 3D comparison table. Root cause: `scripts/aggregate_results.py` had
`--metrics-dir` default to `results/metrics` (the 3D folder), and at some
point it was invoked for the "2D" table without overriding that default.
Nothing in the output JSON recorded which regime (`mode`) produced it, so
the mistake was invisible after the fact.

**Fix in this repo:**
- `scripts/aggregate_results.py` — `--metrics-dir` is now a **required**
  argument (no default at all).
- `src/train.py` now writes a `"mode": "2d_pure"` field (and the resolved
  config path) into every result JSON.
- `scripts/aggregate_results.py` refuses to aggregate a directory whose
  JSON files disagree on `mode` (`--allow-mixed-mode` opts back in,
  explicitly).

### Bug #2 — shared global-pooling block compressing architecture differences
Six of seven architectures (GCN, GAT, GATv2, GINE, AttentiveFP, DMPNN) ended
with an **identical**, hand-written block:
```python
g_mean = global_mean_pool(h, data.batch)[data.batch]
g_max  = global_max_pool(h, data.batch)[data.batch]
h = global_norm(h + global_proj(cat([h, g_mean, g_max], -1)))
```
None of the original papers for these architectures include this — GCN,
GAT, GATv2, GIN/GINE and D-MPNN are all originally node-/edge-level
architectures with no molecule-level pooling. Because the block was
literally shared, a large fraction of predictive power could come from it
rather than from the conv layers actually being compared — this is the
likely cause of the unusually tight R² spread (0.842–0.857) observed
across architectures in the pure-2D benchmark.

**Fix in this repo:** removed entirely for GCN/GAT/GATv2/GINE/DMPNN (see
per-architecture sections below). AttentiveFP now uses its **own**
super-node attention readout instead. GPS needed no change (see below).
`tests/test_no_shared_global_block.py` is a regression test for this.

### Bug #3 — radius-graph density asymmetry (3D-only, informational)
Not applicable to this repository (2D-only), but recorded for when the 3D
benchmark is implemented: GCN/GAT/GATv2 do not consume `edge_attr`, so in
the old 3D benchmark the *only* thing that changed for them between 2D and
3D was topology — and a `cutoff=12Å, max_num_neighbors=32` radius graph is
likely near-complete for small organic molecules, causing oversmoothing.
When you implement `src/models/models_3d/`, either (a) use a chemically
motivated smaller cutoff for these three architectures, or (b) pass
`edge_weight`/`edge_attr` into `GCNConv`/`GATConv`/`GATv2Conv` (all three
support it) so they can learn to down-weight distant neighbours.

### Bug #4 — D-MPNN reverse-edge index built backwards (found while writing tests for this rewrite)
`build_reverse_index()` (inherited from the previous benchmark) built its
lookup dictionary keyed by the *swapped* `(j, i)` pair and then looked up
`(j, i)` again — the net effect was `reverse_idx[e] == e` for every edge,
i.e. **the reverse edge was never found**, and D-MPNN silently degenerated
into subtracting its own current bond state instead of the true reverse
bond state. This defeats the entire point of D-MPNN's reverse-edge masking
(the "totter" fix that distinguishes it from a plain edge-centric MPNN).
Caught by `tests/test_dmpnn_reverse_edge.py`, written for this rewrite —
this is exactly the kind of bug unit tests on toy graphs are for. **Fixed**
in `src/models/dmpnn.py::build_reverse_index`.

### Bug #5 — unnecessary `torch_scatter` dependency
The previous `dmpnn.py` imported `torch_scatter.scatter` even though the
project's own `requirements.txt` claimed no compiled scatter/cluster
extension was needed. Fixed by switching to
`torch_geometric.utils.scatter`, which ships with PyG itself.

---

## 2. Per-architecture provenance

### GCN — `src/models/gcn.py`
- Paper: Kipf & Welling, *"Semi-Supervised Classification with Graph
  Convolutional Networks"*, ICLR 2017. https://arxiv.org/abs/1609.02907
- Official: https://github.com/tkipf/gcn (TensorFlow) /
  https://github.com/tkipf/pygcn (author's PyTorch port)
- Layer used: `torch_geometric.nn.GCNConv`
- Verified against paper eq. 2 (normalized-adjacency propagation) in
  `tests/test_gcn_conv_formula.py`.
- Change vs. previous benchmark: removed the shared global-pooling block.
  Model now ends with the conv stack only, per the original (node-level)
  architecture.

### GAT — `src/models/gat.py`
- Paper: Velickovic et al., *"Graph Attention Networks"*, ICLR 2018.
  https://arxiv.org/abs/1710.10903
- Official: https://github.com/PetarV-/GAT
- Layer used: `torch_geometric.nn.GATConv`
- GATConv does not consume `edge_attr` in the original formulation —
  attention weights come from node features only.
- Change vs. previous benchmark: removed the shared global-pooling block.

### GATv2 — `src/models/gatv2.py`
- Paper: Brody, Alon & Yahav, *"How Attentive are Graph Attention
  Networks?"*, ICLR 2022. https://arxiv.org/abs/2105.14491
- Official: https://github.com/tech-srl/how_attentive_are_gats
- Layer used: `torch_geometric.nn.GATv2Conv`
- Change vs. previous benchmark: removed the shared global-pooling block.

### GINE — `src/models/gine.py`
- Papers: Xu et al., *"How Powerful are Graph Neural Networks?"*, ICLR
  2019 (GIN, https://arxiv.org/abs/1810.00826, official:
  https://github.com/weihua916/powerful-gnns); Hu et al., *"Strategies for
  Pre-training Graph Neural Networks"*, ICLR 2020 (GINE — the edge-feature
  extension, https://arxiv.org/abs/1905.12265, official:
  https://github.com/snap-stanford/pretrain-gnns)
- Layer used: `torch_geometric.nn.GINEConv`
- GIN's own paper only pools to a graph vector for *graph-level* tasks; its
  node-level pretext tasks (context prediction, attribute masking) in the
  official pretrain-gnns repo use the raw per-node output, no pooling —
  the precedent this repo follows for the (node-level) sigma-profile task.
- Change vs. previous benchmark: removed the shared global-pooling block.

### D-MPNN (Chemprop) — `src/models/dmpnn.py`
- Paper: Yang et al., *"Analyzing Learned Molecular Representations for
  Property Prediction"*, J. Chem. Inf. Model. 2019.
  https://arxiv.org/abs/1904.01561
- Official: https://github.com/chemprop/chemprop
- Reimplemented directly from the paper's equations (directed
  message-passing with reverse-edge masking) rather than via a PyG layer,
  since PyG has no built-in D-MPNN conv. Chemprop v2 supports an explicit
  atom-level output mode, which is the precedent for the per-atom readout
  used here (no sum-to-molecule step).
- Changes vs. previous benchmark: removed the shared global-pooling block
  (Bug #2); fixed `build_reverse_index` (Bug #4); dropped `torch_scatter`
  dependency in favour of `torch_geometric.utils.scatter` (Bug #5).
- Verified with `tests/test_dmpnn_reverse_edge.py`.

### AttentiveFP — `src/models/attentive_fp.py`
- Paper: Xiong et al., *"Pushing the Boundaries of Molecular
  Representation for Drug Discovery with the Graph Attention Mechanism"*,
  J. Med. Chem. 2020. https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
- Official reference implementation:
  https://github.com/OpenDrugAI/AttentiveFP
- Layer used: `torch_geometric.nn.models.AttentiveFP` (PyG's official
  implementation) — we instantiate it wholesale and call its internal
  submodules directly (`lin1`, `gate_conv`, `gru`, `atom_convs`,
  `atom_grus`, `mol_conv`, `mol_gru`), rather than reimplementing GATE
  attention ourselves.
- **This is the one architecture where "remove the global block" was not
  the right fix** — AttentiveFP's paper defines its own molecule-level
  super-node attention readout (`mol_conv`/`mol_gru`, verified against
  PyG's source at
  `torch_geometric/nn/models/attentive_fp.py::AttentiveFP.forward`). The
  previous benchmark discarded this paper-specific mechanism and replaced
  it with the generic shared block. This rewrite instead **reuses the
  model's own molecule-embedding stage** verbatim (`global_add_pool` init,
  `mol_conv`/`mol_gru` loop over `num_timesteps`), then broadcasts the
  resulting molecule embedding back onto every atom (concatenated) before
  the final per-atom head — the natural per-atom analogue of what the
  original model does with `lin2` for whole-molecule output.
- Set `model.use_molecule_context: false` in `configs/attentive_fp.yaml`
  to disable this and use atom embeddings alone, if you want a version
  with strictly zero global context for comparison.

### GraphGPS — `src/models/gps.py`
- Paper: Rampasek et al., *"Recipe for a General, Powerful, Scalable
  Graph Transformer"*, NeurIPS 2022. https://arxiv.org/abs/2205.12454
- Official: https://github.com/rampasek/GraphGPS
- Layer used: `torch_geometric.nn.GPSConv`, following PyG's official
  example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
- **No change required.** GPSConv already provides molecule-wide context
  through its built-in global multi-head attention, exactly as the paper
  specifies — this was the one architecture in the previous benchmark that
  did *not* have (and did not need) the shared pooling block. Kept as the
  reference case for "what a model with a genuine, paper-specified
  global-context mechanism looks like".

---

## 3. Shared infrastructure — audited, not part of the bug

These are used identically by every architecture and were reviewed but
**not changed** (no architecture-dependent bug found in them):

- `src/data/constants.py` — element/physical-property lookup tables and
  the sigma-profile bin grid. Pure data, no message-passing logic.
- `src/data/features.py` — the 2D featurizer (14-dim node / 7-dim edge
  features from RDKit chemical bonds only). Identical input for every
  architecture; verified in `tests/test_features_2d_only.py` to contain
  no coordinate-derived quantity.
- `src/metrics.py` — weighted R², weighted MAE, polar MAE, cosine
  similarity, EMD (raw & normalized), and their molecule-level
  (summed-profile) counterparts. Pure evaluation math.
- `src/models/layers.py::ResidualMLP` — the per-atom output head
  (LayerNorm → FC → SiLU → FC+residual → FC). Applied independently to
  each atom's already-computed embedding; unlike the removed global-pool
  block, it does not mix information between atoms, so sharing its
  *shape* across architectures is not a comparability confound — it is
  the standard practice of putting every model through the same
  prediction head before the loss.
- Parameter-budget matching (`scripts/count_params.py`,
  `target_params` in `configs/base.yaml`) follows Dwivedi et al., JMLR
  2022, *"Benchmarking Graph Neural Networks"*: compare architectures at
  matched parameter count (±5%), not matched hidden size, so that
  differences in the table reflect architecture, not raw capacity.

## 4. What to check before trusting a training run
Run, in order, before spending GPU time:
```bash
python -m pytest tests/ -q
python -m scripts.count_params
```
Both are wired into `run_benchmark_2d.sh` already and will abort the job
if they fail.
