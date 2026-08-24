# PROVENANCE.md

This document records, for every architecture in this repository: the
paper, the official reference implementation, which PyG layer we use, and
what (if anything) had to change relative to the previous version of this
benchmark. Its purpose is to make every piece of code in `src/models/`
auditable against a citable source — no unexplained design choices.

This repository originally shipped **2D-only**; the 3D benchmark
(SchNet, PaiNN, DimeNet, DimeNet++, SphereNet, EGNN, TorchMD-Net, MACE,
Uni-Mol) has now been added — see `configs/3d/`, `src/models/models_3d/`,
`src/data/*_3d.py`, and Sections 2.5/3.5/5 below. **Nothing in the 2D
code path (`src/data/features.py`, `src/data/dataset.py`,
`src/models/models_2d/`, `configs/2d/`, `src/train.py`) was modified** to
add this — every 3D file is new, and the only pre-existing files touched
at all are this one and `README.md` (documentation only).

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

### Bug #3 — radius-graph density asymmetry (3D-only)
Recorded here before any 3D code existed; **now addressed** in this 3D
implementation. GCN/GAT/GATv2-style architectures without edge features
(none of which are actually in the 3D registry — every 3D architecture
here consumes distance in some form) would oversmooth on a too-dense
cutoff graph. `src/data/constants_3d.py` fixes `CUTOFF = 5.0` Å /
`MAX_NUM_NEIGHBORS = 32` — matching the cutoff SchNet/DimeNet's own QM9
training used (Schutt et al. 2017; Klicpera et al. 2020), not the denser
`cutoff=12Å` mentioned in the original note — and, importantly, uses the
**same** cutoff graph for every 3D architecture that consumes one (see
`constants_3d.py` docstring), which is the 3D equivalent of "every 2D
model sees the same RDKit bond graph" (controlled comparison, Dwivedi et
al. 2022).

### Bug #6 — two 2D test files iterate the COMBINED model registry, not `MODEL_REGISTRY_2D` (found while adding the 3D benchmark)
`tests/test_forward_shapes.py` and `tests/test_no_shared_global_block.py`
both `from src.models import MODEL_REGISTRY` and iterate
`MODEL_REGISTRY.items()` directly — i.e. they were written against the
**combined** 2D+3D registry (`src/models/__init__.py`'s `MODEL_REGISTRY`),
not the 2D-only `MODEL_REGISTRY_2D` also exported from that same module.
This was invisible before this PR because `MODEL_REGISTRY_3D` was empty
(`src/models/models_3d/__init__.py` did not exist yet), so
`MODEL_REGISTRY == MODEL_REGISTRY_2D` by accident. Now that
`MODEL_REGISTRY_3D` is populated (as `src/models/models_3d/README.md`
already anticipated it would be — no change to `src/models/__init__.py`
was needed for that), both tests fail: they try to instantiate 3D models
(`schnet`, `painn`, ...) with 2D-style constructor kwargs
(`hidden`/`heads`/`num_steps`, no `hidden_channels`/`num_filters`/etc.)
and with 2D-shaped toy graphs.

**This is a latent gap in the 2D test suite, not a bug in the 2D
architectures or their production code** — per this rewrite's own
constraint of not modifying existing 2D code/tests, IT HAS NOT BEEN
FIXED HERE. The one-line fix, for whoever owns the 2D test suite, is to
replace `from src.models import MODEL_REGISTRY` with
`from src.models import MODEL_REGISTRY_2D as MODEL_REGISTRY` (or iterate
`MODEL_REGISTRY_2D` directly) in both files. Until that's done, run the
2D-only test subset explicitly when validating a 2D-only change:
```bash
python -m pytest tests/ -q -k "not _3d"   # will still hit the two tests above
# or, once fixed:
python -m pytest tests/test_forward_shapes.py tests/test_no_shared_global_block.py -q
```
The 3D-specific tests (`tests/test_forward_shapes_3d.py`, etc.) correctly
import `MODEL_REGISTRY_3D` from the start and are unaffected.

### Bug #7 — 3D on-disk cache computed/stored triplet indices for EVERY model, even ones that never use them (found on a real training run: 21GB cache for SchNet)
`ChaosParquet3DDataset` originally computed and cached DimeNet-style
triplet/torsion indices (`build_triplets`/`build_torsions`,
`src/data/geometry_3d.py`) UNCONDITIONALLY, regardless of which model the
dataset was being built for. Only 3 of the 9 3D architectures
(DimeNet, DimeNet++, SphereNet) ever read `tri_idx_*`/`tor_idx_*` — the
other 6 (SchNet, PaiNN, EGNN, TorchMD-Net, MACE, Uni-Mol) never touch
them, but paid the full build-time and disk cost anyway. Triplet counts
scale roughly as (avg node degree)^2, so for a dense real dataset (cutoff
5A, organic/ionic-liquid molecules with many H atoms) this produced tens
of GB of cache for a SchNet run that never needed any of it, and — on one
run — outright failed with a disk write error mid-cache.

Fixed two ways:
1. `precompute_molecule_tensors_3d`/`ChaosParquet3DDataset` gained a
   `compute_triplets` flag; `src/train_3d.py` sets it automatically based
   on the model being trained (`_MODELS_NEEDING_TRIPLETS = {"dimenet",
   "dimenet_pp", "spherenet"}`) — nothing to configure by hand.
2. When triplets ARE computed, their index tensors are now stored as
   `int32` instead of PyTorch's `int64` default (molecules are nowhere
   near 2^31 atoms/edges), roughly halving their footprint on top of (1).
   Verified both PyTorch fancy-indexing (`h[idx]`) and
   `torch_geometric.utils.scatter` accept `int32` index tensors directly
   (no cast needed inside model forward passes); `Data3D.__inc__`'s
   batching-offset addition (`tensor + python_int`) also preserves the
   `int32` dtype.

`_cache_key()` includes `compute_triplets` and bumped its version suffix,
so any pre-existing cache built before this fix is automatically
invalidated (rebuilt once) rather than silently reused in an inconsistent
format.

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

## 2.5. Per-architecture provenance — 3D benchmark

Full equations/adaptation rationale live as docstrings in each model file
(kept there, not duplicated in full here, so there is exactly one place
to update if the code changes); this section is the citation index and a
one-paragraph summary of what, if anything, deviates from the paper.

### SchNet — `src/models/models_3d/schnet.py`
- Paper: Schutt et al., *"SchNet: A Continuous-filter Convolutional
  Neural Network for Modeling Quantum Interactions"*, NeurIPS 2017.
  https://papers.nips.cc/paper/6700
- Official: https://github.com/atomistic-machine-learning/SchNet
- Layer used: `torch_geometric.nn.models.schnet.InteractionBlock` /
  `GaussianSmearing` (PyG's official building blocks), called directly,
  stopping before SchNet's own molecule-level pooling — same "reuse the
  official building blocks, adapt for atom-level output" pattern this
  repo already uses for AttentiveFP/GPS in 2D.
- No `torch_cluster`/`pyg-lib` dependency: the radius graph is
  precomputed once per molecule (`src/data/geometry_3d.py`), not built by
  PyG's `RadiusInteractionGraph` at forward time.

### PaiNN — `src/models/models_3d/painn.py`
- Paper: Schutt, Unke & Gastegger, *"Equivariant Message Passing for the
  Prediction of Tensorial Properties and Molecular Spectra"*, ICML 2021.
  https://proceedings.mlr.press/v139/schutt21a.html
- Official: https://github.com/atomistic-machine-learning/schnetpack
- No PyG built-in layer exists (checked against PyG 2.8) — direct
  reimplementation of the paper's scalar/vector message + update block
  equations (Eqs. 5-13). Rotation/translation invariance of the final
  scalar output verified in `tests/test_equivariance_3d.py`.

### DimeNet / DimeNet++ — `src/models/models_3d/dimenet.py`
- Papers: Klicpera, Gross & Gunnemann, *"Directional Message Passing for
  Molecular Graphs"*, ICLR 2020 (https://arxiv.org/abs/2003.03123);
  Klicpera, Giri, Margraf & Gunnemann, DimeNet++, NeurIPS-W 2020.
- Official: https://github.com/gasteigerjo/dimenet (TensorFlow)
- Layer used: `torch_geometric.nn.models.dimenet`'s
  `BesselBasisLayer`/`SphericalBasisLayer`/`InteractionBlock`/
  `InteractionPPBlock`/`OutputBlock`/`OutputPPBlock` (PyG's official
  building blocks), stopping before the molecule-level sum, exactly as
  for SchNet above.
- Triplets (the `(k -> j -> i)` angular indices) are precomputed with a
  from-scratch, `SparseTensor`-free reimplementation of PyG's own
  `triplets()` (`src/data/geometry_3d.py::build_triplets`) — the official
  function requires `torch_sparse`, a compiled extension this repo does
  not depend on. Verified hand-checkably in `tests/test_geometry_3d.py`.
- One registry entry each: `dimenet` (`pp: false`, original DimeNet) and
  `dimenet_pp` (`pp: true`, the paper's own recommended, faster variant) —
  both from the same `DimeNetSigmaModel` class, a single `pp` flag toggle.

### SphereNet — `src/models/models_3d/spherenet.py`
- Paper: Liu, Wang, Liu, Lin, Zhang, Oztekin & Ji, *"Spherical Message
  Passing for 3D Molecular Graphs"*, ICLR 2022.
  https://arxiv.org/abs/2102.05013
- Official: https://github.com/divelab/DIG
- **Explicit, documented simplification** (see the file's full docstring
  for the exact reasoning): reuses DimeNet++'s (distance, angle) basis
  unchanged, and adds an explicit torsion (dihedral) channel — a 4th atom
  found per triplet (`src/data/geometry_3d.py::build_torsions`), embedded
  with a small Fourier basis and used to multiplicatively gate the
  (distance, angle) spherical basis — rather than reproducing the
  official DIG repo's specific local-reference-frame torsion index
  construction (which additionally requires `torch_sparse`/
  `torch_scatter`). This is a reduced-fidelity SphereNet: it genuinely
  gives the model a third (torsion) geometric channel distance+angle-only
  architectures lack, but is not a byte-for-byte reproduction of the
  official implementation's basis functions. Report it as such (e.g.
  "SphereNet-style spherical message passing", not "SphereNet
  (official)") in any write-up.

### EGNN — `src/models/models_3d/egnn.py`
- Paper: Satorras, Hoogeboom & Welling, *"E(n) Equivariant Graph Neural
  Networks"*, ICML 2021. https://arxiv.org/abs/2102.09844
- Official: https://github.com/vgsatorras/egnn
- No PyG built-in layer exists — direct reimplementation of the paper's
  Eqs. 3-6 (`E_GCL`), matched against the official repo's own QM9
  property-prediction script (coordinate channel updated every layer,
  as in the official pipeline, but discarded after the last layer — only
  the scalar channel is read out).

### TorchMD-Net — `src/models/models_3d/torchmdnet.py`
- Paper: Tholke & De Fabritiis, *"TorchMD-NET: Equivariant Transformers
  for Neural Network based Molecular Potentials"*, ICLR 2022.
  https://arxiv.org/abs/2202.02541
- Official: https://github.com/torchmd/torchmd-net
- No PyG built-in layer exists — direct reimplementation of the paper's
  Equivariant Transformer attention block (distance-filtered,
  multi-head, attention-weighted PaiNN-style scalar/vector message),
  reusing this repo's own PaiNN vector-channel utilities
  (`VectorLinear`, cosine cutoff, `PaiNNMixing` update block) for the
  parts the paper itself describes as "following PaiNN".

### MACE — `src/models/models_3d/mace.py`
- Paper: Batatia, Kovacs, Simm, Ortner & Csanyi, *"MACE: Higher Order
  Equivariant Message Passing Neural Networks for Fast and Accurate Force
  Fields"*, NeurIPS 2022. https://arxiv.org/abs/2206.07697
- Official: https://github.com/ACEsuit/mace
- Uses `e3nn` (https://e3nn.org, see `requirements-3d.txt`) for genuine
  Clebsch-Gordan spherical-harmonics tensor products, rather than
  reimplementing CG coefficients from scratch.
- **Explicit, documented simplification**: implements the paper's 2-body
  message ("A" function) plus ONE self-tensor-product product-basis step
  ("B" function at correlation order nu=2, i.e. a genuine 3-body
  equivariant feature) per layer. The official implementation supports
  higher correlation order (nu up to 3) via a custom, numerically-tuned
  `SymmetricContraction` module; this benchmark caps at nu=2 to keep the
  implementation auditable with plain `e3nn.o3.FullyConnectedTensorProduct`
  calls. Report it as "MACE (nu=2)" or "reduced-order MACE" in any
  write-up, not as the paper's default (nu=3) configuration.
- Equivariance of the vector/nu=2 channels and invariance of the final
  scalar readout verified in `tests/test_equivariance_3d.py`.

### Uni-Mol — `src/models/models_3d/unimol.py`
- Paper: Zhou, Gao, Ding, Zheng, Xu, Wei, Zhang & Ke, *"Uni-Mol: A
  Universal 3D Molecular Representation Learning Framework"*, ICLR 2023.
  https://openreview.net/forum?id=6K2RM6wVqKu
- Official: https://github.com/deepmodeling/Uni-Mol
- Dense, all-pairs SE(3)-invariant Transformer maintaining an atom-level
  and a pair-level representation that communicate every layer (paper
  Eq. 1-2) — the one architecture in this benchmark that does NOT use the
  shared cutoff radius graph (`torch_geometric.utils.to_dense_batch`
  instead; no `pyg-lib`/`torch_cluster` needed).
- **REVISED for closer fidelity** after the user supplied the paper text
  directly (the first version of this file was written from general
  recollection of the architecture, not a line-by-line equation check).
  Two concrete fixes, both in `unimol.py`:
  1. Pair-representation update now accumulates the RAW pre-softmax
     `QK^T/sqrt(d)` score per layer (paper Eq. 1), not the post-softmax
     attention probabilities the first version used — these are different
     quantities; the raw score keeps sign/unbounded magnitude information
     the softmax-normalized version discards.
  2. Distance encoding is now a pair-TYPE-aware Gaussian kernel (GKPT:
     an affine transform of the raw distance, with (mul, bias) looked up
     per atom-type PAIR, applied before a shared Gaussian kernel bank —
     paper Sec. 2.1 + Appendix D.1's own ablation, which found this
     outperforms a plain, non-pair-type-aware kernel), replacing the
     first version's plain Gaussian kernel.
  What still differs from the paper's own configuration, and why (scope/
  budget choices, not correctness bugs — see `unimol.py`'s docstring for
  the full list): depth/width (15 layers/512-dim/64-heads in the paper's
  47M-parameter backbone vs. this benchmark's ~700k-parameter shared
  budget), no [CLS] token (not needed for this benchmark's per-atom, not
  per-molecule, task), no SE(3)-equivariant coordinate-prediction head
  (only used for the paper's own PRETRAINING task, never exercised here).
- **IMPORTANT, not a simplification but a scope limitation**: Uni-Mol's
  headline results in the paper come from large-scale self-supervised
  PRETRAINING (masked atom-type prediction + 3D coordinate denoising on
  ~209M conformers) before task-specific fine-tuning. This benchmark
  entry has no access to that pretraining corpus or the official
  checkpoint, and — like every other model here — trains this ENCODER
  ARCHITECTURE from scratch, directly on the sigma-profile task. Report
  it as "Uni-Mol architecture, trained from scratch" / "Uni-Mol (no
  pretraining)" in any write-up, not as a reproduction of the paper's
  pretrained numbers.
- **A SECOND, separate experiment DOES use the real pretrained
  checkpoint** — via the `unimol_tools` pip package (a lightweight,
  fairseq/Uni-Core-free wrapper released by DeepModeling in 2024) rather
  than the original heavier Uni-Core/LMDB pipeline. See
  `src/models/models_3d/unimol_pretrained.py`,
  `src/data/dataset_unimol_pretrained.py`,
  `src/train_unimol_pretrained.py`, and
  `configs/unimol_pretrained/unimol_pretrained.yaml`. This is
  deliberately kept OUT of the main 9-architecture, 700k-parameter
  comparison table (the real pretrained backbone has ~47M parameters —
  not a budget-matched entry by construction) and reports as a frozen-
  feature-extraction baseline in its own separate table/section instead.
  Requires `requirements-unimol-pretrained.txt`; run
  `python -m scripts.check_unimol_tools_api` first to confirm your
  installed `unimol_tools` version's API and network access to Hugging
  Face before running this on the full dataset.
- **Confirmed finding (via `check_unimol_tools_api.py` on a real toy
  molecule): `unimol_tools`'s `atomic_reprs` includes a PREPENDED
  whole-molecule [CLS] representation** (paper Sec. 2.2: "a special atom
  [CLS] ... is used to represent the whole molecule/pocket", BERT-style
  convention) — a 9-atom test molecule returned 10 rows. Verified by
  comparing row 0 against the separately-returned `cls_repr`.
  `dataset_unimol_pretrained.py` now strips this leading row (when the
  count is exactly `len(atoms)+1`) before checking per-atom alignment
  with `sigma_*` targets, instead of hard-failing on every molecule.

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

## 3.5. Shared infrastructure — 3D benchmark

New files, all additive (see the header note at the top of this
document):

- `src/data/geometry_3d.py` — radius graph, DimeNet-style triplets,
  SphereNet-style torsions, all in pure PyTorch (no `torch_cluster`/
  `torch_sparse`/`pyg-lib`). Verified hand-checkably in
  `tests/test_geometry_3d.py`.
- `src/data/constants_3d.py` — shared cutoff/neighbour-cap (`CUTOFF`,
  `MAX_NUM_NEIGHBORS`, see Bug #3 above) and the 6-dim, purely physical
  (non-topological) node feature set used by every 3D architecture.
- `src/data/features_3d.py` — builds per-molecule node features,
  coordinates (from `coord_x/y/z` parquet columns, or a documented RDKit
  ETKDGv3+MMFF94 conformer-generation fallback if absent), the shared
  radius graph, and triplet/torsion indices. Verified in
  `tests/test_features_3d.py`.
- `src/data/dataset_3d.py` — `ChaosParquet3DDataset` + `Data3D` (a
  `torch_geometric.data.Data` subclass with a corrected `__inc__` for
  batching the triplet/torsion index tensors — see that class's
  docstring and `tests/test_geometry_3d.py`'s dedicated batching test).
- `src/models/models_3d/__init__.py` — `MODEL_REGISTRY_3D`; each import
  wrapped individually so a missing OPTIONAL dependency (only `e3nn`, for
  MACE) never breaks the other 8 architectures.
- `src/train_3d.py` — training entrypoint, structurally identical to
  `src/train.py` except for the dataset/featurizer and
  `"mode": "3d_pure"` (see Bug #1: this is exactly the field whose
  absence caused the original duplicated-table bug).
- `requirements-3d.txt` — ADDITIONAL dependencies (`e3nn`, for MACE only;
  `sympy`, for DimeNet/DimeNet++/SphereNet's basis functions), kept
  separate from `requirements.txt` so the 2D benchmark's dependency
  footprint is unchanged.
- `run_benchmark_3d.sh` — mirrors `run_benchmark_2d.sh`.

Every 3D architecture targets the SAME `target_params` (700,000, see
`configs/3d/base.yaml`) as the 2D benchmark, verified in
`tests/test_param_budget_3d.py` — so 2D and 3D architectures are
comparable to each other, not just within each regime.

---

## 4. What to check before trusting a training run
Run, in order, before spending GPU time:
```bash
# 2D:
python -m pytest tests/ -q -k "not _3d"
python -m scripts.count_params --configs-dir configs/2d
# 3D (needs requirements-3d.txt; DimeNet/DimeNet++/SphereNet tests are
# slow -- tens of seconds -- due to SphericalBasisLayer's sympy setup,
# not a bug):
python -m pytest tests/ -q -k "_3d"
python -m scripts.count_params --configs-dir configs/3d
```
Both `run_benchmark_2d.sh` and `run_benchmark_3d.sh` already run the
relevant checks and abort the job if they fail.

## Uni-Mol pretrained (frozen backbone) — atom-vocabulary limitation
26/7941 (val), TBD/36981 (train), TBD/7800 (test) molecules were excluded
from the frozen-pretrained-Uni-Mol experiment: unimol_tools' mol_pre_all_h_220816.pt
checkpoint silently drops atoms whose element isn't in its pretraining
atom-type vocabulary (confirmed: Ba, Sb, Ge, Ga, Te, Be, In, Bi -- rare/
heavy elements not typical in drug-like organic pretraining data). This
does NOT affect the main from-scratch benchmark (all 9 architectures
there handle arbitrary atomic numbers).
