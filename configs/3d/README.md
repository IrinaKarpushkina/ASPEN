# configs/3d/ — 3D benchmark configs

`base.yaml` + one file per architecture (`schnet.yaml`, `painn.yaml`,
`dimenet.yaml`, `dimenet_pp.yaml`, `spherenet.yaml`, `egnn.yaml`,
`torchmdnet.yaml`, `mace.yaml`, `unimol.yaml`), following the same
`defaults: base.yaml` inheritance pattern as `configs/2d/`.

Every `hidden`/`hidden_channels` value here was set via
`python -m scripts.count_params --configs-dir configs/3d --auto-tune`
(the same, unmodified script the 2D configs use) to land within ±5% of
`base.yaml`'s `target_params` (700,000 — identical to the 2D benchmark's
budget). See each config's inline comment for the measured parameter
count, and `PROVENANCE.md` §3.5/§5 for the full methodology.

See the top-level `README.md` §7 for how to run this, and
`PROVENANCE.md` §2.5 for full per-architecture provenance (paper,
official repo, what — if anything — was adapted for this benchmark's
per-atom sigma-profile task).
