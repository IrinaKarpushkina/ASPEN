"""Idempotent patch: (1) make dimenet_pp_delta trainable, (2) enable `loss: name: physics_v2`.

    python scripts/patch_delta_and_loss_v2.py           # patches models.py and train_3d_physics.py
    python scripts/patch_delta_and_loss_v2.py --check

Independent of scripts/patch_train_3d_physics.py (either order works). Backups: *.bak_pre_v2.
New optional keys in the model yaml for dimenet_pp_delta (defaults keep the old behaviour):
    stable: true        -> StablePhysicsSigmaHead (bisection projection + calibrated area init)
    prior_floor: 1.0e-3 -> mixes the element prior with a uniform floor (min logit -6.9, not -27.6)
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "physics_extension" / "models.py"
TRAIN = ROOT / "src" / "train_3d_physics.py"

MODELS_EDITS = [
    ("from .physics import PhysicsSigmaHead\n",
     "from .physics import PhysicsSigmaHead\nfrom .physics_stable import StablePhysicsSigmaHead\n"),
    ("    def __init__(self, input_dim=22, hidden_channels=118, dropout=0.05, prior_path=None, prior_strength=1.0, **kwargs):\n",
     "    def __init__(self, input_dim=22, hidden_channels=118, dropout=0.05, prior_path=None, prior_strength=1.0,\n"
     "                 prior_floor=0.0, stable=False, **kwargs):\n"),
    ('        self.register_buffer("prior_logp", torch.as_tensor(prior_arr, dtype=torch.float32))\n'
     '        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)\n'
     '        self.head = PhysicsSigmaHead(hidden_channels, 51, prior_strength=prior_strength)\n',
     '        if prior_floor > 0:\n'
     '            pr = np.exp(prior_arr.astype(np.float64))\n'
     '            pr = (pr + prior_floor) / (1.0 + prior_arr.shape[1] * prior_floor)\n'
     '            prior_arr = np.log(pr).astype(np.float32)\n'
     '        self.register_buffer("prior_logp", torch.as_tensor(prior_arr, dtype=torch.float32))\n'
     '        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)\n'
     '        head_cls = StablePhysicsSigmaHead if stable else PhysicsSigmaHead\n'
     '        self.head = head_cls(hidden_channels, 51, prior_strength=prior_strength)\n'),
]

TRAIN_EDITS = [
    ("from .physics_extension.physics import PhysicsSigmaLoss\n",
     "from .physics_extension.physics import PhysicsSigmaLoss\nfrom .physics_extension.loss_v2 import PhysicsSigmaLossV2\n"),
    ('    else:\n        raise ValueError(f"Unknown loss: {loss_name}")\n',
     '    elif loss_name == "physics_v2":\n'
     '        profile_scale = train_profile_scale(train_ds)\n'
     '        logger.info("TRAIN-only profile MSE scale: %.8g", profile_scale)\n'
     '        criterion = PhysicsSigmaLossV2(\n'
     '            bin_weights=bin_weights, profile_scale=profile_scale,\n'
     '            **{k: v for k, v in loss_cfg.items() if k != "name"}).to(device)\n'
     '    else:\n        raise ValueError(f"Unknown loss: {loss_name}")\n'),
    ("    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
     "    for _m in model.modules():\n"
     "        if hasattr(_m, \"calibrate_area\"):\n"
     "            _m.calibrate_area(train_ds)\n"
     "    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"),
    ("                loss = criterion(model(data), data.y) / accum\n",
     "                _out = model(data)\n"
     "                loss = (criterion(_out, data.y, data.batch) if getattr(criterion, \"needs_batch\", False)\n"
     "                        else criterion(_out, data.y)) / accum\n"),
]


def apply(path, edits, marker, check):
    src = path.read_text()
    if marker in src:
        print(f"{path.name}: already patched")
        return
    bad = [o[:60] for o, _ in edits if src.count(o) != 1]
    if bad:
        raise SystemExit(f"{path.name}: anchors not found exactly once:\n  " + "\n  ".join(bad))
    if check:
        print(f"{path.name}: patch would apply cleanly")
        return
    shutil.copy(path, str(path) + ".bak_pre_v2")
    for o, n in edits:
        src = src.replace(o, n)
    path.write_text(src)
    print(f"{path.name}: patched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()
    apply(Path(a.models), MODELS_EDITS, "StablePhysicsSigmaHead", a.check)
    apply(Path(a.train), TRAIN_EDITS, "PhysicsSigmaLossV2", a.check)
