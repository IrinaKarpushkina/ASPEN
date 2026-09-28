"""Which loss terms actually drive the gradient?  (answers 'is w_moments=0.10 doing anything?')

    python -m scripts.loss_gradient_audit results/physics/checkpoints/dimenet_pp_enhanced_physics_seed0_physics.pt \
        [--set w_mol=0.5 w_cos=0.25 moments_mode=relative]

For each loss component c it reports  w_c * ||d c / d pred||  (L2 norm over the batch) and its share
of the total, at the checkpoint's predictions on a few validation batches. A term with share < ~2%
is practically inert at its current weight. Shares at a converged checkpoint differ from shares at
initialisation - look at the trained one.
"""
import argparse

import torch
from torch_geometric.loader import DataLoader

from scripts.ensemble_eval import ds_class
from src.physics_extension.loss_v2 import PhysicsSigmaLossV2
from src.train_3d_physics import _MODELS_NEEDING_TRIPLETS, build_model, train_profile_scale

WMAP = dict(profile="w_profile", area="w_area", charge="w_charge", moments="w_moments",
            smooth="w_smooth", wasserstein="w_wasserstein", polar="w_polar",
            cos="w_cos", mol="w_mol", mol_cdf="w_mol_cdf")


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]; dc = cfg["data"]; name = cfg["model"]["name"]
    feats = cfg.get("features", "base")
    trip = name in _MODELS_NEEDING_TRIPLETS or name.startswith("sigma_dn")
    kw = dict(cache_dir=dc.get("cache_dir"), compute_triplets=trip, require_provided_coords=True)
    cls = ds_class(feats)
    train_ds, val_ds = cls(dc["train_path"], **kw), cls(dc["val_path"], **kw)
    model, _ = build_model(cfg)
    if hasattr(model, "calibrate"):
        model.calibrate(train_ds)
    model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval()

    lc = {k: v for k, v in cfg.get("loss", {}).items() if k != "name"}
    lc.update({k: parse(v) for k, v in (s.split("=") for s in a.set)})
    bw = torch.tensor(train_ds.bin_weights_numpy(), dtype=torch.float32, device=dev)
    crit = PhysicsSigmaLossV2(bw, train_profile_scale(train_ds), **lc).to(dev)

    acc = {k: 0.0 for k in WMAP}
    for i, d in enumerate(DataLoader(val_ds, batch_size=24, shuffle=True)):
        if i >= a.batches:
            break
        d = d.to(dev)
        with torch.no_grad():
            pred = model(d).float()
        pred.requires_grad_(True)
        comps = crit.components(pred, d.y.float(), d.batch)
        for k, wk in WMAP.items():
            w = getattr(crit, wk)
            if w == 0 or not comps[k].requires_grad:
                continue
            g, = torch.autograd.grad(comps[k], pred, retain_graph=True, allow_unused=True)
            if g is not None:
                acc[k] += float(w * g.norm()) / a.batches
    tot = sum(acc.values()) or 1.0
    print(f"{'term':12s} {'w*|grad|':>12s} {'share':>8s}")
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"{k:12s} {v:12.4e} {v/tot:8.1%}")


if __name__ == "__main__":
    main()
