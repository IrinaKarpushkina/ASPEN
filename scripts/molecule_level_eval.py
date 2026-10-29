"""molecule_level_eval.py - по-молекулярные метрики на test для одного чекпоинта (вход для парных тестов по молекулам).

Изменения относительно прежней версии:
  * работает для ВСЕХ моделей: 2D, базовые 3D и physics/xtb/surface (scripts/_model_factory.py);
  * помимо молекулярных mae/polar_mae/cosine/emd сохраняет СУММЫ ПО АТОМАМ каждой молекулы (n_atoms, s_wabs, s_wsq, s_sstot,
    s_polar, s_emd, s_cos): из них точно восстанавливаются weighted_r2, weighted_mae, polar_mae, emd_raw, cosine_similarity
    (stats_compare.py --mol-dir делает по ним парный бутстрап по молекулам);
  * как в src/evaluate.py: autocast по use_amp конфига, preds/targets обрезаются снизу нулём, веса бинов из train-датасета;
  * --check-json сверяет восстановленные метрики с test_metrics из json обучения.

Пример:
  python -m scripts.molecule_level_eval --config configs/3d/dimenet_pp_enhanced_xtb.yaml \
     --checkpoint results/xtb/checkpoints/dimenet_pp_enhanced_xtb_seed0_mse.pt --mode 3d \
     --out results/per_mol_all/dimenet_pp_enhanced_xtb_seed0.npz --check-json results/xtb/metrics/dimenet_pp_enhanced_xtb_seed0_mse.json
Имя выхода: <prefix>_seed<k>.npz, prefix = имя metrics-json до "_seed" (так их находит stats_compare). Пакетно: scripts/molecule_eval_batch.py.
"""
import argparse
import json
import logging
import os

import numpy as np
from scipy.stats import wasserstein_distance

from src.metrics import EPS, POLAR_MASK, SIGMA_BINS, _cosine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _emd(t, p):
    try:
        return wasserstein_distance(SIGMA_BINS, SIGMA_BINS, np.maximum(t, 0.0) + EPS, np.maximum(p, 0.0) + EPS)
    except Exception:
        return float("nan")


def per_molecule_stats(t, p, sizes, mol_ids, w):
    """t, p: (N_atoms, 51), уже обрезанные снизу нулём. w: веса бинов (сумма 1). Возвращает dict массивов по молекулам."""
    w = w / (w.sum() + EPS)
    abs_a = (np.abs(t - p) * w[None]).sum(1)
    sq_a = (((t - p) ** 2) * w[None]).sum(1)
    mt = (t * w[None]).sum(1, keepdims=True)
    tot_a = (((t - mt) ** 2) * w[None]).sum(1)
    pol_a = np.abs(t[:, POLAR_MASK] - p[:, POLAR_MASK]).mean(1)
    emd_a = np.array([_emd(a, b) for a, b in zip(t, p)])
    cos_a = np.array([_cosine(a, b) for a, b in zip(t, p)])
    off = np.concatenate([[0], np.cumsum(sizes)])
    out = {k: [] for k in ("n_atoms", "s_wabs", "s_wsq", "s_sstot", "s_polar", "s_emd", "s_cos",
                          "mae", "polar_mae", "cosine", "emd")}
    for i in range(len(sizes)):
        a, b = off[i], off[i + 1]
        out["n_atoms"].append(b - a)
        out["s_wabs"].append(abs_a[a:b].sum()); out["s_wsq"].append(sq_a[a:b].sum()); out["s_sstot"].append(tot_a[a:b].sum())
        out["s_polar"].append(pol_a[a:b].sum()); out["s_emd"].append(np.nansum(emd_a[a:b])); out["s_cos"].append(cos_a[a:b].sum())
        tm, pm = t[a:b].sum(0), p[a:b].sum(0)
        out["mae"].append(float(np.mean(np.abs(tm - pm))))
        out["polar_mae"].append(float(np.mean(np.abs(tm[POLAR_MASK] - pm[POLAR_MASK]))))
        out["cosine"].append(_cosine(tm / (np.linalg.norm(tm) + EPS), pm / (np.linalg.norm(pm) + EPS)))
        out["emd"].append(_emd(tm, pm))
    return {k: np.array(v) for k, v in out.items()}


def run(config, checkpoint, mode, split, out, batch_size=32, check_json=None):
    import torch
    from torch_geometric.loader import DataLoader
    from src.config import load_config
    from scripts._model_factory import build_for_eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config)
    use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
    model, DatasetCls, kw = build_for_eval(cfg, mode)
    dataset = DatasetCls(cfg["data"][f"{split}_path"], **kw)
    train_ds = DatasetCls(cfg["data"]["train_path"], **kw)
    w = train_ds.bin_weights_numpy().astype(np.float64)
    del train_ds
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    preds, targets, sizes, ids, mi = [], [], [], [], 0
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            nm = int(data.batch.max().item()) + 1
            ids.extend(dataset.mol_ids[mi: mi + nm]); mi += nm
            data = data.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pr = model(data)
            preds.append(pr.float().cpu().numpy()); targets.append(data.y.float().cpu().numpy())
            sizes.extend(torch.bincount(data.batch.cpu()).numpy().tolist())
    assert len(sizes) == len(ids) == len(dataset), "рассогласование mol_id (нужен shuffle=False)"
    p = np.maximum(np.concatenate(preds), 0.0)
    t = np.maximum(np.concatenate(targets), 0.0)
    res = per_molecule_stats(t, p, sizes, ids, w)
    N = res["n_atoms"].sum()
    rec = {"weighted_r2": 1 - res["s_wsq"].sum() / res["s_sstot"].sum(), "weighted_mae": res["s_wabs"].sum() / N,
           "polar_mae": res["s_polar"].sum() / N, "emd_raw": res["s_emd"].sum() / N, "cosine_similarity": res["s_cos"].sum() / N}
    logger.info("восстановлено из сумм: " + ", ".join(f"{k}={v:.5f}" for k, v in rec.items()))
    if check_json:
        ref = json.load(open(check_json))["test_metrics"]
        for k, v in rec.items():
            logger.info(f"  сверка {k}: json={ref[k]:.5f} здесь={v:.5f} Δ={v - ref[k]:+.2e}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, mol_id=np.array(ids, dtype=object), **res)
    logger.info(f"сохранено {len(ids)} молекул -> {out}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["2d", "3d"], default="3d")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--check-json", default=None)
    a = ap.parse_args()
    run(a.config, a.checkpoint, a.mode, a.split, a.out, a.batch_size, a.check_json)


if __name__ == "__main__":
    main()
