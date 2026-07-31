"""
count_params.py — фактическое число параметров каждой архитектуры и
(опционально) авто-подбор hidden-размера под целевой бюджет параметров.

Запуск:
    python -m scripts.count_params
    python -m scripts.count_params --auto-tune
    python -m scripts.count_params --csv results/param_budget_table.csv


# Для 2D
python scripts/count_params.py --configs-dir configs/2d

# Для 3D
python scripts/count_params.py --configs-dir configs/3d

# С авто-подбором
python scripts/count_params.py --configs-dir configs/2d --auto-tune


Матчинг числа параметров между архитектурами — стандартное требование
"controlled comparison" (Dwivedi et al., JMLR 2022, "Benchmarking Graph
Neural Networks": сравнивать архитектуры нужно при примерно равном
бюджете параметров, иначе разница в метриках может объясняться просто
тем, что у одной модели больше весов). Результат этого скрипта стоит
включать в Supplementary как "Table S1: Model parameter budgets".

ВАЖНО после рефакторинга (см. PROVENANCE.md): удаление общего
global_proj/global_norm блока у GCN/GAT/GATv2/GINE/DMPNN заметно изменило
число параметров при том же hidden — почти наверняка нужно перезапустить
--auto-tune и обновить hidden в configs/*.yaml перед финальными прогонами.
"""
from __future__ import annotations
import argparse
import csv
import glob
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.models import MODEL_REGISTRY


def count_params(model_cfg: dict) -> int:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    cfg.pop("n_params_target", None)
    model_cls = MODEL_REGISTRY[name]
    sig = inspect.signature(model_cls.__init__)
    cfg = {k: v for k, v in cfg.items() if k in sig.parameters}
    model = model_cls(**cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _width_key(model_cfg: dict) -> str:
    return "hidden_channels" if "hidden_channels" in model_cfg else "hidden"


def auto_tune_hidden(model_cfg: dict, target: int, step: int = 1, lo: int = 32, hi: int = 1024) -> int:
    width_key = _width_key(model_cfg)
    best_hidden, best_diff = None, None
    h = lo
    while h <= hi:
        if h % step != 0:
            h += 1
            continue
        cfg = dict(model_cfg)
        cfg[width_key] = h
        try:
            n = count_params(cfg)
        except Exception:
            h += step
            continue
        diff = abs(n - target)
        if best_diff is None or diff < best_diff:
            best_diff, best_hidden = diff, h
        h += step
    return best_hidden


def main():
    parser = argparse.ArgumentParser()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_configs = os.path.normpath(os.path.join(_script_dir, "..", "configs", "2d"))
    parser.add_argument("--configs-dir", default=_default_configs)
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    base_cfg = load_config(os.path.join(args.configs_dir, "base.yaml"))
    target = base_cfg["target_params"]

    rows = []
    for path in sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml"))):
        bname = os.path.basename(path)
        if bname.startswith("base"):
            continue
        cfg = load_config(path)
        if "model" not in cfg:
            continue
        model_cfg = cfg["model"]
        name = model_cfg["name"]

        n_params = count_params(model_cfg)
        diff_pct = (n_params - target) / target * 100
        width_key = _width_key(model_cfg)
        width_val = model_cfg.get(width_key)

        suggestion = ""
        if args.auto_tune:
            step = model_cfg.get("heads", 1)
            best_hidden = auto_tune_hidden(model_cfg, target, step=step)
            if best_hidden != width_val:
                tuned_cfg = dict(model_cfg)
                tuned_cfg[width_key] = best_hidden
                tuned_n = count_params(tuned_cfg)
                suggestion = (
                    f"  -> suggest {width_key}={best_hidden} "
                    f"({tuned_n:,} params, {(tuned_n - target) / target * 100:+.2f}%)"
                )

        print(
            f"{name:14s} {width_key}={width_val:4d}  "
            f"params={n_params:>9,}  diff={diff_pct:+6.2f}%{suggestion}"
        )
        rows.append({
            "model": name, "width_param": width_key, "width_value": width_val,
            "n_params": n_params, "target": target, "diff_pct": round(diff_pct, 2),
        })

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fieldnames = ["model", "width_param", "width_value", "n_params", "target", "diff_pct"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {args.csv}")


if __name__ == "__main__":
    main()
