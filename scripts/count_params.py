"""
count_params.py — фактическое число параметров каждой архитектуры.

Запуск:
    python -m scripts.count_params                           # 3D конфиги
    python -m scripts.count_params --configs-dir configs/2d # 2D конфиги
    python -m scripts.count_params --auto-tune               # + подбор hidden
    python -m scripts.count_params --csv results/param_budget_table.csv

Результат идёт в Supplementary как "Table S1: Model parameter budgets" —
обязательное требование fair-comparison (Dwivedi et al., JMLR 2022).
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


def count_params(model_cfg: dict, use_extended: bool, mode: str = "3d") -> int:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    cfg.pop("n_params_target", None)
    model_cls = MODEL_REGISTRY[name]
    sig = inspect.signature(model_cls.__init__)
    if "mode" in sig.parameters:
        cfg["mode"] = mode
    model = model_cls(use_extended=use_extended, **cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _width_key(model_cfg: dict) -> str:
    return "hidden_channels" if "hidden_channels" in model_cfg else "hidden"


def auto_tune_hidden(
    model_cfg: dict, use_extended: bool, target: int,
    step: int = 1, lo: int = 32, hi: int = 1024, mode: str = "3d",
) -> int:
    width_key = _width_key(model_cfg)
    best_hidden, best_diff = None, None
    h = lo
    while h <= hi:
        if h % step != 0:
            h += 1
            continue
        cfg = dict(model_cfg)
        cfg[width_key] = h
        if width_key == "hidden_channels":
            cfg["num_filters"] = h
        try:
            n = count_params(cfg, use_extended, mode=mode)
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
    parser.add_argument(
        "--configs-dir", default="configs",
        help="Папка с конфигами. Используйте 'configs/2d' для 2D-бенчмарка.",
    )
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    # Ищем base.yaml: сначала в самой папке, иначе поднимаемся выше.
    base_yaml = os.path.join(args.configs_dir, "base.yaml")
    if not os.path.exists(base_yaml):
        base_yaml = os.path.join(os.path.dirname(args.configs_dir.rstrip("/")), "base.yaml")
    base_cfg = load_config(base_yaml)
    target = base_cfg["target_params"]
    use_extended = base_cfg["data"]["use_extended"]

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
        cfg_mode = cfg.get("data", {}).get("mode", "3d")

        n_params = count_params(model_cfg, use_extended, mode=cfg_mode)
        diff_pct = (n_params - target) / target * 100
        width_key = _width_key(model_cfg)
        width_val = model_cfg.get(width_key)

        suggestion = ""
        if args.auto_tune:
            step = model_cfg.get("heads", 1)
            best_hidden = auto_tune_hidden(
                model_cfg, use_extended, target, step=step, mode=cfg_mode,
            )
            if best_hidden != width_val:
                tuned_cfg = dict(model_cfg)
                tuned_cfg[width_key] = best_hidden
                if width_key == "hidden_channels":
                    tuned_cfg["num_filters"] = best_hidden
                tuned_n = count_params(tuned_cfg, use_extended, mode=cfg_mode)
                suggestion = (
                    f"  -> suggest {width_key}={best_hidden} "
                    f"({tuned_n:,} params, {(tuned_n - target) / target * 100:+.2f}%)"
                )

        print(
            f"{name:12s} {width_key}={width_val:4d}  "
            f"params={n_params:>9,}  diff={diff_pct:+6.2f}%  [{cfg_mode}]{suggestion}"
        )
        rows.append({
            "model": name,
            "mode": cfg_mode,
            "width_param": width_key,
            "width_value": width_val,
            "n_params": n_params,
            "target": target,
            "diff_pct": round(diff_pct, 2),
        })

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fieldnames = ["model", "mode", "width_param", "width_value",
                      "n_params", "target", "diff_pct"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {args.csv}")


if __name__ == "__main__":
    main()
