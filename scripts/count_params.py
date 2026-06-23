"""
count_params.py — печатает фактическое число параметров каждой архитектуры
и (опционально) подбирает hidden, чтобы попасть в целевой бюджет (±5%).

Запуск:
    python -m scripts.count_params                  # отчёт по текущим configs/*.yaml
    python -m scripts.count_params --auto-tune       # + подбор hidden, печатает
                                                       # рекомендуемые значения
    python -m scripts.count_params --csv results/param_budget_table.csv

Результат идёт в Supplementary как "Table S1: Model parameter budgets" —
обязательное требование fair-comparison (Dwivedi et al., JMLR 2022).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.models import MODEL_REGISTRY


def count_params(model_cfg: dict, use_extended: bool) -> int:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    cfg.pop("n_params_target", None)
    model = MODEL_REGISTRY[name](use_extended=use_extended, **cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _width_key(model_cfg: dict) -> str:
    """Returns the name of the 'width' hyperparameter for this architecture."""
    return "hidden_channels" if "hidden_channels" in model_cfg else "hidden"


def auto_tune_hidden(model_cfg: dict, use_extended: bool, target: int,
                      step: int = 1, lo: int = 32, hi: int = 1024) -> int:
    """
    Бинарный поиск ширины (hidden / hidden_channels), минимизирующий
    |n_params - target|. `step` — допустимая делимость (heads для
    GAT/GATv2, обычно 8; 1 для остальных).
    """
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
            cfg["num_filters"] = h  # keep SchNet's filters == hidden_channels
        try:
            n = count_params(cfg, use_extended)
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
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    base_cfg = load_config(os.path.join(args.configs_dir, "base.yaml"))
    target = base_cfg["target_params"]
    use_extended = base_cfg["data"]["use_extended"]

    rows = []
    for path in sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml"))):
        if os.path.basename(path) == "base.yaml":
            continue
        cfg = load_config(path)
        model_cfg = cfg["model"]
        name = model_cfg["name"]

        n_params = count_params(model_cfg, use_extended)
        diff_pct = (n_params - target) / target * 100
        width_key = _width_key(model_cfg)
        width_val = model_cfg.get(width_key)

        suggestion = ""
        if args.auto_tune:
            step = model_cfg.get("heads", 1)
            best_hidden = auto_tune_hidden(model_cfg, use_extended, target, step=step)
            if best_hidden != width_val:
                tuned_cfg = dict(model_cfg)
                tuned_cfg[width_key] = best_hidden
                if width_key == "hidden_channels":
                    tuned_cfg["num_filters"] = best_hidden
                tuned_n = count_params(tuned_cfg, use_extended)
                suggestion = (
                    f"  -> suggest {width_key}={best_hidden} "
                    f"({tuned_n:,} params, {(tuned_n - target) / target * 100:+.2f}%)"
                )

        print(
            f"{name:8s} {width_key}={width_val:4d}  "
            f"params={n_params:>9,}  diff={diff_pct:+6.2f}%{suggestion}"
        )
        rows.append({
            "model": name,
            "width_param": width_key,
            "width_value": width_val,
            "n_params": n_params,
            "target": target,
            "diff_pct": round(diff_pct, 2),
        })

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        fieldnames = ["model", "width_param", "width_value", "n_params", "target", "diff_pct"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {args.csv}")


if __name__ == "__main__":
    main()
