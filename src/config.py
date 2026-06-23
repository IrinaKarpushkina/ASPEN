"""
config.py — загрузка и слияние YAML-конфигов.

Использование:
    cfg = load_config("configs/gatv2.yaml")
    # cfg содержит объединённые base.yaml + gatv2.yaml (model-specific
    # ключи перекрывают base, остальное наследуется)

Конфиг должен содержать ключ `defaults: base.yaml` (путь относительно
самого файла) — это единственный механизм наследования, без скрытой магии.
"""
from __future__ import annotations
import os
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "defaults":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg["defaults"])
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg
