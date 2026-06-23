"""
config.py — загрузка и слияние YAML-конфигов.

Поддерживает ДВУХУРОВНЕВОЕ наследование через defaults:

    configs/2d/gcn.yaml     → defaults: ../base_2d.yaml
    configs/base_2d.yaml    → defaults: base.yaml
    configs/base.yaml       (корень, нет defaults)

Цепочка разрешается рекурсивно: каждый уровень полностью
сливается перед следующим, child-ключи всегда перекрывают parent.
Старый однуровневый синтаксис полностью совместим.
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
    """Загружает конфиг с рекурсивным разрешением defaults-цепочки."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg["defaults"])
        # Рекурсия: base_path сам может иметь defaults
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg
