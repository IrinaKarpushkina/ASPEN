"""
result_loader.py -- единая загрузка результатов прогонов из ЛЮБЫХ папок и с ЛЮБЫМ loss.

Что считается результатом прогона: любой *.json (рекурсивно), в котором есть
ключ "test_metrics". Имя файла НЕ фильтруется (никаких *_mse.json).

Откуда берутся поля:
    model   <- JSON["model"], иначе из имени файла (часть до '_seed')
    seed    <- JSON["seed"],  иначе из имени файла (число после '_seed')
    loss    <- JSON["loss"],  иначе суффикс имени файла (после последнего '_'),
               иначе 'unknown'
    mode, n_params, best_epoch, n_epochs_run -- из JSON, если есть
    метрики <- JSON["test_metrics"] (все числовые поля)

Дубликаты (одинаковые model+loss+seed в разных папках, например при
пересчёте): --on-duplicate last (по умолчанию; побеждает папка, указанная
ПОЗЖЕ в командной строке) | first | error. Всегда печатается список конфликтов.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import pandas as pd

_STEM_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)(?:_(?P<loss>.+))?$")


def _expand(d: str) -> list[str]:
    """Папка или glob-шаблон -> список json-путей (рекурсивно)."""
    if os.path.isdir(d):
        return sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
    out = []
    for m in sorted(glob.glob(d, recursive=True)):
        if os.path.isdir(m):
            out += sorted(glob.glob(os.path.join(m, "**", "*.json"), recursive=True))
        elif m.endswith(".json"):
            out.append(m)
    return out


def load_runs(dirs, on_duplicate: str = "last", verbose: bool = True) -> pd.DataFrame:
    """Возвращает DataFrame: по строке на (model, loss, seed).

    Колонки: model, loss, seed, mode, n_params, best_epoch, n_epochs_run,
    source (папка), path, label (= model@loss) + все метрики из test_metrics.
    """
    if isinstance(dirs, str):
        dirs = [dirs]
    rows, n_skipped = [], 0
    for order, d in enumerate(dirs):
        paths = _expand(d)
        if not paths and verbose:
            print(f"WARNING: nothing found for '{d}'", file=sys.stderr)
        for p in paths:
            try:
                with open(p) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                n_skipped += 1
                continue
            if not isinstance(r, dict) or not isinstance(r.get("test_metrics"), dict):
                n_skipped += 1  # не файл результатов (конфиг, кривая обучения и т.п.)
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            m = _STEM_RE.match(stem)
            model = r.get("model") or (m.group("model") if m else stem)
            seed = r.get("seed", int(m.group("seed")) if m else None)
            loss = r.get("loss") or ((m.group("loss") if m else None)
                                     or (stem.rsplit("_", 1)[-1] if "_" in stem else "unknown"))
            row = dict(
                model=model, loss=str(loss), seed=seed, mode=r.get("mode", ""),
                n_params=r.get("n_params"), best_epoch=r.get("best_epoch"),
                n_epochs_run=r.get("resources", {}).get("n_epochs_run", len(r.get("history", []) or [])),
                source=str(d), path=p, _order=order,
            )
            # ключи метрик, совпадающие со служебными колонками (напр. test_metrics["loss"]
            # -- это число, а не метка loss), переименовываем в test_<ключ>
            for k, v in r["test_metrics"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[f"test_{k}" if k in row else k] = v
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result JSONs (with 'test_metrics') found in: {list(dirs)}")

    df = pd.DataFrame(rows)
    df["label"] = df["model"] + "@" + df["loss"]

    key = ["model", "loss", "seed"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        if verbose:
            print(f"WARNING: {df[dup].groupby(key).ngroups} duplicated (model, loss, seed) "
                  f"found in several files; policy = '{on_duplicate}':", file=sys.stderr)
            for k, g in df[dup].groupby(key):
                print(f"   {k}: " + " | ".join(g["path"]), file=sys.stderr)
        if on_duplicate == "error":
            raise ValueError("Duplicate runs found; rerun with --on-duplicate first|last")
        df = df.sort_values("_order", kind="stable")
        df = df.drop_duplicates(key, keep="first" if on_duplicate == "first" else "last")
    if verbose:
        print(f"Loaded {len(df)} runs from {len(dirs)} location(s) "
              f"({n_skipped} non-result json skipped).", file=sys.stderr)
    return df.drop(columns="_order").reset_index(drop=True)


def add_source_args(parser):
    """Общие аргументы для всех скриптов."""
    parser.add_argument("--metrics-dir", nargs="+", required=True,
                        help="Одна или несколько папок/glob-шаблонов с json результатов "
                             "(рекурсивно, любые имена файлов, любой loss).")
    parser.add_argument("--on-duplicate", choices=["last", "first", "error"], default="last",
                        help="Что делать, если один и тот же (model, loss, seed) найден в "
                             "нескольких папках. last = берётся из папки, указанной позже.")
    parser.add_argument("--loss", nargs="*", default=None,
                        help="Необязательный фильтр по loss (можно несколько).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Необязательный фильтр: имя модели (любой loss) или model@loss.")


def apply_filters(df: pd.DataFrame, losses=None, models=None) -> pd.DataFrame:
    if losses:
        df = df[df["loss"].isin(losses)]
    if models:
        df = df[df["label"].isin(models) | df["model"].isin(models)]
    if df.empty:
        raise FileNotFoundError("No runs left after --loss/--models filters")
    return df.reset_index(drop=True)
