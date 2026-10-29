"""Общий загрузчик прогонов для статистических скриптов.

Метка прогона (label) = model из json + "@<размер>" из ИМЕНИ ФАЙЛА (0.5x / 2x / 4x; без метки = 1x), потому что конфиги
scaling/*.yaml имеют то же model.name, что и базовые. Если одна метка встречается с разными loss -> "/<loss>";
если внутри метки разные n_params -> "[<n>k]". Дубликаты (label, seed) из нескольких папок: берётся первый по порядку
--dirs. Поле prefix = имя файла до "_seed" (по нему stats_compare находит per-molecule .npz).
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

VARIANT_RE = re.compile(r"_(\d+(?:\.\d+)?x)_seed\d+")
SEED_RE = re.compile(r"_seed(\d+)_(.+)$")

HIGHER_IS_BETTER = {
    "weighted_r2": True, "cosine_similarity": True, "molecular_cosine": True,
    "weighted_mae": False, "polar_mae": False, "emd_raw": False, "emd_normalized": False,
    "molecular_mae": False, "molecular_emd": False, "molecular_polar_mae": False,
}


def parse_stem(stem: str):
    m = SEED_RE.search(stem)
    if not m:
        return stem, "1x", None, "unknown"
    v = VARIANT_RE.search(stem)
    return stem[: m.start()], (v.group(1) if v else "1x"), int(m.group(1)), m.group(2)


def load_runs(dirs, verbose=True):
    runs = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            prefix, variant, seed_f, loss = parse_stem(stem)
            try:
                r = json.load(open(path))
                tm = r["test_metrics"]
            except Exception as e:
                if verbose:
                    print(f"[skip] {path}: {e}")
                continue
            runs.append(dict(model=r.get("model", prefix), variant=variant, loss=loss, seed=r.get("seed", seed_f),
                             prefix=prefix, n_params=r.get("n_params") or 0, epochs=len(r.get("history", [])),
                             metrics=tm, resources=r.get("resources", {}), path=path, json=r))
    by_ml, by_np = defaultdict(set), defaultdict(set)
    for r in runs:
        by_ml[(r["model"], r["variant"])].add(r["loss"])
        by_np[(r["model"], r["variant"], r["loss"])].add(r["n_params"])
    out, seen, dups = [], {}, 0
    for r in runs:
        lab = r["model"] + ("" if r["variant"] == "1x" else "@" + r["variant"])
        if len(by_ml[(r["model"], r["variant"])]) > 1:
            lab += "/" + r["loss"]
        if len(by_np[(r["model"], r["variant"], r["loss"])]) > 1:
            lab += f"[{r['n_params'] // 1000}k]"
        r["label"] = lab
        if (lab, r["seed"]) in seen:
            dups += 1
            continue
        seen[(lab, r["seed"])] = r["path"]
        out.append(r)
    if verbose and dups:
        print(f"[warn] дубликаты (label, seed): пропущено {dups}; использован первый по порядку --dirs")
    return out


def by_label(runs):
    d = defaultdict(list)
    for r in runs:
        d[r["label"]].append(r)
    return d
