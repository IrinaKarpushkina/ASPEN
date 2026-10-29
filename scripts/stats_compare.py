"""Статистические сравнения моделей по ГРУППАМ ВОПРОСОВ (модель vs эталон группы), а не «все со всеми».

Два уровня доказательств:
  seeds      - разброс между обучениями. Welch t-тест по значениям метрики на сидах (НЕ парный по умолчанию: сид 0 модели A
               не связан с сидом 0 модели B; --paired-seeds включает парный тест, если сиды реально общие, напр. одинаковый split/инициализация).
  molecules  - разброс по тестовым молекулам: парный бутстрап по молекулам (единица выборки = молекула; ресэмпл общий для A и B).
               Нужны per-molecule .npz из scripts/molecule_level_eval.py (--mol-dir). Отвечает на вопрос «одинакова ли модель на ЭТОМ
               тесте», но НЕ учитывает дисперсию обучения - поэтому в статье приводят оба уровня.
Для каждого сравнения: улучшение относительно эталона (знак учтён: >0 = лучше), 95% ДИ, p, поправка Холма внутри (группа, метрика, уровень),
вердикт с учётом SESOI (наименьший практически значимый эффект; задаётся --sesoi, по умолчанию в DEFAULT_SESOI).

Запуск:
  python -m scripts.stats_compare --dirs <metrics...> --list
  python -m scripts.stats_compare --dirs <metrics...> --groups configs/stats/groups_ablation.yaml --out-csv results/stats.csv
  python -m scripts.stats_compare --dirs <metrics...> --groups ... --mol-dir results/per_mol_all
  python -m scripts.stats_compare --dirs <metrics...> --ref dimenet_pp_enhanced --vs dimenet_pp_enhanced_xtb 'sigma_dn_*'
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._runs import HIGHER_IS_BETTER, by_label, load_runs  # noqa: E402

DEFAULT_SESOI = {"weighted_r2": 1e-3, "weighted_mae": 7e-4, "polar_mae": 3e-4, "emd_raw": 3e-5,
                 "cosine_similarity": 1e-3, "molecular_mae": 5e-3}
MOL_STATS = ["n_atoms", "s_wabs", "s_wsq", "s_sstot", "s_polar", "s_emd", "s_cos"]
MOL_METRICS = ("weighted_r2", "weighted_mae", "polar_mae", "emd_raw", "cosine_similarity")


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, run, m = np.empty_like(p), 0.0, len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def verdict(imp, lo, hi, p_holm, sesoi, alpha):
    """imp, [lo, hi] - улучшение относительно эталона (>0 = лучше) и его 95% ДИ."""
    sig = p_holm < alpha
    if lo >= -sesoi and hi <= sesoi:
        return "значимо, но пренебрежимо мало (|Δ|<SESOI)" if sig else "≈ эквивалентны (|Δ|<SESOI)"
    if sig and lo > sesoi:
        return "ЛУЧШЕ (значимо и > SESOI)"
    if sig and hi < -sesoi:
        return "ХУЖЕ (значимо и > SESOI)"
    if sig:
        return "лучше, но > SESOI не доказано" if imp > 0 else "хуже, но > SESOI не доказано"
    return "не различимы (данных мало)"


def seed_compare(va, vb, sign, paired=False):
    a, b = np.array(list(va.values()), float), np.array(list(vb.values()), float)
    nan = dict(lo=np.nan, hi=np.nan, p=np.nan, na=len(a), nb=len(b), d=np.nan)
    if len(a) < 2 or len(b) < 2:
        return dict(imp=sign * (a.mean() - b.mean()), **nan)
    common = sorted(set(va) & set(vb))
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if paired and len(common) >= 2:
        diff = np.array([va[s] - vb[s] for s in common], float)
        m, se, df = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), len(diff) - 1
    else:
        m = a.mean() - b.mean()
        v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1)) if se > 0 else 1.0
    if se == 0:
        return dict(imp=sign * m, lo=sign * m, hi=sign * m, p=1.0 if m == 0 else 0.0, na=len(a), nb=len(b), d=np.nan)
    t = stats.t.ppf(0.975, df)
    ci = sorted([sign * (m - t * se), sign * (m + t * se)])
    return dict(imp=sign * m, lo=ci[0], hi=ci[1], p=float(2 * stats.t.sf(abs(m / se), df)),
                na=len(a), nb=len(b), d=sign * m / sd if sd > 0 else np.nan)


# ---------------------------------------------------------------- молекулы
def load_mol_stats(mol_dir, prefix):
    """Усредняет суммы по молекулам по всем сидам модели (метрика ансамбля сидов = среднее метрик по сидам приближённо)."""
    arrs = []
    for f in sorted(glob.glob(os.path.join(mol_dir, f"{prefix}_seed*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "s_wabs" not in z.files:
            print(f"[warn] {f}: старый формат (нет атомных сумм), пропущен - пересчитайте molecule_level_eval.py")
            continue
        arrs.append(z)
    if not arrs:
        return None
    ids = [list(z["mol_id"]) for z in arrs]
    common = sorted(set.intersection(*[set(i) for i in ids]))
    pos = {m: k for k, m in enumerate(common)}
    acc = {k: np.zeros(len(common)) for k in MOL_STATS}
    for z, ii in zip(arrs, ids):
        sel = [k for k, m in enumerate(ii) if m in pos]
        tgt = [pos[ii[k]] for k in sel]
        for k in MOL_STATS:
            acc[k][tgt] += np.asarray(z[k], float)[sel] / len(arrs)
    return common, acc, len(arrs)


def mol_metric(S, metric):
    n = S["n_atoms"]
    return {"weighted_mae": lambda: S["s_wabs"] / n, "weighted_r2": lambda: 1.0 - S["s_wsq"] / S["s_sstot"],
            "polar_mae": lambda: S["s_polar"] / n, "emd_raw": lambda: S["s_emd"] / n,
            "cosine_similarity": lambda: S["s_cos"] / n}[metric]()


def mol_compare(A, B, metric, sign, n_boot=4000, seed=0):
    ids_a, acc_a, _ = A
    ids_b, acc_b, _ = B
    common = sorted(set(ids_a) & set(ids_b))
    ia, ib = {m: k for k, m in enumerate(ids_a)}, {m: k for k, m in enumerate(ids_b)}
    ra, rb = np.array([ia[m] for m in common]), np.array([ib[m] for m in common])
    Sa = np.stack([acc_a[k][ra] for k in MOL_STATS], 1)
    Sb = np.stack([acc_b[k][rb] for k in MOL_STATS], 1)
    M = len(common)
    pt = [mol_metric(dict(zip(MOL_STATS, S.sum(0))), metric) for S in (Sa, Sb)]
    rng, diffs, left = np.random.default_rng(seed), [], n_boot
    while left > 0:
        nb = min(500, left)
        cnt = rng.multinomial(M, np.full(M, 1.0 / M), size=nb).astype(np.float64)
        ma = mol_metric(dict(zip(MOL_STATS, (cnt @ Sa).T)), metric)
        mb = mol_metric(dict(zip(MOL_STATS, (cnt @ Sb).T)), metric)
        diffs.append(ma - mb)
        left -= nb
    diffs = sign * np.concatenate(diffs)
    imp = sign * (pt[0] - pt[1])
    se = diffs.std(ddof=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(2 * stats.norm.sf(abs(imp) / se)) if se > 0 else (1.0 if imp == 0 else 0.0)
    return dict(imp=float(imp), lo=float(lo), hi=float(hi), p=p, na=M, nb=M, d=np.nan)


def expand(patterns, labels):
    out = []
    for p in patterns:
        hit = [l for l in labels if fnmatch.fnmatchcase(l, p)]
        if not hit:
            print(f"[warn] нет модели под шаблон '{p}'")
        out += [h for h in hit if h not in out]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--vs", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="+", default=["weighted_r2", "weighted_mae", "polar_mae", "emd_raw"])
    ap.add_argument("--sesoi", nargs="*", default=[], help="например weighted_r2=0.002 polar_mae=0.0005")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--paired-seeds", action="store_true")
    ap.add_argument("--mol-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    runs = load_runs(a.dirs)
    G = by_label(runs)
    labels = sorted(G)
    if a.list or (not a.groups and not a.ref):
        print(f"{'label':36s} {'loss':9s} {'n_params':>10s} {'сидов':>5s} {'мин.эпох':>8s}  prefix")
        for l in labels:
            rr = G[l]
            print(f"{l:36s} {rr[0]['loss']:9s} {rr[0]['n_params']:>10,} {len(rr):>5d} {min(r['epochs'] for r in rr):>8d}  {rr[0]['prefix']}")
        return

    sesoi = dict(DEFAULT_SESOI)
    for s in a.sesoi:
        k, v = s.split("=")
        sesoi[k] = float(v)
    if a.groups:
        import yaml
        spec = yaml.safe_load(open(a.groups))["groups"]
    else:
        spec = [dict(name="quick", ref=a.ref, vs=a.vs or ["*"])]

    rows = []
    for g in spec:
        r_l = expand([g["ref"]], labels)
        if not r_l:
            print(f"\n### {g['name']}: эталон '{g['ref']}' не найден - группа пропущена")
            continue
        ref = r_l[0]
        vs = [l for l in expand(g.get("vs", []), labels) if l != ref]
        if not vs:
            print(f"\n### {g['name']}: нет моделей для сравнения - пропущена")
            continue
        print("\n" + "=" * 120 + f"\n### {g['name']}\n    эталон: {ref} (сидов {len(G[ref])}, n_params {G[ref][0]['n_params']:,})")
        if g.get("note"):
            print(f"    примечание: {g['note']}")
        for metric in a.metrics:
            sign = 1.0 if HIGHER_IS_BETTER.get(metric, False) else -1.0
            delta = sesoi.get(metric, 0.0)
            for level in ["seeds"] + (["molecules"] if a.mol_dir and metric in MOL_METRICS else []):
                res = []
                for l in vs:
                    if level == "seeds":
                        va = {r["seed"]: r["metrics"][metric] for r in G[l] if metric in r["metrics"]}
                        vb = {r["seed"]: r["metrics"][metric] for r in G[ref] if metric in r["metrics"]}
                        if not va or not vb:
                            continue
                        res.append((l, seed_compare(va, vb, sign, a.paired_seeds) if min(len(va), len(vb)) >= a.min_seeds
                                    else dict(imp=sign * (np.mean(list(va.values())) - np.mean(list(vb.values()))),
                                              lo=np.nan, hi=np.nan, p=np.nan, na=len(va), nb=len(vb), d=np.nan)))
                    else:
                        A, B = load_mol_stats(a.mol_dir, G[l][0]["prefix"]), load_mol_stats(a.mol_dir, G[ref][0]["prefix"])
                        if A is None or B is None:
                            print(f"  [molecules] нет .npz для {l if A is None else ref} - пропущено")
                            continue
                        res.append((l, mol_compare(A, B, metric, sign, a.n_boot)))
                if not res:
                    continue
                ok = [(l, r) for l, r in res if not np.isnan(r["p"])]
                ph = dict(zip([l for l, _ in ok], holm([r["p"] for _, r in ok]))) if ok else {}
                print(f"\n  [{level}] {metric}   (Δ>0 = лучше эталона; SESOI={delta:g}; Холм по группе)")
                print(f"  {'модель':34s} {'n':>6s} {'Δ':>10s} {'95% ДИ':>23s} {'p':>8s} {'p_Holm':>8s}  вердикт")
                for l, r in res:
                    n_txt = f"{r['na']}v{r['nb']}"
                    if np.isnan(r["p"]):
                        print(f"  {l:34s} {n_txt:>6s} {r['imp']:+10.5f} {'—':>23s} {'—':>8s} {'—':>8s}  теста нет (сидов < {a.min_seeds})")
                        v, pp = "нет теста", np.nan
                    else:
                        pp = ph[l]
                        v = verdict(r["imp"], r["lo"], r["hi"], pp, delta, a.alpha)
                        print(f"  {l:34s} {n_txt:>6s} {r['imp']:+10.5f} [{r['lo']:+9.5f},{r['hi']:+9.5f}] {r['p']:8.3g} {pp:8.3g}  {v}")
                    rows.append(dict(group=g["name"], ref=ref, model=l, level=level, metric=metric, n_a=r["na"], n_b=r["nb"],
                                     improvement=r["imp"], ci_lo=r["lo"], ci_hi=r["hi"], p=r["p"], p_holm=pp, sesoi=delta,
                                     cohen_d=r.get("d"), verdict=v))
    if a.out_csv and rows:
        os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {a.out_csv}")


if __name__ == "__main__":
    main()
