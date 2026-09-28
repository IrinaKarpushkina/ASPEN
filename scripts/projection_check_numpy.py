"""numpy-only reproduction of the moment-projection check (no torch needed).

    python scripts/projection_check_numpy.py

Compares the ORIGINAL projection (8 undamped Newton steps, lambda clamp +-25) with bisection on
(a) an element-like peaked prior with log-floor 1e-12, and (b) peaked learned-like logits.
Synthetic data: it shows the mechanism, it does not prove what happened in your training run.
"""
import numpy as np

s = np.linspace(-1, 1, 51)
rng = np.random.default_rng(0)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def newton(logits, target, n=8):
    lam = np.zeros(len(target))
    for _ in range(n):
        p = softmax(logits + lam[:, None] * s)
        m = (p * s).sum(-1)
        v = (p * (s - m[:, None]) ** 2).sum(-1)
        lam = np.clip(lam + (target - m) / (v + 1e-4), -25, 25)
    return softmax(logits + lam[:, None] * s)


def bisect(logits, target, n=32, L=200.0):
    lo, hi = np.full(len(target), -L), np.full(len(target), L)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        m = (softmax(logits + mid[:, None] * s) * s).sum(-1)
        up = m < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return softmax(logits + (0.5 * (lo + hi))[:, None] * s)


def report(tag, logits, target):
    for name, fn in (("newton x8", newton), ("bisection", bisect)):
        err = np.abs((fn(logits, target) * s).sum(-1) - target)
        print(f"{tag:34s} {name:10s} first-moment error: median {np.median(err):.2e}  p95 {np.percentile(err, 95):.2e}")


y = np.zeros(51); y[15:36] = np.exp(-0.5 * ((np.arange(15, 36) - 24) / 3.0) ** 2)
prior = np.log(np.maximum(y / y.sum(), 1e-12))
report("peaked prior, floor 1e-12", prior[None] + 0.3 * rng.normal(size=(200, 51)), rng.uniform(-0.9, 0.9, 200))
for w in (8, 4, 2):
    c = rng.uniform(10, 40, 300)
    lg = -2.0 * ((np.arange(51)[None] - c[:, None]) / w) ** 2 + 0.2 * rng.normal(size=(300, 51))
    tgt = np.clip((c - 25) / 25 + rng.uniform(-0.25, 0.25, 300), -0.95, 0.95)
    report(f"learned-like bump, width {w} bins", lg, tgt)
