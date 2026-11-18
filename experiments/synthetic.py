"""Controlled study: correction gain follows the Bayes gap, not the error (Figure 3).

    python -m experiments.synthetic             # 10 seeds x 12 points
    python -m experiments.synthetic --seeds 2   # quick check

Bayes target pi(X) = softmax(W X / tau); frozen model m(X) = softmax(W X / tau + g V X).
tau sets the label ambiguity and g the model bias. For each tau on a grid, g is
tuned so that base accuracy stays at 0.65. Along the sweep the error stays fixed
while the Bayes gap E||m - pi||^2 shrinks to zero.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from guard.losses import CROSS_ENTROPY as CE
from guard.policy import select_beta
from guard.retrieval import hard_label_values, knn_average

D, C = 10, 4
N_SOURCE, N_FIT, N_TEST = 4000, 600, 4000
TARGET_ACC, K = 0.65, 50


def softmax(u):
    u = u - u.max(1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(1, keepdims=True)


def sample(rng, n, W, V, tau, g):
    """Inputs, labels, Bayes target and frozen-model output."""
    X = rng.standard_normal((n, D))
    z = X @ W / tau
    pi = softmax(z)
    y = np.array([rng.choice(C, p=pi[i]) for i in range(n)])
    return X, y, pi, softmax(z + g * (X @ V))


def base_accuracy(W, V, tau, g, n=6000):
    _, y, _, m = sample(np.random.default_rng(7), n, W, V, tau, g)
    return (m.argmax(1) == y).mean()


def bisect(lo, hi, f, iters=18):
    """Find x with f(x) = TARGET_ACC, for f decreasing in x."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > TARGET_ACC else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        W = rng.standard_normal((D, C))
        V = rng.standard_normal((D, C)) / np.sqrt(D)
        # largest tau: an unbiased model (g = 0) already has the target accuracy
        tau_max = bisect(0.2, 30.0, lambda t: base_accuracy(W, V, t, 0.0))
        for tau in np.linspace(0.6, tau_max, a.points):
            g = bisect(0.0, 8.0, lambda gg: base_accuracy(W, V, tau, gg))
            Xs, ys, _, _ = sample(rng, N_SOURCE, W, V, tau, g)
            Xf, yf, _, mf = sample(rng, N_FIT, W, V, tau, g)
            Xt, yt, pit, mt = sample(rng, N_TEST, W, V, tau, g)

            values = hard_label_values(ys, C, True)
            beta = select_beta(mf, knn_average(Xf, Xs, values, K), yf, CE)
            corrected = (1 - beta) * mt + beta * knn_average(Xt, Xs, values, K)
            acc = float((mt.argmax(1) == yt).mean())
            rows.append(dict(seed=seed, tau=float(tau), g=float(g), base_acc=acc,
                             bayes_gap=float(((mt - pit) ** 2).sum(1).mean()),
                             beta=beta,
                             loss_gain=float((CE(mt, yt) - CE(corrected, yt)).mean()),
                             acc_gain=float((corrected.argmax(1) == yt).mean() - acc)))
        print(f"seed {seed} done", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "synthetic.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Bayes gap':>10} {'base acc':>9} {'beta':>6} {'acc gain':>9} {'loss gain':>10}")
    for i in range(a.points):
        pts = rows[i::a.points]  # the same grid point across seeds
        m = {k: np.mean([r[k] for r in pts]) for k in ("bayes_gap", "base_acc", "beta", "acc_gain", "loss_gain")}
        print(f"{m['bayes_gap']:10.4f} {m['base_acc']:9.3f} {m['beta']:6.2f} "
              f"{m['acc_gain']:+9.4f} {m['loss_gain']:+10.4f}")
    gap = [r["bayes_gap"] for r in rows]
    gain = [r["acc_gain"] for r in rows]
    print(f"\ncorrelation(Bayes gap, accuracy gain) over runs: {np.corrcoef(gap, gain)[0, 1]:.3f}")
    print(f"rows written to {path}")


if __name__ == "__main__":
    main()
