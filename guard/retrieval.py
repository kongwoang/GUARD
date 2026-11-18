"""Recalibrate: estimate the best prediction from the observed inputs by retrieval.

Find the k nearest pool rows, then average either their labels (hard-label
target) or the model's outputs on them under a richer mask (cross-mask target,
no labels needed).
"""

from __future__ import annotations

import numpy as np


def retrieval_space(reference: np.ndarray, kind: str = "standardise"):
    """Feature map fitted on the pool: z-score (``standardise``), or z-score
    and unit length (``cosine``)."""
    mu = reference.mean(0)
    sd = reference.std(0) + 1e-9

    def z(x):
        return (np.asarray(x, dtype=np.float64) - mu) / sd

    if kind == "standardise":
        return z
    if kind == "cosine":
        def unit(x):
            v = z(x)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return unit
    raise ValueError(f"unknown retrieval space {kind!r}")


def knn_average(query, pool, values, k, weighting="uniform", chunk=1024):
    """Average ``values`` over the k nearest pool rows of each query.

    With ``distance`` weighting, closer neighbours count more. The kernel width
    comes from the pool only, so a query's target does not depend on other queries.
    """
    if len(pool) < k:
        raise ValueError(f"pool has {len(pool)} rows but k={k}")
    pool_sq = (pool ** 2).sum(1)
    tau = _pool_scale(pool, pool_sq, k) if weighting == "distance" else None
    out = np.empty((len(query), values.shape[1]), dtype=np.float64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ pool.T)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        if weighting == "uniform":
            out[i:i + chunk] = values[idx].mean(1)
        elif weighting == "distance":
            d = np.sqrt(np.maximum(np.take_along_axis(d2, idx, 1), 0.0))
            w = np.exp(-d / tau)
            dead = w.sum(1) == 0  # a query far from the whole pool underflows
            if dead.any():
                w[dead] = np.exp(-(d[dead] - d[dead].min(1, keepdims=True)) / tau)
            w /= w.sum(1, keepdims=True)
            out[i:i + chunk] = (values[idx] * w[:, :, None]).sum(1)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
    return out


def _pool_scale(pool, pool_sq, k, sample=2048, seed=0):
    n = len(pool)
    idx = np.arange(n) if n <= sample else np.random.default_rng(seed).choice(n, sample, replace=False)
    q = pool[idx]
    d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ pool.T)
    kk = min(k, n - 1)
    part = np.partition(d2, kk, axis=1)[:, :kk + 1]  # includes the self match
    return float(np.median(np.sqrt(np.maximum(part.max(1), 0.0)))) + 1e-12


def temper(probs: np.ndarray, temperature: float, simplex: bool = True) -> np.ndarray:
    """Temperature scaling, per row for classes and per label for multi-label outputs."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    if simplex:
        logits = np.log(p) / float(temperature)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1.0 - p)) / float(temperature)))


def hard_label_values(pool_labels: np.ndarray, n_out: int, simplex: bool) -> np.ndarray:
    """t(y) for every pool row."""
    if simplex:
        return np.eye(n_out)[np.asarray(pool_labels, dtype=int)]
    return np.asarray(pool_labels, dtype=np.float64)


def cross_mask_values(richer_outputs_on_pool: np.ndarray) -> np.ndarray:
    """Richer-mask outputs on the pool, used in place of labels."""
    return np.asarray(richer_outputs_on_pool, dtype=np.float64)
