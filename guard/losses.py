"""Canonical losses and decision metrics.

A loss is canonical for a link m when the score gradient of the loss is
m(z) - t(y). GUARD uses two:

    cross_entropy   m = softmax              t(y) = one-hot(y)
    bernoulli       m = sigmoid per label    t(y) = y in {0, 1}^K

The Bernoulli loss is averaged over labels, so on multi-label outputs the harm
tolerance delta is per label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Loss:
    name: str
    pointwise: Callable[[np.ndarray, np.ndarray], np.ndarray]
    simplex: bool  # True for one distribution per row, False for independent labels

    def __call__(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pointwise(p, y)


def _cross_entropy(p, y):
    q = np.asarray(p[np.arange(len(y)), y], dtype=np.float64)
    return -np.log(np.clip(q, EPS, None))


def _bernoulli(p, y):
    # float64 before clipping: in float32, 1 - EPS rounds to 1
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean(axis=1)


CROSS_ENTROPY = Loss("cross_entropy", _cross_entropy, simplex=True)
BERNOULLI = Loss("bernoulli", _bernoulli, simplex=False)
LOSSES = {loss.name: loss for loss in (CROSS_ENTROPY, BERNOULLI)}


def get(name: str) -> Loss:
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name]


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def check_outputs(p: np.ndarray, loss: Loss, name: str = "probs") -> None:
    """Reject outputs that are not probabilities, e.g. logits passed by mistake."""
    q = np.asarray(p, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError(f"{name}: non-finite values")
    if q.min() < -1e-6 or q.max() > 1 + 1e-6:
        raise ValueError(f"{name}: values outside [0, 1]")
    if loss.simplex and np.abs(q.sum(1) - 1).max() > 1e-3:
        raise ValueError(f"{name}: rows do not sum to one")


def accuracy(p: np.ndarray, y: np.ndarray, keep: np.ndarray | None = None) -> float:
    """Decision accuracy; per-label accuracy when ``y`` is a label matrix.

    ``keep`` restricts the average to a subset of rows, as for the non-neutral
    samples of CMU-MOSEI.
    """
    if np.ndim(y) == 2:
        ok = ((p > 0.5) == (y > 0.5)).mean(1)
    else:
        ok = p.argmax(1) == y
    return float(ok[keep].mean() if keep is not None else ok.mean())
