"""GUARD on one deployment condition.

Steps:
  1. Recalibrate: pick the retrieval settings and the blend weight on D_fit,
     then build the candidate p_corr = (1 - beta) * p_temp + beta * pi_hat.
  2. Certify: fit the action score on D_fit, calibrate its threshold on D_conf,
     and apply the candidate only above the threshold.
The frozen output p_0 is kept wherever the gate says no.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from . import certify as C
from . import losses as L
from . import retrieval as R

BETA_GRID = np.linspace(0.0, 1.0, 41)
GRID = {
    "temperature": (1.0, 2.0),
    "space": ("standardise", "cosine"),
    "weighting": ("uniform", "distance"),
    "k": (3, 5, 8, 12, 20, 35, 50),
}


@dataclass
class Condition:
    """Saved outputs of a frozen model under one deployment condition.

    ``split = (pool, fit, conf, test)`` holds four disjoint sets of row indices;
    pool is the retrieval pool D_source. ``richer`` is the same model's output
    with more modalities (only for the cross-mask target). ``keep`` marks the
    rows the accuracy is computed on.
    """

    probs: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    split: tuple
    richer: np.ndarray | None = None
    keep: np.ndarray | None = None
    loss: str = "cross_entropy"
    targets: tuple = ("hard",)

    def __post_init__(self):
        roles = [np.asarray(r) for r in self.split]
        for i in range(4):
            for j in range(i + 1, 4):
                if np.intersect1d(roles[i], roles[j]).size:
                    raise ValueError("pool, fit, conf and test must be disjoint")
        self.split = tuple(roles)


def random_split(n: int, seed: int = 0) -> tuple:
    """Four equal random parts: pool, fit, conf, test."""
    return tuple(np.array_split(np.random.default_rng(seed).permutation(n), 4))


def select_beta(m, t, y, loss) -> float:
    """Blend weight with the lowest mean loss on D_fit."""
    return float(min(BETA_GRID, key=lambda b: loss((1 - b) * m + b * t, y).mean()))


def _values(cond, target, temperature, loss):
    """What retrieval averages: pool labels (hard) or richer-mask outputs (cross)."""
    pool = cond.split[0]
    if target == "hard":
        return R.hard_label_values(cond.labels[pool], cond.probs.shape[1], loss.simplex)
    richer = cond.richer if temperature == 1.0 else R.temper(cond.richer, temperature, loss.simplex)
    return R.cross_mask_values(richer[pool])


def build_candidate(cond: Condition, grid: dict = GRID) -> dict:
    """Recalibrate: choose the settings by accuracy on D_fit and build the candidate.

    The test rows are not used to choose anything.
    """
    loss = L.get(cond.loss)
    L.check_outputs(cond.probs, loss)
    if cond.richer is not None:
        L.check_outputs(cond.richer, loss, "richer")
    pool, fit, conf, test = cond.split
    y, keep = cond.labels, cond.keep

    def acc(p, idx):
        return L.accuracy(p, y[idx], None if keep is None else keep[idx])

    # features in each retrieval space, fitted on the pool
    spaces = {}
    for sp in grid["space"]:
        z = R.retrieval_space(cond.features[pool], sp)
        spaces[sp] = {name: z(cond.features[idx])
                      for name, idx in zip(("pool", "fit", "conf", "test"), cond.split)}

    best = None
    for T in grid["temperature"]:
        p = cond.probs if T == 1.0 else R.temper(cond.probs, T, loss.simplex)
        for target in cond.targets:
            values = _values(cond, target, T, loss)
            for sp, wt, k in product(grid["space"], grid["weighting"], grid["k"]):
                k = min(k, len(pool) - 1)
                t_fit = R.knn_average(spaces[sp]["fit"], spaces[sp]["pool"], values, k, weighting=wt)
                beta = select_beta(p[fit], t_fit, y[fit], loss)
                score = acc((1 - beta) * p[fit] + beta * t_fit, fit)
                if best is None or score > best[0]:
                    best = (score, dict(temperature=T, target=target, space=sp,
                                        weighting=wt, k=k, beta=beta))
    s = best[1]

    # the candidate on every labelled role, with the chosen settings
    f = spaces[s["space"]]
    p_temp = cond.probs if s["temperature"] == 1.0 else R.temper(cond.probs, s["temperature"], loss.simplex)
    values = _values(cond, s["target"], s["temperature"], loss)
    roles = dict(fit=fit, conf=conf, test=test)
    target = {r: R.knn_average(f[r], f["pool"], values, s["k"], weighting=s["weighting"]) for r in roles}
    corrected = {r: (1 - s["beta"]) * p_temp[idx] + s["beta"] * target[r] for r, idx in roles.items()}
    # temperature belongs to the candidate only: the base output stays p_0
    base = {r: cond.probs[idx] for r, idx in roles.items()}
    return dict(settings=s, target=target, corrected=corrected, base=base, loss=loss)


def run(cond: Condition, alpha: float = 0.2, delta: float = 0.05, grid: dict = GRID) -> dict:
    """GUARD on one condition. Returns gain, joint harm and apply rate on D_test."""
    cand = build_candidate(cond, grid)
    loss, b, t, c = cand["loss"], cand["base"], cand["target"], cand["corrected"]
    _, fit, conf, test = cond.split
    y = cond.labels

    model = C.fit_action_score(b["fit"], t["fit"], c["fit"], y[fit], loss)
    gate = C.certify(model, b["conf"], t["conf"], c["conf"], y[conf], b["test"], t["test"],
                     loss, alpha, delta, fit=(b["fit"], t["fit"], c["fit"], y[fit]))

    harmful = (loss(c["test"], y[test]) - loss(b["test"], y[test])) > delta
    keep = None if cond.keep is None else cond.keep[test]
    base_acc = L.accuracy(b["test"], y[test], keep)

    def outcome(apply):
        p = np.where(apply[:, None], c["test"], b["test"])
        return L.accuracy(p, y[test], keep) - base_acc, float((apply & harmful).mean())

    gain, harm = outcome(gate["apply"])
    blanket_gain, blanket_harm = outcome(np.ones(len(test), bool))
    return dict(base=base_acc, gain=gain, joint_harm=harm, apply=float(gate["apply"].mean()),
                blanket_gain=blanket_gain, blanket_harm=blanket_harm,
                **cand["settings"])
