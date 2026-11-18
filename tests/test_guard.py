"""Checks on synthetic data. No download needed: run ``pytest -q``."""

import numpy as np
import pytest

from guard import Condition, random_split, run
from guard.certify import crc_threshold, tighten
from guard.retrieval import knn_average, temper

FAST = {"temperature": (1.0,), "space": ("standardise",), "weighting": ("uniform",), "k": (50,)}


def make_condition(n=6000, classes=4, gap=0.6, bias=0.0, seed=0, split_seed=0):
    """A frozen model that is too flat (gap) and favours class 0 (bias)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    logits = 1.5 * x[:, :classes]
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(classes, p=row) for row in p_true])
    m = np.exp(logits * (1 - gap) + bias * np.eye(classes)[0])
    return Condition(m / m.sum(1, keepdims=True), x, y, random_split(n, split_seed))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_joint_harm_within_budget_on_average(alpha):
    # The bound holds in expectation over calibration draws, not on every split.
    harms = [run(make_condition(split_seed=s), alpha=alpha, grid=FAST)["joint_harm"] for s in range(30)]
    assert np.mean(harms) <= alpha + 0.02


def test_correction_helps_when_there_is_a_gap():
    r = run(make_condition(gap=0.0, bias=2.0), grid=FAST)
    assert r["gain"] > 0.02 and r["apply"] > 0


def test_useless_target_does_not_hurt():
    cond = make_condition()
    cond.features = np.random.default_rng(1).normal(size=cond.features.shape)
    r = run(cond, grid=FAST)
    assert r["gain"] > -0.02 and r["joint_harm"] <= 0.22


def test_roles_must_be_disjoint():
    with pytest.raises(ValueError):
        Condition(np.full((40, 2), 0.5), np.zeros((40, 1)), np.zeros(40, int),
                  (np.arange(10), np.arange(5, 15), np.arange(20, 30), np.arange(30, 40)))


def test_crc_uses_the_corrected_level():
    n, alpha = 200, 0.2
    score = np.linspace(0, 1, n)
    assert (score > crc_threshold(score, np.zeros(n, bool), alpha)).all()
    lam = crc_threshold(score, np.ones(n, bool), alpha)
    assert (score > lam).sum() <= int(n * (alpha - (1 - alpha) / n)) + 1


def test_tied_scores_are_charged_together():
    score = np.concatenate([np.zeros(60), np.full(40, 0.7)])
    harmful = np.concatenate([np.zeros(60, bool), np.ones(40, bool)])
    lam = crc_threshold(score, harmful, 0.2)
    applied = np.zeros(100, bool) if lam is None else score > lam
    assert not applied[60:].any()


def test_tightening_never_lowers_the_threshold():
    assert tighten(np.linspace(0, 0.4, 50), -np.ones(50), 0.9) >= 0.9


def test_multilabel_temperature_is_per_label():
    p = np.array([[0.9, 0.8, 0.7], [0.2, 0.3, 0.1]])
    out = temper(p, 2.0, simplex=False)
    assert np.allclose(out, 1 / (1 + np.exp(-np.log(p / (1 - p)) / 2)))
    assert ((out > 0.5) == (p > 0.5)).all()


def test_target_does_not_depend_on_other_queries():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 6))
    values = np.eye(3)[rng.integers(0, 3, 300)]
    q = rng.normal(size=(40, 6))
    alone = knn_average(q[:1], pool, values, 10, weighting="distance")
    batch = knn_average(q, pool, values, 10, weighting="distance")[:1]
    assert np.allclose(alone, batch)
