"""Certify: decide where the correction replaces the frozen output.

An action score, fitted on D_fit, predicts whether the correction lowers the loss.
Its threshold is calibrated on D_conf by conformal risk control, so that under
exchangeability

    P(apply and loss increase > delta) <= alpha

whatever the score. Raising the threshold afterwards never breaks this bound.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def action_features(base: np.ndarray, target: np.ndarray, simplex: bool = True) -> np.ndarray:
    """Five features that need no label.

    Base confidence, base entropy, target confidence, an agreement flag and the
    l1 distance between base and target. With more than 20 classes the flag is
    always 1.
    """
    b = np.clip(np.asarray(base, dtype=np.float64), EPS, 1.0 - EPS)
    t = np.asarray(target, dtype=np.float64)
    if simplex:
        ent = -(b * np.log(b)).sum(1)
    else:
        ent = -(b * np.log(b) + (1 - b) * np.log(1 - b)).sum(1)
    return np.column_stack([
        b.max(1),
        ent,
        t.max(1),
        (np.abs(b - t).mean(1) < 0.1).astype(float),
        np.abs(b - t).sum(1),
    ])


def fit_action_score(base_fit, target_fit, corrected_fit, labels_fit, loss):
    """Logistic model of "the correction lowers the loss". None if nothing to learn."""
    z = (loss(corrected_fit, labels_fit) < loss(base_fit, labels_fit)).astype(int)
    if len(set(z.tolist())) < 2:
        return None
    x = action_features(base_fit, target_fit, loss.simplex)
    return LogisticRegression(max_iter=2000).fit(x, z)


def action_score(model, base, target, simplex: bool = True) -> np.ndarray:
    if model is None:
        return np.zeros(len(base))
    return model.predict_proba(action_features(base, target, simplex))[:, 1]


def crc_threshold(score_conf: np.ndarray, harmful_conf: np.ndarray, alpha: float) -> float | None:
    """Lowest threshold whose corrected harm rate on D_conf is at most alpha.

    The rule is score > lambda. Returns None (no intervention) if no threshold works.
    """
    n = len(score_conf)
    level = alpha - (1.0 - alpha) / n
    if level <= 0:
        return None
    s = np.asarray(score_conf, dtype=np.float64)
    h = np.asarray(harmful_conf, dtype=float)
    cand = np.concatenate([[-np.inf], np.unique(s), [np.nextafter(s.max(), np.inf)]])
    risk = np.array([float(h[s > c].sum()) / n for c in cand])
    ok = np.nonzero(risk <= level)[0]
    if len(ok) == 0:
        return None
    return float(cand[ok[0]])


def tighten(scores_fit: np.ndarray, gain_fit: np.ndarray, lam: float) -> float:
    """Raise the threshold on D_fit.

    Keep the highest threshold whose total gain is within one standard error of
    the best. ``gain_fit`` is the loss decrease per sample. Never below ``lam``.
    """
    cand = np.unique(scores_fit[scores_fit > lam])
    cand = np.concatenate([[lam], cand]) if len(cand) else np.array([lam])
    tot, se = [], []
    for c in cand:
        g = gain_fit[scores_fit > c]
        tot.append(float(g.sum()))
        se.append(float(g.std(ddof=1) * np.sqrt(len(g))) if len(g) > 1 else 0.0)
    tot, se = np.array(tot), np.array(se)
    best = int(np.argmax(tot))
    if tot[best] <= 0:
        return float(max(lam, np.nextafter(scores_fit.max(), np.inf)))
    ok = np.nonzero(tot >= tot[best] - se[best])[0]
    return float(cand[ok[-1]])


def certify(model, base_conf, target_conf, corrected_conf, labels_conf,
            base_test, target_test, loss, alpha, delta, fit) -> dict:
    """Calibrate on D_conf, tighten on D_fit, then decide on the test rows.

    ``fit`` is (base, target, corrected, labels) on D_fit.
    """
    harmful = (loss(corrected_conf, labels_conf) - loss(base_conf, labels_conf)) > delta
    if model is None:
        n = len(base_test)
        return {"apply": np.zeros(n, bool), "lambda": None, "lambda_crc": None,
                "score_test": np.zeros(n)}
    lam_crc = crc_threshold(action_score(model, base_conf, target_conf, loss.simplex), harmful, alpha)
    lam = lam_crc
    if lam is not None:
        b_f, t_f, c_f, y_f = fit
        lam = tighten(action_score(model, b_f, t_f, loss.simplex), loss(b_f, y_f) - loss(c_f, y_f), lam)
    s_test = action_score(model, base_test, target_test, loss.simplex)
    apply = np.zeros(len(s_test), bool) if lam is None else s_test > lam
    return {"apply": apply, "lambda": lam, "lambda_crc": lam_crc, "score_test": s_test}


# Learn-then-Test, used only as a baseline in experiments/compare_rules.py.

def _kl_bernoulli(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(risk: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha."""
    if risk >= alpha:
        return 1.0
    p_bentkus = float(np.e * binom.cdf(np.ceil(n * risk), n, alpha))
    p_hoeffding = float(np.exp(-n * _kl_bernoulli(risk, alpha)))
    return float(min(1.0, p_bentkus, p_hoeffding))


def ltt_threshold(score_conf, harmful_conf, alpha, zeta=0.05, grid_scores=None, n_grid=100):
    """Lowest threshold that passes fixed-sequence testing at level zeta.

    Candidate thresholds come from ``grid_scores`` (the D_fit scores), so they are
    fixed before D_conf is read. The rule is score >= lambda.
    """
    n = len(score_conf)
    ref = score_conf if grid_scores is None else grid_scores
    chosen = None
    for lam in np.quantile(ref, np.linspace(1.0, 0.0, n_grid)):
        risk = float(((score_conf >= lam) & harmful_conf).mean())
        if hb_pvalue(risk, n, alpha) > zeta:
            break
        chosen = float(lam)
    return chosen
