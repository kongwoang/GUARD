"""Compare intervention rules on one fixed corrector (Table 3 of the paper).

    python -m experiments.compare_rules mosei
    python -m experiments.compare_rules ptbxl --limit 6

Every rule sees the same frozen model, candidate correction, splits and budget.
Only the score (learned, confidence, l1 distance, random), the calibration
(conformal risk control or Learn-then-Test) and the D_fit tightening change.
Each cell is accuracy gain / joint harm.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.benchmarks import LOADERS, conditions
from guard import build_candidate
from guard import certify as C
from guard import losses as L

ALPHA, DELTA, ZETA = 0.2, 0.05, 0.05

# (column, score, calibration, tightened)
RULES = [
    ("conf", "confidence", "crc", False),
    ("ltt", "learned", "ltt", False),
    ("learned", "learned", "crc", False),
    ("conf_t", "confidence", "crc", True),
    ("l1_t", "l1", "crc", True),
    ("random_t", "random", "crc", True),
    ("ltt_t", "learned", "ltt", True),
    ("guard", "learned", "crc", True),
]


def compare(cond):
    cand = build_candidate(cond)
    loss, b, t, c = cand["loss"], cand["base"], cand["target"], cand["corrected"]
    _, fit, conf, test = cond.split
    y = cond.labels
    keep = None if cond.keep is None else cond.keep[test]

    harmful = (loss(c["test"], y[test]) - loss(b["test"], y[test])) > DELTA
    harmful_conf = (loss(c["conf"], y[conf]) - loss(b["conf"], y[conf])) > DELTA
    gain_fit = loss(b["fit"], y[fit]) - loss(c["fit"], y[fit])
    base_acc = L.accuracy(b["test"], y[test], keep)
    model = C.fit_action_score(b["fit"], t["fit"], c["fit"], y[fit], loss)

    def scores(role):
        base, target = b[role], t[role]
        conf_score = -np.abs(base - 0.5).mean(1) if not loss.simplex else -base.max(1)
        return dict(learned=C.action_score(model, base, target, loss.simplex),
                    confidence=conf_score,
                    l1=np.abs(base - target).sum(1),
                    random=np.random.default_rng(0).random(len(base)))

    s_fit, s_conf, s_test = scores("fit"), scores("conf"), scores("test")

    def outcome(apply):
        p = np.where(apply[:, None], c["test"], b["test"])
        return L.accuracy(p, y[test], keep) - base_acc, float((apply & harmful).mean())

    row = {}
    row["blanket_gain"], row["blanket_harm"] = outcome(np.ones(len(test), bool))
    for col, name, calib, tight in RULES:
        apply = np.zeros(len(test), bool)
        if not (name == "learned" and model is None):
            if calib == "crc":
                lam = C.crc_threshold(s_conf[name], harmful_conf, ALPHA)
            else:
                lam = C.ltt_threshold(s_conf[name], harmful_conf, ALPHA, ZETA, grid_scores=s_fit[name])
            if lam is not None and tight:
                apply = s_test[name] > C.tighten(s_fit[name], gain_fit, lam)
            elif lam is not None:
                apply = s_test[name] >= lam if calib == "ltt" else s_test[name] > lam
        row[f"{col}_gain"], row[f"{col}_harm"] = outcome(apply)
    return row


def job(args):
    name, kwargs = args
    return {**kwargs, **compare(LOADERS[name](**kwargs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=sorted(LOADERS))
    ap.add_argument("--limit", type=int, default=None, help="run only the first N conditions")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    jobs = [(a.benchmark, kw) for kw in conditions(a.benchmark)[:a.limit]]
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"rules_{a.benchmark}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{a.benchmark}: {len(rows)} conditions, alpha={ALPHA}, delta={DELTA}")
    print(f"  {'rule':10s} {'gain':>8s} {'harm':>7s}")
    for col in ["blanket"] + [r[0] for r in RULES]:
        g = np.mean([r[f"{col}_gain"] for r in rows])
        h = np.mean([r[f"{col}_harm"] for r in rows])
        print(f"  {col:10s} {g:+8.3f} {h:7.3f}{' *' if h > ALPHA else ''}")
    print(f"rows written to {path}  (* harm above alpha)")


if __name__ == "__main__":
    main()
