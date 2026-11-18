"""Run GUARD on every condition of one benchmark.

    python -m experiments.run_guard mosei
    python -m experiments.run_guard iemocap --limit 10 --procs 4

Writes one row per condition to results/guard_<benchmark>.csv and prints the
mean over conditions. Gains are in accuracy (per-label accuracy on PTB-XL).
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import guard
from experiments.benchmarks import LOADERS, conditions

COLUMNS = ("base", "gain", "joint_harm", "apply", "blanket_gain", "blanket_harm")


def job(args):
    name, kwargs, alpha, delta = args
    row = guard.run(LOADERS[name](**kwargs), alpha=alpha, delta=delta)
    return {**kwargs, **row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=sorted(LOADERS))
    ap.add_argument("--alpha", type=float, default=0.2, help="harm budget")
    ap.add_argument("--delta", type=float, default=0.05, help="tolerated loss increase")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N conditions")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    jobs = [(a.benchmark, kw, a.alpha, a.delta) for kw in conditions(a.benchmark)[:a.limit]]
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"guard_{a.benchmark}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{a.benchmark}: {len(rows)} conditions, alpha={a.alpha}, delta={a.delta}")
    for col in COLUMNS:
        print(f"  {col:13s} {np.mean([r[col] for r in rows]):+.4f}")
    print(f"rows written to {path}")


if __name__ == "__main__":
    main()
