from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from typing import List, Optional, Sequence

from .runner import run_benchmark


def _parse_shots(value: str) -> List[int]:
    return [int(s) for s in value.split(",") if s.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantumuq-benchmark",
        description=(
            "Train a reference variational classifier and sweep shot count, "
            "reporting accuracy and calibration metrics at each shot count."
        ),
    )
    parser.add_argument("--backend", choices=["pennylane", "qiskit"], required=True)
    parser.add_argument(
        "--dataset", choices=["moons", "iris", "breast_cancer"], default="moons"
    )
    parser.add_argument(
        "--shots",
        type=_parse_shots,
        default=[100, 500, 1000, 10000],
        metavar="S1,S2,...",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="ShotBootstrap resamples per shot count",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default=None, help="Optional CSV path to save results"
    )
    args = parser.parse_args(argv)

    results = run_benchmark(
        dataset=args.dataset,
        backend=args.backend,
        shots_list=args.shots,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    rows = [asdict(r) for r in results]

    print(
        f"{'shots':>8}  {'accuracy':>8}  {'nll':>7}  {'ece':>7}  {'brier':>7}  {'entropy':>8}  {'uncertainty':>11}"
    )
    for r in rows:
        print(
            f"{r['shots']:>8}  {r['accuracy']:>8.3f}  {r['nll']:>7.3f}  {r['ece']:>7.3f}  "
            f"{r['brier']:>7.3f}  {r['mean_predictive_entropy']:>8.3f}  {r['mean_uncertainty']:>11.4f}"
        )

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
