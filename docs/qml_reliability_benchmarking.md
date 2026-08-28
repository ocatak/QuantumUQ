---
title: Reliability Benchmarking for Quantum Machine Learning
description: >-
  Reference datasets, models, metrics, and the quantumuq-benchmark CLI for
  reproducible reliability benchmarking of quantum machine learning models
  across shot counts and backends.
---

# Reliability Benchmarking for Quantum Machine Learning

`quantumuq.benchmarks` trains a small reference variational classifier and
sweeps shot count, reporting accuracy and calibration metrics
reproducibly -- so a paper can cite a fixed benchmark ("we evaluated
uncertainty using the QuantumUQ benchmark") rather than just "we used a
Python package."

This page documents exactly what's currently implemented. It's a
lightweight benchmark harness, not a rigorous ML pipeline -- the
[Limitations](#limitations) section below is not an afterthought.

## Installation

```bash
pip install "quantumuq[benchmarks]"  # adds scikit-learn, for iris/breast_cancer
```

`moons` has no extra dependency; `iris` and `breast_cancer` require the
`benchmarks` extra.

## Supported datasets

| Dataset | Extra dependency | Notes |
| --- | --- | --- |
| `moons` | none | Two-moons toy dataset (`quantumuq.datasets.toy.make_moons`) |
| `iris` | scikit-learn | Binary subset (setosa vs. versicolor), reduced to 2 features |
| `breast_cancer` | scikit-learn | Breast Cancer Wisconsin dataset, reduced to 2 features via PCA |

All three are reduced to exactly 2 features, so the same small reference
circuit architecture applies to each.

## Supported frameworks

- **PennyLane**: a 2-qubit variational classifier, gradient-trained via
  `qml.grad`.
- **Qiskit**: a 2-qubit variational classifier, trained via SPSA (a
  gradient-free optimizer), since Qiskit circuits aren't differentiable
  through this library.

## Metrics reported per shot count

- `accuracy`
- `nll` (negative log-likelihood)
- `ece` (Expected Calibration Error)
- `brier` (Brier score)
- `mean_predictive_entropy`
- `mean_uncertainty` (mean `ShotBootstrap` standard deviation)

## Shot sweeps

`run_benchmark` trains the reference model once, then evaluates it at each
shot count in `shots_list` (default `(100, 500, 1000, 10000)`) using
`ShotBootstrap`, so all reported metrics come from the same fitted model
across the sweep.

## CLI usage

```bash
quantumuq-benchmark --backend pennylane --dataset moons --shots 100,500,1000,10000
quantumuq-benchmark --backend qiskit --dataset breast_cancer --shots 100,1000 --output results.csv
```

Flags: `--backend` (`pennylane` or `qiskit`, required), `--dataset`
(`moons`, `iris`, or `breast_cancer`; default `moons`), `--shots`
(comma-separated; default `100,500,1000,10000`), `--n-samples`
(`ShotBootstrap` resamples per shot count; default `8`), `--seed` (default
`0`), and `--output` (optional CSV path).

## Python usage

```python
from quantumuq.benchmarks import run_benchmark

results = run_benchmark(
    dataset="moons",
    backend="pennylane",
    shots_list=[100, 500, 1000, 10000],
)
for r in results:
    print(r.shots, r.accuracy, r.ece, r.mean_uncertainty)
```

`run_benchmark` returns a list of `BenchmarkResult` dataclass instances, one
per shot count.

## Limitations

This is a lightweight harness, not a rigorous ML pipeline:

- A single, fixed train/test split -- no cross-validation.
- Datasets larger than `max_samples` (default 150) are randomly subsampled
  so runtime stays small and consistent regardless of the source dataset's
  native size; this is not the full dataset.
- All datasets are reduced to 2 features and a shared 2-qubit circuit
  architecture, for a consistent comparison across datasets and backends --
  not a tuned model for any one of them.
- Quantum kernel classifiers and hybrid QNN models are not implemented.
- Only shot-count noise is swept; depolarizing and readout noise models are
  not yet implemented.
- Because a symmetric cross-entropy loss doesn't distinguish which class is
  "0" versus "1", either backend's optimizer can converge to a solution
  with the two classes consistently swapped. `run_benchmark` detects this
  on the training set and corrects for it automatically, but it's a real
  property of the optimization, not something to be surprised by if you
  extend the reference models yourself.

## Intended role in reproducible research

The point of a fixed benchmark suite is that "we evaluated calibration
using QuantumUQ's benchmark on the two-moons dataset at 1,000 shots" is a
reproducible claim in a way that an ad hoc script isn't. Use it to compare
shot budgets, backends, or your own modified reference models on equal
footing -- not as a substitute for evaluating your own production model
directly with `ShotBootstrap`/`DeepEnsemble`/`NoiseProfile`.

## Getting started

- [Try QuantumUQ in 15 Minutes](fifteen_minutes.md)
- [Shot noise and finite-shot uncertainty](qml_shot_noise.md) -- the
  statistical background behind the shot sweep.
- [Calibration in quantum machine learning](qml_calibration.md) -- how to
  read the `ece`/`nll`/`brier` columns.
- Source: `quantumuq/benchmarks/` in the
  [GitHub repository](https://github.com/ocatak/QuantumUQ).
