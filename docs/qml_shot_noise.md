---
title: Shot Noise and Finite-Shot Uncertainty in Quantum Machine Learning
description: >-
  What quantum measurement shots are, why finite shots introduce
  statistical uncertainty, and how QuantumUQ's ShotBootstrap quantifies
  the effect of shot count on a quantum classifier's predictions.
---

# Shot Noise and Finite-Shot Uncertainty in Quantum Machine Learning

## What a "shot" is

Measuring a qubit doesn't return its full quantum state -- it returns a
single classical outcome, sampled according to the state's probability
distribution over measurement outcomes. A "shot" is one such measurement.
Running the same circuit for `N` shots and counting outcomes gives an
*estimate* of the underlying probabilities, not the exact probabilities
themselves. This is true even for a perfect, noiseless simulator: shot
noise is a property of quantum measurement, not of hardware imperfection.

## Why finite shots introduce statistical uncertainty

Each shot's outcome is a random draw. Estimating a probability `p` from `N`
shots gives a standard error that scales approximately as:

```
standard error ≈ sqrt(p(1-p) / N)  ≈  O(1 / sqrt(N))
```

Doubling the shot count does not halve this error -- shrinking it by a
given factor requires multiplying the shot count by that factor *squared*.
Going from 100 to 10,000 shots (100x more shots) shrinks the standard error
by roughly `sqrt(100) = 10x`, not 100x. This is why quantum measurement
budgets grow quickly: precision is expensive.

This shows up directly in QuantumUQ's own measurements. On a trained
variational classifier, the
[PennyLane Community Demo notebook](https://github.com/ocatak/QuantumUQ/blob/main/examples/notebooks/08_pennylane_community_demo.ipynb)
records a mean `ShotBootstrap` uncertainty of `0.0405` at 100 shots, falling
to `0.0037` at 10,000 shots -- a roughly 11x reduction for a 100x increase
in shots, matching the `sqrt(100) ≈ 10x` expectation.

## Why 100 vs. 1,000 vs. 10,000 shots matters

A low shot budget (e.g. 100 shots) makes each individual prediction's
measured probabilities noisier -- re-running the same circuit on the same
input can give a visibly different answer. A higher shot budget (e.g.
10,000 shots) makes each measurement more precise, at proportionally higher
execution cost (more circuit executions on hardware or in simulation).
Choosing a shot budget is a real trade-off between measurement precision
and cost, and QuantumUQ's shot sweeps
(`NoiseProfile`, `quantumuq.benchmarks.run_benchmark`) exist to let you see
that trade-off on your own model rather than guessing at it.

## Finite-shot uncertainty vs. epistemic uncertainty

Shot noise is **aleatoric with respect to measurement**: it comes from the
randomness of quantum measurement itself, and doesn't go away by training a
better model -- only by taking more shots (or reducing what you're
estimating). This is a different thing from **epistemic (model)
uncertainty**, which comes from limited training data or model
misspecification, and *can* shrink with a better model or more data. See
[Uncertainty Quantification in Quantum Machine Learning](qml_uncertainty_quantification.md)
for how QuantumUQ separates the two (`ShotBootstrap` for the former,
`DeepEnsemble` for the latter). It is also a different question from
**calibration** -- see
[Calibration in Quantum Machine Learning](qml_calibration.md) for why
reducing shot noise does not, by itself, fix a miscalibrated model.

## How `ShotBootstrap` works

`ShotBootstrap` estimates finite-shot uncertainty by repeating the
measurement `n_samples` times at a given shot count and reporting the
spread across those repeats:

```python
from quantumuq import ShotBootstrap

uq = ShotBootstrap(n_samples=16, shots=1000, seed=0)
uq_model = predictor.with_uq(uq)
dist = uq_model.predict_dist(X)  # dist.mean, dist.std over the 16 repeats
```

One practical gotcha: some quantum primitives reset their random state on
every call if seeded with a plain integer, which makes every repeat
identical and silently produces a `dist.std` of zero. Seeding with a
`numpy.random.Generator` instance instead lets the state advance across
calls, so repeats are genuinely independent -- covered in detail (with a
real before/after comparison) in the
[Qiskit V2 primitives notebook](https://github.com/ocatak/QuantumUQ/blob/main/examples/notebooks/07_qiskit_v2_primitives.ipynb).

## Analyzing shot-count sensitivity with the benchmark suite

Sweeping shot count on a specific circuit is straightforward with
`NoiseProfile` (see the notebooks below), but for a reproducible,
end-to-end sweep on a reference model, use
`quantumuq.benchmarks.run_benchmark`:

```bash
quantumuq-benchmark --backend pennylane --dataset moons --shots 100,500,1000,10000
```

This reports accuracy, `nll`, `ece`, `brier`, `predictive_entropy`, and mean
`ShotBootstrap` uncertainty at each shot count in the sweep. See
[Reliability Benchmarking](qml_reliability_benchmarking.md) for the full
CLI and Python API.

## Getting started

- [Qiskit quickstart](quickstart_qiskit.md) and
  [PennyLane quickstart](quickstart_pennylane.md) -- wrapping a circuit or
  QNode before applying `ShotBootstrap`.
- Runnable notebooks: `examples/notebooks/04_shots_sweep_noise_profile.ipynb`
  (both backends) and
  `examples/notebooks/07_qiskit_v2_primitives.ipynb` (the seeding gotcha
  above), in the
  [GitHub repository](https://github.com/ocatak/QuantumUQ/tree/main/examples/notebooks).
