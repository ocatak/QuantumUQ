---
title: Uncertainty Quantification in Quantum Machine Learning
description: >-
  A technical introduction to uncertainty quantification in quantum machine
  learning: finite-shot uncertainty, epistemic uncertainty, predictive
  uncertainty, and which QuantumUQ method addresses each one.
---

# Uncertainty Quantification in Quantum Machine Learning

Uncertainty quantification in quantum machine learning asks a question that
a single accuracy number cannot answer: given one specific prediction, how
much should you trust it? This page covers where that uncertainty comes
from, why it matters, and how QuantumUQ measures each source.

## Why accuracy alone is insufficient

A trained variational quantum classifier reports a class label, or a vector
of class probabilities, for each input. Reporting only the argmax label
discards information the model already computed: how close the decision
was, and how much that closeness itself might vary if you asked the model
again. Two classifiers with identical test accuracy can behave very
differently in practice -- one might be confident and correct, the other
confident and *wrong* on the same fraction of cases. Accuracy, measured
once, cannot distinguish between them. Uncertainty quantification is what
lets you tell the difference, and it requires looking at more than the
single number a standard training loop optimizes for.

Quantum models add a wrinkle classical models don't have: even a fixed,
already-trained model gives you a different-looking answer on every run,
because the answer comes from measuring a quantum state a finite number of
times.

## Sources of uncertainty in QML

### Finite-shot (measurement) uncertainty

A quantum circuit's output -- a probability distribution over measurement
outcomes -- is only ever *estimated* from a finite number of shots. Re-run
the same circuit, on the same input, with the same parameters, and the
result will differ each time, purely from measurement statistics. This is
not a bug or noise in the usual sense; it's the nature of quantum
measurement. See
[Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md) for the
statistics behind it.

### Epistemic (model) uncertainty

Independently trained models -- different initializations, different
training data splits, different optimizers -- can converge to different
parameters that all perform similarly on average but disagree on individual
predictions, especially near decision boundaries or on inputs unlike
anything seen during training. This variability reflects what the model
*doesn't know*, as distinct from measurement noise in how it's queried.

### Predictive uncertainty

Predictive uncertainty is the combined spread of a model's output for a
given input -- however much of it comes from finite shots, from model
variability, or both. QuantumUQ represents this directly as a
`PredictiveDistribution`: a set of samples together with their `mean` and
`std`, plus helpers for a central interval (`.interval(alpha)`) and, for
classification, predictive entropy (`.entropy()`).

## How QuantumUQ approaches these problems

QuantumUQ doesn't ship its own uncertainty-aware model architectures.
Instead, it wraps a `Predictor` you already have -- a PennyLane QNode via
`wrap_qnode`, or a Qiskit V2 `Sampler`/`Estimator` primitive via
`wrap_qiskit_sampler`/`wrap_qiskit_estimator` -- and layers a
model-agnostic uncertainty method on top:

| Source of uncertainty | QuantumUQ method | What it does |
| --- | --- | --- |
| Finite-shot / measurement uncertainty | `ShotBootstrap` | Repeats measurement at a fixed shot count, reporting the mean and spread across repeats |
| Epistemic / model uncertainty | `DeepEnsemble` | Aggregates predictions from a list of independently trained predictors |
| Shot-count sensitivity | `NoiseProfile` | Sweeps a list of shot counts and reports how predictions and entropy shift with each |

Every method takes a `Predictor` and returns a `PredictiveDistribution`
through the same interface (`UQModel.predict_dist`), regardless of which
backend the underlying model runs on or how it was trained.

## Relevant metrics

Once you have a `PredictiveDistribution`, `quantumuq.core.metrics` turns it
into numbers you can report or track over training:

- **Classification**: `predictive_entropy` (uncertainty per prediction),
  `nll` and `brier` (accuracy-and-confidence combined), `ece` (calibration
  -- see below).
- **Regression**: `rmse` and `gaussian_nll`.

Predictive entropy and calibration answer different questions -- a
prediction can be low-entropy (confident) and still miscalibrated (wrong
more often than the confidence implies). See
[Calibration in Quantum Machine Learning](qml_calibration.md) for that
distinction in detail.

## Getting started

- [PennyLane quickstart](quickstart_pennylane.md) -- wrapping a QNode with
  `wrap_qnode`.
- [Qiskit quickstart](quickstart_qiskit.md) -- wrapping a V2 Sampler or
  Estimator primitive.
- [Reliability benchmarking](qml_reliability_benchmarking.md) -- measuring
  these quantities reproducibly across a shot-count sweep with
  `quantumuq.benchmarks`.
- Runnable notebooks: `examples/notebooks/00_pennylane_quickstart.ipynb` and
  `examples/notebooks/01_qiskit_quickstart.ipynb` in the
  [GitHub repository](https://github.com/ocatak/QuantumUQ/tree/main/examples/notebooks).
