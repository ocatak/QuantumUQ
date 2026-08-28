---
title: QuantumUQ — Uncertainty Quantification for Quantum Machine Learning
description: >-
  QuantumUQ is an open-source Python library for uncertainty quantification,
  calibration, shot-noise analysis, and reliability benchmarking in quantum
  machine learning, with support for Qiskit and PennyLane.
---

# QuantumUQ

QuantumUQ is a lightweight, open-source Python library for **uncertainty
quantification in quantum machine learning (QML)**. It wraps quantum models
you've already built in **Qiskit** or **PennyLane** and adds the tools to
answer a question raw accuracy can't: *how much should you trust a given
prediction?*

Point predictions from a quantum classifier hide two separate problems.
Finite measurement shots make every prediction statistically noisy, and
nothing about a model's raw accuracy tells you whether its reported
confidence is trustworthy. QuantumUQ addresses both directly, alongside
reliability benchmarking so results are reproducible across shot budgets
and backends.

```bash
pip install quantumuq
```

New to the library? [Try QuantumUQ in 15 Minutes](fifteen_minutes.md) runs a
complete example, end to end, in a single copy-pasteable script.

## What can QuantumUQ measure?

- **Finite-shot (measurement) uncertainty** -- how much a prediction would
  change if you re-ran the circuit with the same shot budget, via
  `ShotBootstrap`.
- **Epistemic (model) uncertainty** -- variability across independently
  trained models, via `DeepEnsemble`.
- **Shot-count sensitivity** -- how predictions and their uncertainty change
  as shot count varies, via `NoiseProfile` and the benchmark suite's shot
  sweeps.
- **Calibration** -- whether a classifier's stated confidence matches its
  actual accuracy, via Expected Calibration Error (`ece`) and reliability
  diagrams.
- **Predictive uncertainty and reliability metrics** -- `predictive_entropy`,
  Brier score (`brier`), and negative log-likelihood (`nll`) for
  classification; `rmse` and Gaussian NLL for regression.
- **Reproducible reliability benchmarking** -- `quantumuq.benchmarks` trains
  a reference model and sweeps shot count, reporting these metrics
  consistently across datasets and backends.

QuantumUQ measures and reports these quantities; it does not (yet)
automatically correct miscalibration for you -- see
[Calibration in Quantum Machine Learning](qml_calibration.md) for what that
distinction means in practice.

## Start here

- [Try QuantumUQ in 15 Minutes](fifteen_minutes.md) -- install, run one
  example, and produce uncertainty and calibration metrics.
- [Qiskit quickstart](quickstart_qiskit.md) -- wrapping a Qiskit V2
  primitive.
- [PennyLane quickstart](quickstart_pennylane.md) -- wrapping a PennyLane
  QNode.
- [Uncertainty quantification in quantum machine learning](qml_uncertainty_quantification.md) --
  the concepts and which QuantumUQ method addresses which source of
  uncertainty.
- [Calibration in quantum machine learning](qml_calibration.md) -- ECE,
  Brier score, NLL, and why more shots don't fix a miscalibrated model.
- [Shot noise and finite-shot uncertainty](qml_shot_noise.md) -- what
  measurement shots are and how `ShotBootstrap` quantifies their effect.
- [Reliability benchmarking](qml_reliability_benchmarking.md) -- the
  `quantumuq.benchmarks` suite and `quantumuq-benchmark` CLI.

## Design

- **Framework-agnostic core**: every uncertainty method operates on a small
  `Predictor` protocol (`predict`, `predict_proba`, `task`), so the same
  `ShotBootstrap`/`DeepEnsemble`/`NoiseProfile` code works against either
  backend.
- **Built on your own models**: QuantumUQ wraps a circuit or QNode you
  provide (`wrap_qiskit_sampler`, `wrap_qiskit_estimator`, `wrap_qnode`); it
  does not ship its own quantum model architectures for general use, beyond
  the small reference classifiers used internally by the benchmark suite.
- **Designed for research**: small, explicit code that's easy to inspect and
  extend, not a black box.

QuantumUQ is research software (PyPI classifier: Alpha). See the
[paper](https://doi.org/10.36227/techrxiv.177205048.88644983/v1) for the
motivation and design, and the
[GitHub repository](https://github.com/ocatak/QuantumUQ) for source,
issues, and the full [changelog](https://github.com/ocatak/QuantumUQ/blob/main/CHANGELOG.md).
