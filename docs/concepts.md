---
title: Concepts — Uncertainty and Calibration in Quantum Machine Learning
description: >-
  An overview of the sources of uncertainty in quantum machine learning
  models and the QuantumUQ methods that address each one.
---

# Concepts: Uncertainty in Quantum Machine Learning

Quantum models introduce several sources of uncertainty:

- **Shot noise (statistical)**: finite sampling of measurement outcomes.
- **Hardware noise (aleatoric)**: decoherence, gate errors, readout noise.
- **Model uncertainty (epistemic)**: limited training data, model misspecification.

QuantumUQ focuses on *model-agnostic* techniques that sit on top of existing QML models:

- **ShotBootstrap**: resample shots / repeated forward passes.
- **DeepEnsemble**: independent predictors trained from different initializations or data splits.
- **NoiseProfile**: sweep shots and quantify stability (entropy and probability variance).

All methods work on a small **Predictor protocol** (`predict`, `predict_proba`, and `task`).

## Read more

Each topic below expands on one part of this picture in more depth:

- [Uncertainty quantification in quantum machine learning](qml_uncertainty_quantification.md) --
  why accuracy alone is insufficient, and which QuantumUQ method addresses
  which source of uncertainty.
- [Calibration in quantum machine learning](qml_calibration.md) -- whether a
  model's stated confidence matches its actual accuracy, and why that's a
  different question from how noisy its measurements are.
- [Shot noise and finite-shot uncertainty](qml_shot_noise.md) -- what
  measurement shots are, the `1/sqrt(N)` intuition, and how `ShotBootstrap`
  quantifies their effect.
- [Reliability benchmarking](qml_reliability_benchmarking.md) -- measuring
  all of the above reproducibly across datasets, backends, and shot budgets
  with `quantumuq.benchmarks`.
