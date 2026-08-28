---
title: Uncertainty Quantification for Qiskit Quantum Machine Learning
description: >-
  Learn how to measure predictive uncertainty, shot noise, calibration,
  ECE, entropy, and reliability in Qiskit quantum machine learning models
  using QuantumUQ.
---

# Quickstart: Qiskit

Qiskit quantum machine learning uncertainty quantification with QuantumUQ
starts with wrapping a circuit you've already built -- no changes to the
circuit itself are required.

## V2 primitives

QuantumUQ's Qiskit adapters (`wrap_qiskit_sampler`, `wrap_qiskit_estimator`)
require Qiskit's **V2 primitives** (`BaseSamplerV2`/`BaseEstimatorV2` --
e.g. `StatevectorSampler`, `StatevectorEstimator`, or a hardware-backed V2
primitive), not the removed V1 `Sampler`/`Estimator` classes. A `Sampler`
returns bitstring counts per circuit; `wrap_qiskit_sampler` converts those
into class probabilities. See the
[Qiskit V2 primitives notebook](https://github.com/ocatak/QuantumUQ/blob/main/examples/notebooks/07_qiskit_v2_primitives.ipynb)
for the full Sampler/Estimator walkthrough and a seeding gotcha that will
otherwise silently zero out your uncertainty estimates.

## Wrap a circuit and measure uncertainty

```python
from quantumuq import wrap_qiskit_sampler, ShotBootstrap
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np

theta = Parameter("theta")
qc = QuantumCircuit(1)
qc.ry(theta, 0)
qc.measure_all()

def feature_map(X: np.ndarray):
    return [[float(x[0])] for x in np.atleast_2d(X)]

# A Generator instance (not a plain int) makes the sampler's RNG state
# advance across calls, so repeated ShotBootstrap draws actually differ.
sampler = StatevectorSampler(seed=np.random.default_rng(0))
predictor = wrap_qiskit_sampler(
    sampler,
    circuit=qc,
    task="classification",
    n_classes=2,
    feature_map=feature_map,
)
uq = ShotBootstrap(n_samples=8, shots=1000, seed=0)
uq_model = predictor.with_uq(uq)
dist = uq_model.predict_dist(np.random.randn(4, 1))
print(dist.mean.shape, dist.std.shape)
```

Finite shots make every measurement statistically noisy -- `dist.std`
reports exactly how much, at the shot count you asked for. See
[Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md) for why that
matters and how it scales with shot count.

## Beyond one prediction

- `NoiseProfile` sweeps shot counts on this same predictor -- see
  `examples/notebooks/01_qiskit_quickstart.ipynb`.
- `quantumuq.core.metrics.ece`/`nll`/`brier`/`predictive_entropy` turn
  `dist.mean` into calibration and uncertainty metrics -- see
  [Calibration in Quantum Machine Learning](qml_calibration.md).
- For a reproducible shot-count sweep on a reference model instead of your
  own circuit, see [Reliability Benchmarking](qml_reliability_benchmarking.md)
  and `quantumuq-benchmark --backend qiskit`.

## Learn more

- [Uncertainty Quantification in Quantum Machine Learning](qml_uncertainty_quantification.md)
- [Calibration in Quantum Machine Learning](qml_calibration.md)
- [Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md)
- [Reliability Benchmarking](qml_reliability_benchmarking.md)

The corresponding runnable notebook is
`examples/notebooks/01_qiskit_quickstart.ipynb` in the
[GitHub repository](https://github.com/ocatak/QuantumUQ/tree/main/examples/notebooks).
