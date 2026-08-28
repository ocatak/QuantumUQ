---
title: Uncertainty Quantification for PennyLane Quantum Machine Learning
description: >-
  Add uncertainty quantification, shot-bootstrap analysis, calibration,
  and reliability metrics to PennyLane quantum machine learning models
  with QuantumUQ.
---

# Quickstart: PennyLane

PennyLane quantum machine learning uncertainty quantification with
QuantumUQ starts with a QNode you've already built and (optionally)
trained -- `wrap_qnode` handles the rest.

## Wrap a QNode

```python
from quantumuq import wrap_qnode, ShotBootstrap
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def circuit(x, params):
    qml.AngleEmbedding(x, wires=[0, 1])
    qml.StronglyEntanglingLayers(params, wires=[0, 1])
    return qml.probs(wires=[0, 1])

# 2 qubits -> 4 outcomes (|00>,|01>,|10>,|11>); collapse to 2 classes.
def probs_4_to_2(p):
    p = np.asarray(p)
    if p.ndim == 1:
        return np.array([p[0] + p[1], p[2] + p[3]])
    return np.stack([p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]], axis=-1)

params = 0.1 * np.random.default_rng(0).standard_normal((1, 2, 3))
predictor = wrap_qnode(
    circuit, task="classification", n_classes=2, params=params, postprocess=probs_4_to_2
)
```

`wrap_qnode` doesn't train anything -- pass in `params` you've already
fitted with your own training loop (see
`examples/notebooks/00_pennylane_quickstart.ipynb` for a full training
example), or use it as above with fixed/untrained parameters to try the
uncertainty-quantification workflow itself.

## `ShotBootstrap`: measuring finite-shot uncertainty

```python
uq_model = predictor.with_uq(ShotBootstrap(n_samples=16, shots=1000, seed=0))
dist = uq_model.predict_dist(np.random.randn(4, 2))
print(dist.mean.shape, dist.std.shape)  # (4, 2) (4, 2)
```

`ShotBootstrap` repeats the measurement `n_samples` times at the given shot
count and reports the mean and spread across repeats -- `dist.std` is how
much a single prediction would wobble if you re-ran the circuit with that
shot budget. See
[Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md) for the
statistics behind it and how it scales with shot count.

## Calibration and predictive uncertainty

`dist.mean` is a predicted-probability array like any other classifier
output, so QuantumUQ's metrics apply directly:

```python
from quantumuq import ece, predictive_entropy

y_true = np.array([0, 1, 0, 1])  # your real labels
print("ECE:", ece(y_true, dist.mean))
print("Predictive entropy:", predictive_entropy(dist.mean))
```

`ece` answers whether the model's confidence matches its accuracy;
`predictive_entropy` answers how spread out an individual prediction is,
regardless of whether it's correct. These are different questions -- see
[Calibration in Quantum Machine Learning](qml_calibration.md), and the
[PennyLane Community Demo notebook](https://github.com/ocatak/QuantumUQ/blob/main/examples/notebooks/08_pennylane_community_demo.ipynb)
for a worked example showing that sweeping shot count moves the uncertainty
estimate but barely moves calibration.

## Learn more

- [Uncertainty Quantification in Quantum Machine Learning](qml_uncertainty_quantification.md)
- [Calibration in Quantum Machine Learning](qml_calibration.md)
- [Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md)
- [Reliability Benchmarking](qml_reliability_benchmarking.md)

The corresponding runnable notebook is
`examples/notebooks/00_pennylane_quickstart.ipynb` in the
[GitHub repository](https://github.com/ocatak/QuantumUQ/tree/main/examples/notebooks).
