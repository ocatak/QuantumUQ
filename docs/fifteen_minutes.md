---
title: Quantum Machine Learning Uncertainty Quantification in 15 Minutes
description: >-
  A practical tutorial for adding uncertainty quantification, calibration,
  and predictive-entropy metrics to Qiskit and PennyLane quantum machine
  learning models with QuantumUQ.
---

# Try QuantumUQ in 15 Minutes

## 1. Install

```bash
pip install quantumuq

# You'll also need whichever framework your model uses:
pip install pennylane
# or: pip install qiskit
```

## 2. Run one example

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
uq_model = predictor.with_uq(ShotBootstrap(n_samples=16, shots=1000, seed=0))

X = np.random.default_rng(1).standard_normal((4, 2))
dist = uq_model.predict_dist(X)
print(dist.mean.shape, dist.std.shape)  # (4, 2) (4, 2)
```

`predictor` above wraps an **untrained** circuit purely so this runs standalone
in a few seconds -- swap in your own trained `params` and this is a real
predictive distribution.

## 3. Add your own Qiskit/PennyLane model

Wrap whatever you already have instead of the toy circuit above:

```python
# PennyLane: any QNode you've already built and trained.
predictor = wrap_qnode(
    my_qnode, task="classification", n_classes=2, params=my_trained_params
)
```

```python
# Qiskit: a V2 Sampler/Estimator primitive + your circuit.
from quantumuq import wrap_qiskit_sampler

predictor = wrap_qiskit_sampler(
    my_sampler,  # a BaseSamplerV2, e.g. StatevectorSampler
    circuit=my_circuit,
    task="classification",
    n_classes=2,
    feature_map=my_feature_map,
)
```

Same `with_uq(...)` / `predict_dist(...)` calls as above, regardless of which
backend or how the model was trained. See the
[PennyLane](quickstart_pennylane.md) and [Qiskit](quickstart_qiskit.md)
quickstarts for the full trained-model versions, or `examples/notebooks/` for
runnable ones, including a
[Qiskit V2 primitives guide](examples/07_qiskit_v2_primitives.ipynb) covering
a seeding gotcha that will otherwise silently zero out your uncertainty
estimates.

## 4. Produce uncertainty + ECE + predictive entropy

```python
from quantumuq import ece, predictive_entropy

y_true = np.array([0, 1, 0, 1])  # your real labels

print("ECE:", ece(y_true, dist.mean))
print("Predictive entropy:", predictive_entropy(dist.mean))
print("Per-sample uncertainty (std):", dist.std)
```

`ece` tells you whether the model's confidence matches its accuracy;
`predictive_entropy` and `dist.std` tell you how uncertain each individual
prediction is. They answer different questions -- see
[Calibration in Quantum Machine Learning](qml_calibration.md), or
[this notebook](examples/08_pennylane_community_demo.ipynb) for why shot
count moves one but not the other.

Want a single number across a shot-count sweep instead of writing this by
hand? `quantumuq.benchmarks.run_benchmark(...)` (or the `quantumuq-benchmark`
CLI) does exactly this end to end on a reference model -- see
[Reliability Benchmarking](qml_reliability_benchmarking.md).

## 5. Send feedback / open an issue

Found a bug, a confusing error message, or something this page didn't cover?

- [Open an issue](https://github.com/ocatak/QuantumUQ/issues/new)
- [Start a discussion](https://github.com/ocatak/QuantumUQ/discussions)

---

> **Using QuantumUQ in your research?**
>
> I'd be very interested to hear about your use case. Open a
> [Discussion](https://github.com/ocatak/QuantumUQ/discussions) or
> [Issue](https://github.com/ocatak/QuantumUQ/issues), or contact me directly
> at f.ozgur.catak@uis.no.
