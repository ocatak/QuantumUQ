QuantumUQ
=========

Uncertainty Quantification for Quantum Machine Learning on **PennyLane** and **Qiskit**.

### Installation

```bash
pip install quantumuq
# With optional Qiskit Aer support:
pip install "quantumuq[aer]"
```

### Quick examples

PennyLane:

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

params = 0.1 * np.random.default_rng(0).standard_normal((1, 2, 3))
predictor = wrap_qnode(circuit, task="classification", n_classes=2, params=params)
uq = ShotBootstrap(n_samples=16, shots=1000, seed=0)
uq_model = predictor.with_uq(uq)
dist = uq_model.predict_dist(np.random.randn(4, 2))
print(dist.mean.shape, dist.std.shape)
```

Qiskit:

```python
from quantumuq import wrap_qiskit_sampler, ShotBootstrap
from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(1)
qc.ry(0.0, 0)
qc.measure_all()

def feature_map(X: np.ndarray):
    return [[float(x[0])] for x in np.atleast_2d(X)]

sampler = Sampler()
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

### Methods & metrics

- **Uncertainty methods**: `ShotBootstrap`, `DeepEnsemble`, `NoiseProfile`
- **Metrics (classification)**: `nll`, `brier`, `ece`, `predictive_entropy`
- **Metrics (regression)**: `rmse`, `gaussian_nll`

### Roadmap (v0.2 ideas)

- Richer model adapters (more flexible outputs, calibration hooks)
- Additional metrics and visualization utilities
- Optional integrations with experiment tracking tools

### License

MIT License. See `LICENSE` for details.

### Citation

If you use QuantumUQ in academic work, please cite:

```bibtex
@misc{quantumuq2026,
  title        = {QuantumUQ: Uncertainty Quantification for Quantum Machine Learning},
  author       = {Catak, Ferhat Ozgur},
  year         = {2026},
  note         = {Python library, version 0.1.0},
  howpublished = {\url{https://github.com/ocatak/QuantumUQ}}
}
```

