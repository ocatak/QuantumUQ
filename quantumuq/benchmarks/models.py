from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

__all__ = ["train_pennylane_vqc", "train_qiskit_vqc"]


def train_pennylane_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_layers: int = 2,
    epochs: int = 25,
    lr: float = 0.2,
    shots: int = 1000,
    seed: int = 0,
) -> Tuple[Any, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Train a small 2-qubit PennyLane variational classifier.

    Assumes ``X_train`` has exactly 2 features -- every benchmark-suite
    dataset (:mod:`quantumuq.benchmarks.datasets`) is reduced to 2 features
    precisely so this one reference circuit architecture applies to all of
    them.

    Returns ``(qnode, params, postprocess)``, ready to pass straight into
    :func:`quantumuq.wrap_qnode`:
    ``wrap_qnode(qnode, task="classification", n_classes=2, params=params,
    postprocess=postprocess)``.
    """

    import pennylane as qml
    import pennylane.numpy as pnp

    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    weights_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=n_layers, n_wires=n_qubits
    )
    rng = np.random.default_rng(seed)
    params = 0.1 * pnp.array(rng.standard_normal(weights_shape))

    @qml.qnode(dev)
    def vqc(features, weights):
        qml.AngleEmbedding(features, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    # 2 qubits -> 4 outcomes (|00>, |01>, |10>, |11>); collapse to 2 classes
    # by grouping on the first qubit.
    def postprocess(probs):
        p = pnp.array(probs)
        if p.ndim == 1:
            return pnp.array([p[0] + p[1], p[2] + p[3]])
        return pnp.stack([p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]], axis=-1)

    # y_train is a constant (not differentiated): plain numpy, not
    # pennylane.numpy. pnp.eye(2)[y_train] silently collapses to a 0-d array
    # under autograd's tracing in some pennylane/autograd/numpy version
    # combinations -- see examples/notebooks/00_pennylane_quickstart.ipynb.
    y_onehot = np.eye(2)[y_train]

    def loss(weights):
        logits = [vqc(x, weights) for x in X_train]
        probs = postprocess(pnp.stack(logits))
        probs = pnp.clip(probs, 1e-12, 1.0)
        return -pnp.mean(pnp.sum(y_onehot * pnp.log(probs), axis=1))

    grad_fn = qml.grad(loss, argnum=0)
    for _ in range(epochs):
        g = grad_fn(params)
        params = params - lr * g

    return vqc, np.asarray(params), postprocess


def train_qiskit_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampler: Optional[Any] = None,
    spsa_iters: int = 30,
    seed: int = 0,
) -> Tuple[Any, Any, Callable[[np.ndarray], list]]:
    """Train a small 2-qubit Qiskit variational classifier via SPSA.

    Assumes ``X_train`` has exactly 2 features, matching
    :func:`train_pennylane_vqc`'s architecture for a fair comparison across
    backends. Qiskit circuits aren't differentiable through this library, so
    training uses SPSA (as in
    ``examples/notebooks/03_qiskit_training_spsa.ipynb``) rather than
    gradients -- each iteration perturbs only the trainable rotation
    parameters, while the two data-encoding parameters are re-bound per
    sample on every evaluation, so training genuinely uses the input
    features (not just a global bias).

    Returns ``(sampler, circuit, feature_map)``, ready to pass straight into
    :func:`quantumuq.wrap_qiskit_sampler`:
    ``wrap_qiskit_sampler(sampler, circuit=circuit, task="classification",
    n_classes=2, feature_map=feature_map)``.
    """

    from qiskit.circuit import ParameterVector, QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    from ..adapters.qiskit_adapter import wrap_qiskit_sampler

    n_qubits = 2
    phi = ParameterVector("phi", 2)  # data encoding
    theta = ParameterVector("theta", 2)  # trainable

    circuit = QuantumCircuit(n_qubits)
    circuit.ry(phi[0], 0)
    circuit.ry(phi[1], 1)
    circuit.cx(0, 1)
    circuit.ry(theta[0], 0)
    circuit.ry(theta[1], 1)
    circuit.measure_all()

    if sampler is None:
        sampler = StatevectorSampler(seed=np.random.default_rng(seed))

    def make_feature_map(theta_vals: np.ndarray) -> Callable[[np.ndarray], list]:
        def feature_map(X_batch: np.ndarray) -> list:
            X_arr = np.atleast_2d(X_batch)
            return [
                [float(x[0]), float(x[1]), float(theta_vals[0]), float(theta_vals[1])]
                for x in X_arr
            ]

        return feature_map

    y_onehot = np.eye(2)[y_train]

    def loss(theta_vals: np.ndarray) -> float:
        predictor = wrap_qiskit_sampler(
            sampler,
            circuit=circuit,
            task="classification",
            n_classes=2,
            feature_map=make_feature_map(theta_vals),
        )
        probs = predictor.predict_proba(X_train, shots=500)
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))

    rng = np.random.default_rng(seed)
    # A zero (or otherwise symmetric) initialization leaves the two class
    # labels indistinguishable to the loss -- SPSA can converge to a
    # decision boundary with the labels consistently swapped, which looks
    # like a below-chance accuracy despite the model having genuinely
    # learned. A small random initial perturbation breaks that symmetry.
    theta_vals = 0.1 * rng.standard_normal(2)
    for k in range(spsa_iters):
        a = 0.3 / (k + 1) ** 0.602
        c = 0.2 / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=2)
        l_plus = loss(theta_vals + c * delta)
        l_minus = loss(theta_vals - c * delta)
        g_hat = (l_plus - l_minus) / (2 * c * delta)
        theta_vals = theta_vals - a * g_hat

    return sampler, circuit, make_feature_map(theta_vals)
