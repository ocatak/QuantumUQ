from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from quantumuq import ShotBootstrap, wrap_qiskit_estimator, wrap_qiskit_sampler


def _feature_map(X: np.ndarray):
    return [[float(x[0])] for x in np.atleast_2d(X)]


def _rotation_circuit(measure: bool) -> QuantumCircuit:
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    if measure:
        qc.measure_all()
    return qc


def test_wrap_qiskit_sampler_classification_roundtrip() -> None:
    circuit = _rotation_circuit(measure=True)
    sampler = StatevectorSampler(seed=np.random.default_rng(0))
    predictor = wrap_qiskit_sampler(
        sampler,
        circuit=circuit,
        task="classification",
        n_classes=2,
        feature_map=_feature_map,
    )

    X = np.array([[0.1], [1.5], [3.0]])
    probs = predictor.predict_proba(X, shots=1000)

    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)


def test_wrap_qiskit_sampler_with_shot_bootstrap_has_nonzero_variance() -> None:
    circuit = _rotation_circuit(measure=True)
    # A plain int seed makes StatevectorSampler reset to the same state on
    # every .run() call, which would make ShotBootstrap variance zero -- use
    # a Generator instance so state advances across calls.
    sampler = StatevectorSampler(seed=np.random.default_rng(0))
    predictor = wrap_qiskit_sampler(
        sampler,
        circuit=circuit,
        task="classification",
        n_classes=2,
        feature_map=_feature_map,
    )
    model = predictor.with_uq(ShotBootstrap(n_samples=8, shots=200, seed=0))

    dist = model.predict_dist(np.array([[1.0]]))
    assert dist.std.max() > 0.0


def test_wrap_qiskit_sampler_rejects_v1_style_object() -> None:
    class NotAV2Sampler:
        pass

    circuit = _rotation_circuit(measure=True)
    predictor = wrap_qiskit_sampler(
        NotAV2Sampler(),
        circuit=circuit,
        task="classification",
        n_classes=2,
        feature_map=_feature_map,
    )
    with pytest.raises(TypeError, match="BaseSamplerV2"):
        predictor.predict_proba(np.array([[0.5]]))


def test_wrap_qiskit_estimator_classification_roundtrip() -> None:
    circuit = _rotation_circuit(measure=False)
    observables = [SparsePauliOp("Z"), SparsePauliOp("X")]
    estimator = StatevectorEstimator(seed=np.random.default_rng(0))
    predictor = wrap_qiskit_estimator(
        estimator,
        circuit=circuit,
        observables=observables,
        task="classification",
        n_classes=2,
        feature_map=_feature_map,
    )

    X = np.array([[0.2], [2.0]])
    probs = predictor.predict_proba(X, shots=1000)

    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_wrap_qiskit_estimator_regression() -> None:
    circuit = _rotation_circuit(measure=False)
    estimator = StatevectorEstimator(seed=np.random.default_rng(0))
    predictor = wrap_qiskit_estimator(
        estimator,
        circuit=circuit,
        observables=[SparsePauliOp("Z")],
        task="regression",
        feature_map=_feature_map,
    )

    X = np.array([[0.0], [np.pi]])
    y = predictor.predict(X, shots=1000)

    assert y.shape == (2, 1)
    # RY(0) leaves the qubit in |0>, so <Z> should be close to +1.
    assert y[0, 0] > 0.9
    # RY(pi) flips the qubit to |1>, so <Z> should be close to -1.
    assert y[1, 0] < -0.9
