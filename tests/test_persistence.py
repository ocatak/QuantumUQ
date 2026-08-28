from __future__ import annotations

import json

import numpy as np
import pennylane as qml
import pytest

from quantumuq import ShotBootstrap, wrap_qnode
from quantumuq.adapters.qiskit_adapter import (
    _QiskitEstimatorPredictor,
    _QiskitSamplerPredictor,
)
from quantumuq.core.methods import DeepEnsemble, NoiseProfile
from quantumuq.core.predictors import UQModel, _predictor_to_checkpoint


def _pennylane_circuit():
    dev = qml.device("default.qubit", wires=1, shots=1000)

    @qml.qnode(dev)
    def circuit(x, params):
        qml.RY(x[0] * params[0], wires=0)
        return qml.probs(wires=0)

    return circuit


def test_save_load_pennylane_roundtrip(tmp_path) -> None:
    circuit = _pennylane_circuit()
    params = np.array([0.7])
    predictor = wrap_qnode(circuit, task="classification", n_classes=2, params=params)
    model = predictor.with_uq(ShotBootstrap(n_samples=2, shots=1000, seed=0))

    ckpt_path = tmp_path / "model.json"
    model.save(ckpt_path)

    checkpoint = json.loads(ckpt_path.read_text())
    assert checkpoint["predictor"]["backend"] == "pennylane"
    assert checkpoint["predictor"]["params"] == [0.7]
    assert checkpoint["method"] == {
        "type": "ShotBootstrap",
        "config": {"n_samples": 2, "shots": 1000, "shots_jitter": None, "seed": 0},
    }

    loaded = UQModel.load(ckpt_path, qnode=circuit)
    assert np.allclose(loaded.base_predictor.params, params)
    assert loaded.method.n_samples == 2
    assert loaded.method.shots == 1000
    assert loaded.method.seed == 0

    X = np.array([[0.3]])
    dist = loaded.predict_dist(X)
    assert dist.mean.shape == (1, 2)


def test_load_pennylane_without_qnode_raises(tmp_path) -> None:
    circuit = _pennylane_circuit()
    predictor = wrap_qnode(
        circuit, task="classification", n_classes=2, params=np.array([0.1])
    )
    model = predictor.with_uq(ShotBootstrap(n_samples=1, shots=100, seed=0))
    ckpt_path = tmp_path / "model.json"
    model.save(ckpt_path)

    with pytest.raises(ValueError, match="requires qnode"):
        UQModel.load(ckpt_path)


def test_save_deep_ensemble_raises(tmp_path) -> None:
    member = _QiskitSamplerPredictor(
        sampler=object(), circuit=object(), task="classification", n_classes=2
    )
    model = UQModel(member, DeepEnsemble(predictors=[member, member]))

    with pytest.raises(TypeError, match="DeepEnsemble"):
        model.save(tmp_path / "model.json")


def test_qiskit_sampler_checkpoint_serialization() -> None:
    predictor = _QiskitSamplerPredictor(
        sampler=object(),
        circuit=object(),
        task="classification",
        n_classes=2,
        params=np.array([1.0, 2.0]),
    )
    checkpoint = _predictor_to_checkpoint(predictor)
    assert checkpoint == {
        "backend": "qiskit_sampler",
        "task": "classification",
        "n_classes": 2,
        "params": [1.0, 2.0],
    }


def test_qiskit_estimator_checkpoint_serialization() -> None:
    predictor = _QiskitEstimatorPredictor(
        estimator=object(),
        circuit=object(),
        observables=[object()],
        task="regression",
        n_classes=None,
        params=None,
    )
    checkpoint = _predictor_to_checkpoint(predictor)
    assert checkpoint == {
        "backend": "qiskit_estimator",
        "task": "regression",
        "n_classes": None,
        "params": None,
    }


def test_noise_profile_checkpoint_roundtrip(tmp_path) -> None:
    circuit = _pennylane_circuit()
    predictor = wrap_qnode(
        circuit, task="classification", n_classes=2, params=np.array([0.2])
    )
    model = predictor.with_uq(NoiseProfile(sweep_shots=[100, 500], n_repeats=3))

    ckpt_path = tmp_path / "model.json"
    model.save(ckpt_path)
    loaded = UQModel.load(ckpt_path, qnode=circuit)

    assert loaded.method.sweep_shots == [100, 500]
    assert loaded.method.n_repeats == 3
