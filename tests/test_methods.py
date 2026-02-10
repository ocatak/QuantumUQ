from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantumuq.core.methods import DeepEnsemble, NoiseProfile, ShotBootstrap
from quantumuq.core.predictors import Predictor, TaskType, UQModel


@dataclass
class _ToyClassifier(Predictor):
    task: TaskType = "classification"

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        probs = self.predict_proba(X, shots=shots)
        return probs.argmax(axis=-1)

    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        logits = X_arr @ np.array([[1.0], [-1.0]])
        logits = np.concatenate([-logits, logits], axis=1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def test_shot_bootstrap_and_uq_model() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 2))
    base = _ToyClassifier()
    method = ShotBootstrap(n_samples=4, shots=100, seed=0)
    model = UQModel(base, method)

    dist = model.predict_dist(X)
    assert dist.samples.shape[0] == 4
    assert dist.mean.shape == (16, 2)
    assert dist.std.shape == (16, 2)


def test_deep_ensemble() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    predictors = [_ToyClassifier(), _ToyClassifier(), _ToyClassifier()]
    method = DeepEnsemble(predictors)
    base = predictors[0]
    model = UQModel(base, method)
    dist = model.predict_dist(X)
    assert dist.samples.shape[0] == len(predictors)
    assert dist.mean.shape == (8, 2)


def test_noise_profile() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2))
    base = _ToyClassifier()
    profile = NoiseProfile(sweep_shots=[50, 100], n_repeats=3)
    res = profile(base, X)
    assert set(res.keys()) == {50, 100}
    for stats in res.values():
        assert stats["mean_prob"].shape == (4, 2)
        assert stats["std_prob"].shape == (4, 2)
        assert stats["mean_entropy"].shape == (4,)
        assert stats["std_entropy"].shape == (4,)

