from __future__ import annotations

"""Lightweight CLI smoke tests for QuantumUQ.

This script intentionally avoids depending on PennyLane or Qiskit so it can
run even in minimal environments. It exercises the core abstractions and
methods using a simple in-memory predictor.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantumuq.core.metrics import ece, nll, rmse
from quantumuq.core.methods import DeepEnsemble, ShotBootstrap
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


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 2))
    y = (X[:, 0] > 0).astype(int)

    base = _ToyClassifier()
    method = ShotBootstrap(n_samples=8, shots=100, seed=0)
    model = UQModel(base, method)
    dist = model.predict_dist(X)

    y_pred = dist.mean.argmax(axis=1)
    cls_nll = nll(y, dist.mean)
    cls_ece = ece(y, dist.mean)

    # Regression-style RMSE on a toy target.
    y_reg = X[:, 0]
    y_hat = dist.mean[:, 0]
    reg_rmse = rmse(y_reg, y_hat)

    # Simple deep ensemble using two independent classifiers.
    ens_method = DeepEnsemble([_ToyClassifier(), _ToyClassifier()])
    ens_model = UQModel(base, ens_method)
    ens_dist = ens_model.predict_dist(X)

    print("QuantumUQ smoke tests:")
    print(f"  y_pred shape       : {y_pred.shape}")
    print(f"  dist.mean shape    : {dist.mean.shape}")
    print(f"  dist.std shape     : {dist.std.shape}")
    print(f"  NLL (toy)          : {cls_nll:.4f}")
    print(f"  ECE (toy)          : {cls_ece:.4f}")
    print(f"  RMSE (toy)         : {reg_rmse:.4f}")
    print(f"  Ensemble mean shape: {ens_dist.mean.shape}")


if __name__ == "__main__":
    main()

