from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

TaskType = Literal["classification", "regression"]


@runtime_checkable
class Predictor(Protocol):
    """Protocol for quantum predictors used by UQ methods.

    Implementations must expose:

    - ``task``: either ``"classification"`` or ``"regression"``.
    - ``predict(X, shots=None)``: point predictions.
    - ``predict_proba(X, shots=None)``: class probabilities for classification.
    """

    task: TaskType

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class PredictiveDistribution:
    """Container for predictive samples and summary statistics.

    Attributes
    ----------
    samples:
        Array of samples with shape ``(S, N, C)`` for classification or
        ``(S, N, D)`` for regression.
    mean:
        Mean over the sample dimension, shape ``(N, C)`` or ``(N, D)``.
    std:
        Standard deviation over the sample dimension, same shape as ``mean``.
    """

    samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def interval(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Return central prediction interval for given ``alpha``.

        Parameters
        ----------
        alpha:
            Confidence level in (0, 1). E.g. ``0.95`` for a 95% interval.
        """

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q
        lower = np.quantile(self.samples, lower_q, axis=0)
        upper = np.quantile(self.samples, upper_q, axis=0)
        return lower, upper

    def entropy(self) -> np.ndarray:
        """Predictive entropy for classification tasks.

        Uses the mean class probabilities over samples and returns entropy
        per data point with shape ``(N,)``.
        """

        # Assume last dimension is class dimension.
        mean_probs = self.mean
        # Normalize defensively.
        mean_probs = np.clip(mean_probs, 1e-12, 1.0)
        mean_probs = mean_probs / mean_probs.sum(axis=-1, keepdims=True)
        return -np.sum(mean_probs * np.log(mean_probs), axis=-1)


class UQModel:
    """Wrap a base predictor with an uncertainty method.

    Parameters
    ----------
    base_predictor:
        Object implementing the :class:`Predictor` protocol.
    method:
        Callable that given ``(predictor, X, shots)`` returns a
        :class:`PredictiveDistribution`.
    """

    def __init__(self, base_predictor: Predictor, method: "UncertaintyMethod") -> None:
        self.base_predictor = base_predictor
        self.method = method

    @property
    def task(self) -> TaskType:
        return self.base_predictor.task

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        return self.base_predictor.predict(X, shots=shots)

    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        return self.base_predictor.predict_proba(X, shots=shots)

    def predict_dist(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> PredictiveDistribution:
        return self.method(self.base_predictor, X, shots=shots)


class UncertaintyMethod(Protocol):
    """Protocol for uncertainty methods compatible with :class:`UQModel`."""

    def __call__(
        self, predictor: Predictor, X: np.ndarray, shots: Optional[int] = None
    ) -> PredictiveDistribution:  # pragma: no cover - protocol
        ...


def stack_ensemble_samples(samples: Sequence[np.ndarray]) -> PredictiveDistribution:
    """Utility to convert a sequence of per-model predictions into a distribution.

    Parameters
    ----------
    samples:
        Sequence of arrays of identical shape ``(N, C)`` or ``(N, D)``.
    """

    arr = np.stack(samples, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return PredictiveDistribution(samples=arr, mean=mean, std=std)

