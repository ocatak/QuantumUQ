from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

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

    def predict(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def predict_proba(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:  # pragma: no cover - protocol
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

    def save(self, path: Union[str, Path]) -> None:
        """Save trained parameters and method config to ``path`` as JSON.

        Only plain data is written: the predictor's task/params and the
        uncertainty method's configuration. Live backend objects -- the
        PennyLane QNode, the Qiskit circuit/Sampler/Estimator, and any
        user-supplied ``feature_map``/``postprocess``/``bitstring_to_class``
        callables -- are not serialized. Recreate those yourself and pass
        them back into :meth:`load`.

        Not supported for a :class:`~quantumuq.core.methods.DeepEnsemble`
        method, since each ensemble member is itself a live predictor.
        """

        checkpoint = {
            "quantumuq_checkpoint_version": 1,
            "predictor": _predictor_to_checkpoint(self.base_predictor),
            "method": _method_to_checkpoint(self.method),
        }
        Path(path).write_text(json.dumps(checkpoint, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path], **backend_kwargs: Any) -> "UQModel":
        """Reconstruct a UQModel from a checkpoint written by :meth:`save`.

        Pass the live backend object(s) matching the checkpoint's predictor
        backend as keyword arguments:

        - PennyLane: ``qnode=...`` (optionally ``postprocess=...``, ``batched=...``)
        - Qiskit Sampler: ``sampler=...``, ``circuit=...`` (optionally
          ``feature_map=...``, ``bitstring_to_class=...``)
        - Qiskit Estimator: ``estimator=...``, ``circuit=...``,
          ``observables=...`` (optionally ``feature_map=...``, ``postprocess=...``)
        """

        checkpoint = json.loads(Path(path).read_text())
        predictor = _predictor_from_checkpoint(
            checkpoint["predictor"], **backend_kwargs
        )
        method = _method_from_checkpoint(checkpoint["method"])
        return cls(predictor, method)


class UncertaintyMethod(Protocol):
    """Protocol for uncertainty methods compatible with :class:`UQModel`."""

    def __call__(
        self, predictor: Predictor, X: np.ndarray, shots: Optional[int] = None
    ) -> PredictiveDistribution:  # pragma: no cover - protocol
        ...


def _predictor_to_checkpoint(predictor: Predictor) -> Dict[str, Any]:
    from ..adapters.pennylane_adapter import _QNodePredictor
    from ..adapters.qiskit_adapter import (
        _QiskitEstimatorPredictor,
        _QiskitSamplerPredictor,
    )

    def _params(p: Optional[Any]) -> Optional[list]:
        return None if p is None else np.asarray(p).tolist()

    if isinstance(predictor, _QNodePredictor):
        return {
            "backend": "pennylane",
            "task": predictor.task,
            "n_classes": predictor.n_classes,
            "batched": predictor.batched,
            "params": _params(predictor.params),
        }
    if isinstance(predictor, _QiskitSamplerPredictor):
        return {
            "backend": "qiskit_sampler",
            "task": predictor.task,
            "n_classes": predictor.n_classes,
            "params": _params(predictor.params),
        }
    if isinstance(predictor, _QiskitEstimatorPredictor):
        return {
            "backend": "qiskit_estimator",
            "task": predictor.task,
            "n_classes": predictor.n_classes,
            "params": _params(predictor.params),
        }
    raise TypeError(
        f"UQModel.save does not support predictor type {type(predictor).__name__}; "
        "only wrap_qnode/wrap_qiskit_sampler/wrap_qiskit_estimator predictors "
        "are supported."
    )


def _predictor_from_checkpoint(
    checkpoint: Dict[str, Any], **backend_kwargs: Any
) -> Predictor:
    from ..adapters.pennylane_adapter import wrap_qnode
    from ..adapters.qiskit_adapter import wrap_qiskit_estimator, wrap_qiskit_sampler

    backend = checkpoint["backend"]
    params = None if checkpoint["params"] is None else np.array(checkpoint["params"])

    if backend == "pennylane":
        qnode = backend_kwargs.get("qnode")
        if qnode is None:
            raise ValueError(
                "UQModel.load requires qnode=... for a pennylane checkpoint"
            )
        batched = backend_kwargs.get("batched")
        return wrap_qnode(
            qnode,
            task=checkpoint["task"],
            n_classes=checkpoint["n_classes"],
            params=params,
            postprocess=backend_kwargs.get("postprocess"),
            batched=checkpoint["batched"] if batched is None else batched,
        )

    if backend == "qiskit_sampler":
        sampler = backend_kwargs.get("sampler")
        circuit = backend_kwargs.get("circuit")
        if sampler is None or circuit is None:
            raise ValueError(
                "UQModel.load requires sampler=... and circuit=... for a "
                "qiskit_sampler checkpoint"
            )
        return wrap_qiskit_sampler(
            sampler,
            circuit=circuit,
            task=checkpoint["task"],
            n_classes=checkpoint["n_classes"],
            params=params,
            feature_map=backend_kwargs.get("feature_map"),
            bitstring_to_class=backend_kwargs.get("bitstring_to_class"),
        )

    if backend == "qiskit_estimator":
        estimator = backend_kwargs.get("estimator")
        circuit = backend_kwargs.get("circuit")
        observables = backend_kwargs.get("observables")
        if estimator is None or circuit is None or observables is None:
            raise ValueError(
                "UQModel.load requires estimator=..., circuit=..., and "
                "observables=... for a qiskit_estimator checkpoint"
            )
        return wrap_qiskit_estimator(
            estimator,
            circuit=circuit,
            observables=observables,
            task=checkpoint["task"],
            n_classes=checkpoint["n_classes"],
            params=params,
            feature_map=backend_kwargs.get("feature_map"),
            postprocess=backend_kwargs.get("postprocess"),
        )

    raise ValueError(f"Unknown predictor backend in checkpoint: {backend!r}")


def _method_to_checkpoint(method: "UncertaintyMethod") -> Dict[str, Any]:
    from dataclasses import asdict

    from .methods import DeepEnsemble, NoiseProfile, ShotBootstrap

    if isinstance(method, ShotBootstrap):
        return {"type": "ShotBootstrap", "config": asdict(method)}
    if isinstance(method, NoiseProfile):
        return {"type": "NoiseProfile", "config": asdict(method)}
    if isinstance(method, DeepEnsemble):
        raise TypeError(
            "UQModel.save does not support DeepEnsemble: each ensemble member "
            "is itself a live predictor and cannot be captured automatically. "
            "Save/load each member's checkpoint individually and rebuild the "
            "ensemble by hand."
        )
    raise TypeError(
        f"UQModel.save does not support method type {type(method).__name__}"
    )


def _method_from_checkpoint(checkpoint: Dict[str, Any]) -> "UncertaintyMethod":
    from .methods import NoiseProfile, ShotBootstrap

    method_type = checkpoint["type"]
    config = checkpoint["config"]
    if method_type == "ShotBootstrap":
        return ShotBootstrap(**config)
    if method_type == "NoiseProfile":
        return NoiseProfile(**config)
    raise ValueError(f"Unknown uncertainty method type in checkpoint: {method_type!r}")


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
