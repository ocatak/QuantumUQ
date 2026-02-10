from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np

from ..core.predictors import Predictor, TaskType, UQModel


PostprocessFn = Callable[[np.ndarray], np.ndarray]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


@dataclass
class _QNodePredictor(Predictor):
    qnode: Any
    task: TaskType
    n_classes: Optional[int] = None
    params: Optional[Any] = None
    postprocess: Optional[PostprocessFn] = None
    batched: bool = False

    def _call_qnode(self, x: np.ndarray, shots: Optional[int]) -> np.ndarray:
        """Call QNode and handle optional shots & params."""

        args: list[Any] = [x]
        if self.params is not None:
            args.append(self.params)

        kwargs: dict[str, Any] = {}
        if shots is not None:
            # Try both keyword and device configuration; fall back gracefully.
            try:
                kwargs["shots"] = shots
            except TypeError:
                kwargs = {}

        try:
            out = self.qnode(*args, **kwargs)
        except TypeError:
            # If QNode does not accept shots kwarg, try without.
            out = self.qnode(*args)
        return np.asarray(out)

    def _prepare_output(self, raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw)
        if self.postprocess is not None:
            arr = np.asarray(self.postprocess(arr))
        if self.task == "classification":
            if arr.ndim == 1:
                arr = arr[None, :]
            if self.n_classes is not None and arr.shape[-1] != self.n_classes:
                raise ValueError(
                    f"Expected {self.n_classes} classes, got shape {arr.shape}"
                )
            # Normalize to probabilities defensively.
            arr = np.clip(arr, 1e-12, None)
            arr = arr / arr.sum(axis=-1, keepdims=True)
        return arr

    def predict_proba(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only valid for classification tasks")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.batched:
            raw = self._call_qnode(X_arr, shots=shots)
            return self._prepare_output(raw)

        outputs = [self._call_qnode(x, shots=shots) for x in X_arr]
        raw = np.stack(outputs, axis=0)
        return self._prepare_output(raw)

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.task == "classification":
            probs = self.predict_proba(X_arr, shots=shots)
            return probs.argmax(axis=-1)

        # Regression: QNode returns scalar or vector expectations.
        if self.batched:
            raw = self._call_qnode(X_arr, shots=shots)
        else:
            outputs = [self._call_qnode(x, shots=shots) for x in X_arr]
            raw = np.stack(outputs, axis=0)
        return np.asarray(raw)

    def with_uq(self, method: Any) -> UQModel:
        """Attach an uncertainty method and return a :class:`UQModel`."""

        return UQModel(self, method)


def wrap_qnode(
    qnode: Any,
    task: Literal["classification", "regression"],
    n_classes: Optional[int] = None,
    params: Optional[Any] = None,
    postprocess: Optional[PostprocessFn] = None,
    batched: bool = False,
) -> _QNodePredictor:
    """Wrap a PennyLane QNode as a QuantumUQ predictor.

    Parameters
    ----------
    qnode:
        PennyLane QNode. For classification it should return probabilities,
        logits, or expectations convertible to probabilities.
    task:
        Either ``"classification"`` or ``"regression"``.
    n_classes:
        Number of classes for classification tasks.
    params:
        Optional trainable parameters; the QNode is expected to have signature
        ``qnode(features, params)`` when this is provided.
    postprocess:
        Optional function mapping raw outputs to probabilities (classification)
        or predictions (regression). If not provided for classification, a
        defensive normalization is applied, or softmax when outputs look like
        logits.
    batched:
        If ``True``, the QNode is expected to handle a batch of inputs at once.
    """

    if task not in ("classification", "regression"):
        raise ValueError('task must be "classification" or "regression"')
    if task == "classification" and n_classes is None:
        raise ValueError("n_classes must be provided for classification tasks")

    # Import lazily to avoid hard dependency at import time.
    try:
        import pennylane as qml  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "wrap_qnode requires PennyLane to be installed: `pip install pennylane`"
        ) from exc

    if task == "classification" and postprocess is None:
        # Default classification behaviour: try probabilities, otherwise softmax.
        def default_postprocess(raw: np.ndarray) -> np.ndarray:
            arr = np.asarray(raw)
            # If already looks like probabilities (non-negative and sum ~ 1).
            if np.all(arr >= -1e-8):
                sums = arr.sum(axis=-1, keepdims=True)
                # Avoid division by zero.
                sums = np.clip(sums, 1e-12, None)
                return arr / sums
            return _softmax(arr)

        postprocess = default_postprocess

    return _QNodePredictor(
        qnode=qnode,
        task=task,
        n_classes=n_classes,
        params=params,
        postprocess=postprocess,
        batched=batched,
    )

