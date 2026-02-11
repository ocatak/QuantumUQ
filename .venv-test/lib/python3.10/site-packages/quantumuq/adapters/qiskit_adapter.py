from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np

from ..core.predictors import Predictor, TaskType, UQModel

FeatureMapFn = Callable[[np.ndarray], Sequence[Sequence[float]]]
PostprocessFn = Callable[[np.ndarray], np.ndarray]
BitstringToClassFn = Callable[[str | int], int]


def _import_qiskit_primitives() -> Any:
    try:
        from qiskit.primitives import BaseEstimator, BaseSampler  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Qiskit primitives are required. Install with `pip install qiskit`."
        ) from exc
    return BaseSampler, BaseEstimator


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _default_bitstring_to_class(bitstring: str | int, n_classes: int) -> int:
    if isinstance(bitstring, str):
        idx = int(bitstring, 2)
    else:
        idx = int(bitstring)
    return idx % n_classes


@dataclass
class _QiskitSamplerPredictor(Predictor):
    sampler: Any
    circuit: Any
    task: TaskType
    n_classes: int
    params: Optional[Any] = None
    feature_map: Optional[FeatureMapFn] = None
    bitstring_to_class: Optional[BitstringToClassFn] = None

    def _run(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> List[Dict[str | int, float]]:
        BaseSampler, _ = _import_qiskit_primitives()
        if not isinstance(self.sampler, BaseSampler):  # pragma: no cover - defensive
            raise TypeError("sampler must be a Qiskit BaseSampler or compatible")

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.feature_map is None:
            raise ValueError("feature_map must be provided for wrap_qiskit_sampler")
        parameter_values = list(self.feature_map(X_arr))

        circuits = [self.circuit] * len(parameter_values)
        run_kwargs: Dict[str, Any] = {}
        if shots is not None:
            run_kwargs["shots"] = int(shots)

        try:
            job = self.sampler.run(
                circuits=circuits,
                parameter_values=parameter_values,
                **run_kwargs,
            )
        except TypeError:
            # Fallback without explicit shots argument.
            job = self.sampler.run(
                circuits=circuits,
                parameter_values=parameter_values,
            )
        result = job.result()

        # Support both legacy and modern attributes.
        quasi_list: Iterable[Any]
        if hasattr(result, "quasi_dists"):
            quasi_list = result.quasi_dists
        elif hasattr(result, "quasi_distributions"):
            quasi_list = result.quasi_distributions
        else:  # pragma: no cover - version guard
            raise RuntimeError("Unsupported Sampler result type")

        return [dict(q) for q in quasi_list]

    def predict_proba(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only valid for classification tasks")

        quasi_list = self._run(X, shots=shots)
        probs = []
        for quasi in quasi_list:
            vec = np.zeros(self.n_classes, dtype=float)
            for bit, p in quasi.items():
                if self.bitstring_to_class is not None:
                    cls = self.bitstring_to_class(bit)
                else:
                    cls = _default_bitstring_to_class(bit, self.n_classes)
                if 0 <= cls < self.n_classes:
                    vec[cls] += float(p)
            # Normalize defensively.
            s = vec.sum()
            if s <= 0.0:
                vec[:] = 1.0 / self.n_classes
            else:
                vec /= s
            probs.append(vec)
        return np.vstack(probs)

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        probs = self.predict_proba(X, shots=shots)
        return probs.argmax(axis=-1)

    def with_uq(self, method: Any) -> UQModel:
        return UQModel(self, method)


@dataclass
class _QiskitEstimatorPredictor(Predictor):
    estimator: Any
    circuit: Any
    observables: Sequence[Any]
    task: TaskType
    n_classes: Optional[int] = None
    params: Optional[Any] = None
    feature_map: Optional[FeatureMapFn] = None
    postprocess: Optional[PostprocessFn] = None

    def _run(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:
        _, BaseEstimator = _import_qiskit_primitives()
        if not isinstance(self.estimator, BaseEstimator):  # pragma: no cover
            raise TypeError("estimator must be a Qiskit BaseEstimator or compatible")

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.feature_map is None:
            raise ValueError("feature_map must be provided for wrap_qiskit_estimator")
        parameter_values = list(self.feature_map(X_arr))

        circuits = [self.circuit] * len(parameter_values)
        observables = [self.observables] * len(parameter_values)
        run_kwargs: Dict[str, Any] = {}
        if shots is not None:
            run_kwargs["shots"] = int(shots)

        try:
            job = self.estimator.run(
                circuits=circuits,
                observables=observables,
                parameter_values=parameter_values,
                **run_kwargs,
            )
        except TypeError:
            job = self.estimator.run(
                circuits=circuits,
                observables=observables,
                parameter_values=parameter_values,
            )
        result = job.result()
        # Expect an array of expectation values per observable.
        if hasattr(result, "values"):
            values = np.asarray(result.values)
        elif hasattr(result, "quasi_dists"):  # pragma: no cover - defensive
            values = np.asarray(result.quasi_dists)
        else:  # pragma: no cover
            raise RuntimeError("Unsupported Estimator result type")
        # values shape: (N, C) or (N,) depending on number of observables.
        return values

    def _prepare_output(self, raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw)
        if self.postprocess is not None:
            arr = np.asarray(self.postprocess(arr))
        elif self.task == "classification":
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = _softmax(arr)
        return arr

    def predict_proba(
        self, X: np.ndarray, shots: Optional[int] = None
    ) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only valid for classification tasks")
        raw = self._run(X, shots=shots)
        probs = self._prepare_output(raw)
        if self.n_classes is not None and probs.shape[-1] != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} classes, got shape {probs.shape}"
            )
        return probs

    def predict(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        if self.task == "classification":
            probs = self.predict_proba(X, shots=shots)
            return probs.argmax(axis=-1)
        raw = self._run(X, shots=shots)
        return np.asarray(raw).reshape(-1, *np.asarray(raw).shape[1:])

    def with_uq(self, method: Any) -> UQModel:
        return UQModel(self, method)


def wrap_qiskit_sampler(
    sampler: Any,
    circuit: Any,
    task: Literal["classification"] = "classification",
    n_classes: int = 2,
    params: Optional[Any] = None,
    feature_map: Optional[FeatureMapFn] = None,
    bitstring_to_class: Optional[BitstringToClassFn] = None,
) -> _QiskitSamplerPredictor:
    """Wrap a Qiskit Sampler primitive as a classification predictor."""

    if task != "classification":
        raise ValueError("wrap_qiskit_sampler currently supports classification only")
    if n_classes <= 1:
        raise ValueError("n_classes must be >= 2 for classification")

    # Lazy import to keep import-time surface small.
    _import_qiskit_primitives()

    return _QiskitSamplerPredictor(
        sampler=sampler,
        circuit=circuit,
        task=task,
        n_classes=n_classes,
        params=params,
        feature_map=feature_map,
        bitstring_to_class=bitstring_to_class,
    )


def wrap_qiskit_estimator(
    estimator: Any,
    circuit: Any,
    observables: Sequence[Any],
    task: Literal["classification", "regression"],
    n_classes: Optional[int] = None,
    params: Optional[Any] = None,
    feature_map: Optional[FeatureMapFn] = None,
    postprocess: Optional[PostprocessFn] = None,
) -> _QiskitEstimatorPredictor:
    """Wrap a Qiskit Estimator primitive as a predictor.

    For classification, use one observable per class to obtain logits, and
    apply a softmax by default. For regression, use a single observable per
    data point and return expectations directly.
    """

    if task == "classification" and n_classes is None:
        n_classes = len(observables)
    if task == "classification" and (n_classes is None or n_classes <= 1):
        raise ValueError("n_classes must be >= 2 for classification")

    _import_qiskit_primitives()

    if task == "classification" and postprocess is None:
        postprocess = _softmax

    return _QiskitEstimatorPredictor(
        estimator=estimator,
        circuit=circuit,
        observables=observables,
        task=task,
        n_classes=n_classes,
        params=params,
        feature_map=feature_map,
        postprocess=postprocess,
    )

