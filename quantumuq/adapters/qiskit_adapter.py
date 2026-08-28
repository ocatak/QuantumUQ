from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np

from ..core.predictors import Predictor, TaskType, UQModel

FeatureMapFn = Callable[[np.ndarray], Sequence[Sequence[float]]]
PostprocessFn = Callable[[np.ndarray], np.ndarray]
BitstringToClassFn = Callable[[str | int], int]


def _import_qiskit_primitives() -> Any:
    try:
        from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Qiskit V2 primitives are required (BaseSamplerV2/BaseEstimatorV2, "
            "available since qiskit>=1.0). Install/upgrade with `pip install "
            "-U qiskit`. Pass a V2 primitive such as StatevectorSampler, "
            "StatevectorEstimator, BackendSamplerV2, or BackendEstimatorV2."
        ) from exc
    return BaseSamplerV2, BaseEstimatorV2


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

    def _run(self, X: np.ndarray, shots: Optional[int] = None) -> List[Dict[int, int]]:
        BaseSamplerV2, _ = _import_qiskit_primitives()
        if not isinstance(self.sampler, BaseSamplerV2):  # pragma: no cover - defensive
            raise TypeError(
                "sampler must be a Qiskit BaseSamplerV2 (V2 primitive) or compatible, "
                "e.g. StatevectorSampler or BackendSamplerV2"
            )

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.feature_map is None:
            raise ValueError("feature_map must be provided for wrap_qiskit_sampler")
        parameter_values = list(self.feature_map(X_arr))

        pubs = [(self.circuit, pv) for pv in parameter_values]
        run_kwargs: Dict[str, Any] = {}
        if shots is not None:
            run_kwargs["shots"] = int(shots)

        job = self.sampler.run(pubs, **run_kwargs)
        result = job.result()

        counts_list: List[Dict[int, int]] = []
        for pub_result in result:
            data_items = list(pub_result.data.items())
            if not data_items:
                raise RuntimeError(
                    "Sampler result has no classical register data; does the "
                    "circuit include measurements (e.g. circuit.measure_all())?"
                )
            _, bit_array = data_items[0]
            counts_list.append(bit_array.get_int_counts())
        return counts_list

    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only valid for classification tasks")

        counts_list = self._run(X, shots=shots)
        probs = []
        for counts in counts_list:
            vec = np.zeros(self.n_classes, dtype=float)
            for bit, count in counts.items():
                if self.bitstring_to_class is not None:
                    cls = self.bitstring_to_class(bit)
                else:
                    cls = _default_bitstring_to_class(bit, self.n_classes)
                if 0 <= cls < self.n_classes:
                    vec[cls] += float(count)
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

    def _run(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        _, BaseEstimatorV2 = _import_qiskit_primitives()
        if not isinstance(self.estimator, BaseEstimatorV2):  # pragma: no cover
            raise TypeError(
                "estimator must be a Qiskit BaseEstimatorV2 (V2 primitive) or "
                "compatible, e.g. StatevectorEstimator or BackendEstimatorV2"
            )

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if self.feature_map is None:
            raise ValueError("feature_map must be provided for wrap_qiskit_estimator")
        parameter_values = list(self.feature_map(X_arr))

        pubs = [(self.circuit, self.observables, pv) for pv in parameter_values]
        run_kwargs: Dict[str, Any] = {}
        if shots is not None:
            # V2 estimators target a standard error ("precision") instead of a
            # literal shot count. For a Pauli observable (eigenvalues +/-1),
            # the standard error over `shots` samples is at most 1/sqrt(shots),
            # which is the conversion Qiskit's own migration guide recommends.
            run_kwargs["precision"] = 1.0 / float(np.sqrt(shots))

        job = self.estimator.run(pubs, **run_kwargs)
        result = job.result()
        # Each pub's `.data.evs` has shape (len(observables),); stacking over
        # pubs gives (N, len(observables)).
        values = np.stack([pub_result.data.evs for pub_result in result], axis=0)
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

    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
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
    """Wrap a Qiskit V2 Sampler primitive as a classification predictor.

    ``sampler`` must implement ``BaseSamplerV2`` (e.g. ``StatevectorSampler``,
    ``BackendSamplerV2``, or a hardware-backed V2 sampler). V1 primitives
    (``qiskit.primitives.Sampler``) are not supported.
    """

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
    """Wrap a Qiskit V2 Estimator primitive as a predictor.

    ``estimator`` must implement ``BaseEstimatorV2`` (e.g.
    ``StatevectorEstimator``, ``BackendEstimatorV2``, or a hardware-backed V2
    estimator). V1 primitives (``qiskit.primitives.Estimator``) are not
    supported.

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
