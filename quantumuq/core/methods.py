from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .predictors import PredictiveDistribution, Predictor, UncertaintyMethod, stack_ensemble_samples


@dataclass
class ShotBootstrap(UncertaintyMethod):
    """Repeated forward passes with (optionally) varying shots.

    This method is fully model-agnostic and works for both PennyLane and Qiskit
    predictors, as long as they implement the :class:`Predictor` protocol.
    """

    n_samples: int
    shots: Optional[int] = None
    shots_jitter: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self._rng = np.random.default_rng(self.seed)

    def _sample_shots(self, base_shots: Optional[int]) -> Optional[int]:
        if base_shots is None and self.shots is None:
            return None
        base = self.shots if self.shots is not None else base_shots
        if base is None:
            return None
        if self.shots_jitter is None or self.shots_jitter <= 0:
            return base
        low = max(1, base - self.shots_jitter)
        high = base + self.shots_jitter + 1
        return int(self._rng.integers(low, high))

    def __call__(
        self, predictor: Predictor, X: np.ndarray, shots: Optional[int] = None
    ) -> PredictiveDistribution:
        X_arr = np.asarray(X)
        samples: List[np.ndarray] = []
        for _ in range(self.n_samples):
            sample_shots = self._sample_shots(shots)
            if predictor.task == "classification":
                y = predictor.predict_proba(X_arr, shots=sample_shots)
            else:
                y = predictor.predict(X_arr, shots=sample_shots)
            samples.append(np.asarray(y))
        stacked = np.stack(samples, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        return PredictiveDistribution(samples=stacked, mean=mean, std=std)


@dataclass
class DeepEnsemble(UncertaintyMethod):
    """Deep ensemble over a list of already trained predictors."""

    predictors: Sequence[Predictor]

    def __post_init__(self) -> None:
        if not self.predictors:
            raise ValueError("predictors list must not be empty")
        first_task = self.predictors[0].task
        if any(p.task != first_task for p in self.predictors[1:]):
            raise ValueError("all predictors in an ensemble must share the same task")

    def __call__(
        self, predictor: Predictor, X: np.ndarray, shots: Optional[int] = None
    ) -> PredictiveDistribution:
        # Ignore the predictor argument and use the ensemble instead.
        X_arr = np.asarray(X)
        per_model: List[np.ndarray] = []
        if self.predictors[0].task == "classification":
            for p in self.predictors:
                per_model.append(np.asarray(p.predict_proba(X_arr, shots=shots)))
        else:
            for p in self.predictors:
                per_model.append(np.asarray(p.predict(X_arr, shots=shots)))
        return stack_ensemble_samples(per_model)


@dataclass
class NoiseProfile:
    """Probe prediction stability as a function of shots.

    For each value in ``sweep_shots``, the predictor is evaluated ``n_repeats``
    times and stability statistics are computed.
    """

    sweep_shots: Sequence[int]
    n_repeats: int = 5

    def __post_init__(self) -> None:
        if not self.sweep_shots:
            raise ValueError("sweep_shots must not be empty")
        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be positive")

    def __call__(self, predictor: Predictor, X: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        if predictor.task != "classification":
            raise ValueError("NoiseProfile currently supports classification predictors only")

        X_arr = np.asarray(X)
        results: Dict[int, Dict[str, np.ndarray]] = {}
        for shots in self.sweep_shots:
            probs_list: List[np.ndarray] = []
            entropies: List[np.ndarray] = []
            for _ in range(self.n_repeats):
                probs = np.asarray(predictor.predict_proba(X_arr, shots=shots))
                probs = np.clip(probs, 1e-12, 1.0)
                probs = probs / probs.sum(axis=-1, keepdims=True)
                probs_list.append(probs)
                ent = -np.sum(probs * np.log(probs), axis=-1)
                entropies.append(ent)

            stacked_probs = np.stack(probs_list, axis=0)  # (R, N, C)
            stacked_ent = np.stack(entropies, axis=0)  # (R, N)

            results[int(shots)] = {
                "mean_prob": stacked_probs.mean(axis=0),
                "std_prob": stacked_probs.std(axis=0),
                "mean_entropy": stacked_ent.mean(axis=0),
                "std_entropy": stacked_ent.std(axis=0),
            }

        return results

