from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.methods import ShotBootstrap
from ..core.metrics import brier, ece, nll, predictive_entropy
from .datasets import load_breast_cancer, load_iris, load_moons

__all__ = ["BenchmarkResult", "run_benchmark"]

_DATASET_LOADERS = {
    "moons": load_moons,
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
}
_BACKENDS = ("pennylane", "qiskit")


@dataclass
class BenchmarkResult:
    """One row of `run_benchmark` output: metrics at a single shot count."""

    dataset: str
    backend: str
    shots: int
    accuracy: float
    nll: float
    ece: float
    brier: float
    mean_predictive_entropy: float
    mean_uncertainty: float


def run_benchmark(
    dataset: str,
    backend: str,
    shots_list: Sequence[int] = (100, 500, 1000, 10000),
    n_samples: int = 8,
    test_size: float = 0.3,
    max_samples: int = 150,
    seed: int = 0,
    train_kwargs: Optional[Dict] = None,
) -> List[BenchmarkResult]:
    """Train one reference variational classifier and sweep shot count,
    reporting accuracy and calibration metrics (via `ShotBootstrap`) at each
    shot count.

    This is a lightweight benchmark harness, not a rigorous ML pipeline:

    - Datasets are reduced to 2 features (see
      :mod:`quantumuq.benchmarks.datasets`) so the same small reference
      circuit architecture applies across `"moons"`, `"iris"`, and
      `"breast_cancer"`.
    - Any dataset larger than `max_samples` is randomly subsampled (with
      `seed`) before splitting, so runtime stays consistent and small
      regardless of the source dataset's native size -- e.g. the Breast
      Cancer dataset's 569 samples would otherwise make training far too
      slow for a benchmark meant to run in a few minutes.
    - The train/test split uses a single fixed seed; there is no
      cross-validation.

    Parameters
    ----------
    dataset:
        One of `"moons"`, `"iris"`, `"breast_cancer"`.
    backend:
        One of `"pennylane"`, `"qiskit"`.
    shots_list:
        Shot counts to sweep. At each value, `ShotBootstrap` resamples the
        trained predictor `n_samples` times.
    train_kwargs:
        Extra keyword arguments forwarded to
        :func:`quantumuq.benchmarks.models.train_pennylane_vqc` or
        :func:`quantumuq.benchmarks.models.train_qiskit_vqc`.
    """

    if dataset not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose from {list(_DATASET_LOADERS)}"
        )
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {_BACKENDS}")

    data = _DATASET_LOADERS[dataset](random_state=seed)
    X, y = data.X, data.y

    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        keep = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[keep], y[keep]

    perm = rng.permutation(len(X))
    n_test = max(1, int(len(X) * test_size))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_kwargs = dict(train_kwargs or {})

    if backend == "pennylane":
        from ..adapters.pennylane_adapter import wrap_qnode
        from .models import train_pennylane_vqc

        qnode, params, postprocess = train_pennylane_vqc(
            X_train, y_train, seed=seed, **train_kwargs
        )
        predictor = wrap_qnode(
            qnode,
            task="classification",
            n_classes=2,
            params=params,
            postprocess=postprocess,
        )
    else:
        from ..adapters.qiskit_adapter import wrap_qiskit_sampler
        from .models import train_qiskit_vqc

        sampler, circuit, feature_map = train_qiskit_vqc(
            X_train, y_train, seed=seed, **train_kwargs
        )
        predictor = wrap_qiskit_sampler(
            sampler,
            circuit=circuit,
            task="classification",
            n_classes=2,
            feature_map=feature_map,
        )

    # Cross-entropy loss is invariant under a global label swap, so either
    # backend's optimizer can converge to a solution where the two classes
    # are consistently swapped -- a benign labeling artifact, not a failed
    # fit, but one that would otherwise show up as suspiciously-below-chance
    # accuracy. Detect it once on the training set and correct for it below.
    train_dist = predictor.with_uq(
        ShotBootstrap(n_samples=1, shots=1000, seed=seed)
    ).predict_dist(X_train)
    train_accuracy = float((train_dist.mean.argmax(axis=1) == y_train).mean())
    swap_classes = train_accuracy < 0.5

    results: List[BenchmarkResult] = []
    for shots in shots_list:
        uq = ShotBootstrap(n_samples=n_samples, shots=int(shots), seed=seed)
        model = predictor.with_uq(uq)
        dist = model.predict_dist(X_test)
        mean = dist.mean[:, ::-1] if swap_classes else dist.mean
        accuracy = float((mean.argmax(axis=1) == y_test).mean())
        results.append(
            BenchmarkResult(
                dataset=dataset,
                backend=backend,
                shots=int(shots),
                accuracy=accuracy,
                nll=nll(y_test, mean),
                ece=ece(y_test, mean),
                brier=brier(y_test, mean),
                mean_predictive_entropy=float(predictive_entropy(mean).mean()),
                mean_uncertainty=float(dist.std.mean()),
            )
        )
    return results
