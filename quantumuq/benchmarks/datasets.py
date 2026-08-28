from __future__ import annotations

from typing import Optional

import numpy as np

from ..datasets.toy import ToyDataset, make_moons

__all__ = ["load_moons", "load_iris", "load_breast_cancer"]


def load_moons(
    n_samples: int = 200, noise: float = 0.15, random_state: Optional[int] = 0
) -> ToyDataset:
    """Two-moons dataset (2 features, 2 classes). No external dependency.

    Thin wrapper around :func:`quantumuq.datasets.toy.make_moons`, kept here
    for naming consistency with the other benchmark-suite loaders.
    """

    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def _scale_to_range(
    X: np.ndarray, low: float = -np.pi, high: float = np.pi
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    return low + (X - x_min) / span * (high - low)


def _require_sklearn(loader_name: str):
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            f"{loader_name} requires scikit-learn. Install with "
            '`pip install "quantumuq[benchmarks]"`.'
        ) from exc


def load_iris(random_state: Optional[int] = 0) -> ToyDataset:
    """Binary subset of the classic iris dataset (setosa vs. versicolor).

    Reduced to petal length and petal width -- the two most class-separating
    raw features -- and scaled to ``[-pi, pi]`` so the values are suitable
    inputs for an angle-embedding circuit. Scaling uses the full dataset's
    min/max, not a train-only fit: fine for a lightweight benchmark, not a
    rigorous train/test-safe preprocessing pipeline.

    Requires scikit-learn: ``pip install "quantumuq[benchmarks]"``.
    """

    _require_sklearn("load_iris")
    from sklearn.datasets import load_iris as _sk_load_iris

    data = _sk_load_iris()
    mask = data.target < 2  # setosa (0) vs versicolor (1); virginica (2) dropped
    X = data.data[mask][:, 2:4]  # petal length, petal width
    y = data.target[mask]
    X = _scale_to_range(X)

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X))
    return ToyDataset(X=X[perm], y=y[perm])


def load_breast_cancer(random_state: Optional[int] = 0) -> ToyDataset:
    """Breast Cancer Wisconsin (diagnostic) dataset, reduced from 30 to 2
    features via PCA and scaled to ``[-pi, pi]`` for an angle-embedding
    circuit.

    Requires scikit-learn: ``pip install "quantumuq[benchmarks]"``.
    """

    _require_sklearn("load_breast_cancer")
    from sklearn.datasets import load_breast_cancer as _sk_load_breast_cancer
    from sklearn.decomposition import PCA

    data = _sk_load_breast_cancer()
    X = PCA(n_components=2, random_state=random_state).fit_transform(data.data)
    y = data.target
    X = _scale_to_range(X)

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X))
    return ToyDataset(X=X[perm], y=y[perm])
