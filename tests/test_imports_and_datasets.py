from __future__ import annotations

import numpy as np

from quantumuq import (
    DeepEnsemble,
    NoiseProfile,
    PredictiveDistribution,
    ShotBootstrap,
    UQModel,
    brier,
    ece,
    gaussian_nll,
    nll,
    predictive_entropy,
    rmse,
    wrap_qiskit_sampler,
    wrap_qnode,
)
from quantumuq.datasets.toy import make_moons


def test_public_api_imports() -> None:
    # Simply ensure symbols are present.
    assert callable(ShotBootstrap)
    assert callable(DeepEnsemble)
    assert callable(NoiseProfile)
    assert callable(wrap_qnode)
    assert callable(wrap_qiskit_sampler)
    assert PredictiveDistribution is not None
    # Metrics
    for fn in (nll, brier, ece, predictive_entropy, rmse, gaussian_nll):
        assert callable(fn)


def test_toy_dataset_shapes() -> None:
    ds = make_moons(n_samples=50, noise=0.1, random_state=0)
    assert ds.X.shape == (50, 2)
    assert ds.y.shape == (50,)
    assert set(np.unique(ds.y)) == {0, 1}

