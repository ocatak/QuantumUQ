from __future__ import annotations

import numpy as np

from quantumuq.core.metrics import (
    brier,
    ece,
    gaussian_nll,
    nll,
    predictive_entropy,
    rmse,
)


def test_classification_metrics_basic() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.3, 0.7],
        ]
    )
    assert nll(y, p) > 0.0
    assert 0.0 <= ece(y, p) <= 1.0
    assert brier(y, p) >= 0.0
    ent = predictive_entropy(p)
    assert ent.shape == (4,)
    assert np.all(ent >= 0.0)


def test_ece_counts_full_confidence_predictions() -> None:
    # Predictions with confidence exactly 1.0 must not be dropped from the
    # last bin: a model that is fully confident and always wrong should
    # report near-maximal ECE, not zero.
    y_wrong = np.array([0, 0, 0, 0])
    p_wrong = np.array([[0.0, 1.0]] * 4)
    assert ece(y_wrong, p_wrong) == 1.0

    y_correct = np.array([0, 1, 0, 1])
    p_correct = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert ece(y_correct, p_correct) == 0.0


def test_regression_metrics_basic() -> None:
    y = np.array([0.0, 1.0, 2.0])
    y_hat = np.array([0.1, 0.9, 1.9])
    std = np.array([0.5, 0.5, 0.5])
    assert rmse(y, y_hat) >= 0.0
    assert gaussian_nll(y, y_hat, std) > 0.0

