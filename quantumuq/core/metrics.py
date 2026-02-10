from __future__ import annotations

import numpy as np


def nll(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Multiclass negative log-likelihood."""

    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)
    p_pred = np.clip(p_pred, eps, 1.0)
    idx = (np.arange(y_true.shape[0]), y_true)
    log_p = np.log(p_pred[idx])
    return float(-log_p.mean())


def brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Multiclass Brier score."""

    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)
    n = y_true.shape[0]
    n_classes = p_pred.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    diff = p_pred - one_hot
    return float(np.mean(np.sum(diff * diff, axis=1)))


def ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""

    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)
    confidences = p_pred.max(axis=1)
    predictions = p_pred.argmax(axis=1)
    accuracies = predictions == y_true

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        weight = mask.mean()
        ece_val += weight * abs(bin_acc - bin_conf)
    return float(ece_val)


def predictive_entropy(p_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Predictive entropy from class probabilities."""

    p_pred = np.asarray(p_pred, dtype=float)
    p_pred = np.clip(p_pred, eps, 1.0)
    p_pred = p_pred / p_pred.sum(axis=-1, keepdims=True)
    return -np.sum(p_pred * np.log(p_pred), axis=-1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Gaussian negative log-likelihood for regression."""

    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    var = np.clip(std, eps, None) ** 2
    per_point = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * ((y_true - mean) ** 2) / var
    return float(np.mean(per_point))

