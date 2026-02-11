from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyDataset:
    X: np.ndarray
    y: np.ndarray


def make_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int | None = 0,
) -> ToyDataset:
    """Simple implementation of a 2D two-moons dataset (no sklearn dependency)."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer moon
    outer_theta = np.linspace(0, np.pi, n_samples_out)
    outer_x = np.column_stack(
        (np.cos(outer_theta), np.sin(outer_theta))
    )

    # Inner moon
    inner_theta = np.linspace(0, np.pi, n_samples_in)
    inner_x = np.column_stack(
        (1 - np.cos(inner_theta), 1 - np.sin(inner_theta) - 0.5)
    )

    X = np.vstack([outer_x, inner_x])
    X += noise * rng.standard_normal(size=X.shape)

    y = np.concatenate(
        [np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)]
    )
    return ToyDataset(X=X, y=y)

