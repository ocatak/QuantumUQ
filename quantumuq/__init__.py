"""Public API for QuantumUQ."""

from ._version import __version__
from .adapters.pennylane_adapter import wrap_qnode
from .adapters.qiskit_adapter import wrap_qiskit_estimator, wrap_qiskit_sampler
from .core.methods import DeepEnsemble, NoiseProfile, ShotBootstrap
from .core.metrics import (
    brier,
    ece,
    gaussian_nll,
    nll,
    predictive_entropy,
    rmse,
)
from .core.predictors import PredictiveDistribution, UQModel

__all__ = [
    "__version__",
    # Adapters
    "wrap_qnode",
    "wrap_qiskit_estimator",
    "wrap_qiskit_sampler",
    # Core abstractions
    "PredictiveDistribution",
    "UQModel",
    # Methods
    "ShotBootstrap",
    "DeepEnsemble",
    "NoiseProfile",
    # Metrics
    "ece",
    "nll",
    "brier",
    "predictive_entropy",
    "rmse",
    "gaussian_nll",
]

