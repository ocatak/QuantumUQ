"""QuantumUQ benchmark suite.

Reference datasets (`quantumuq.benchmarks.moons`, `.iris`,
`.breast_cancer`), a small variational classifier per backend
(`quantumuq.benchmarks.models`), and a shot-count sweep harness
(`run_benchmark`) that reports accuracy and calibration metrics
reproducibly, so papers can cite a fixed benchmark rather than "we used a
Python package."

`load_iris`/`load_breast_cancer` require scikit-learn:
``pip install "quantumuq[benchmarks]"``. `load_moons` has no extra
dependency.
"""

from .datasets import load_breast_cancer, load_iris, load_moons
from .runner import BenchmarkResult, run_benchmark

__all__ = [
    "load_moons",
    "load_iris",
    "load_breast_cancer",
    "BenchmarkResult",
    "run_benchmark",
]
