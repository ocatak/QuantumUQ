from __future__ import annotations

import numpy as np
import pytest

from quantumuq.benchmarks import (
    BenchmarkResult,
    load_breast_cancer,
    load_iris,
    load_moons,
    run_benchmark,
)


def test_load_moons_shape() -> None:
    data = load_moons(n_samples=40, random_state=0)
    assert data.X.shape == (40, 2)
    assert data.y.shape == (40,)
    assert set(np.unique(data.y).tolist()) == {0, 1}


def test_load_iris_is_binary_two_features() -> None:
    sklearn = pytest.importorskip("sklearn")
    data = load_iris(random_state=0)
    assert data.X.shape[1] == 2
    assert set(np.unique(data.y).tolist()) == {0, 1}
    assert data.X.min() >= -np.pi - 1e-9
    assert data.X.max() <= np.pi + 1e-9
    del sklearn


def test_load_breast_cancer_two_features() -> None:
    pytest.importorskip("sklearn")
    data = load_breast_cancer(random_state=0)
    assert data.X.shape[1] == 2
    assert set(np.unique(data.y).tolist()) == {0, 1}
    assert data.X.min() >= -np.pi - 1e-9
    assert data.X.max() <= np.pi + 1e-9


def test_run_benchmark_pennylane_fast() -> None:
    results = run_benchmark(
        "moons",
        "pennylane",
        shots_list=[100, 1000],
        n_samples=2,
        max_samples=16,
        seed=0,
        train_kwargs={"epochs": 2},
    )
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results)
    assert [r.shots for r in results] == [100, 1000]
    for r in results:
        assert r.dataset == "moons"
        assert r.backend == "pennylane"
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.ece <= 1.0
        assert r.nll >= 0.0
        assert r.mean_uncertainty >= 0.0


def test_run_benchmark_qiskit_fast() -> None:
    results = run_benchmark(
        "moons",
        "qiskit",
        shots_list=[100, 1000],
        n_samples=2,
        max_samples=16,
        seed=0,
        train_kwargs={"spsa_iters": 3},
    )
    assert len(results) == 2
    for r in results:
        assert r.backend == "qiskit"
        assert 0.0 <= r.accuracy <= 1.0


def test_run_benchmark_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        run_benchmark("not_a_dataset", "pennylane")


def test_run_benchmark_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        run_benchmark("moons", "not_a_backend")


def test_run_benchmark_subsamples_large_datasets() -> None:
    pytest.importorskip("sklearn")
    results = run_benchmark(
        "breast_cancer",
        "pennylane",
        shots_list=[100],
        n_samples=1,
        max_samples=12,
        test_size=0.5,
        seed=0,
        train_kwargs={"epochs": 1},
    )
    # 12 samples total, 50% test -> 6 test points, so accuracy must be k/6
    # for some integer k in [0, 6]. If max_samples didn't subsample, the
    # denominator would be a different (larger) number of test points.
    assert len(results) == 1
    scaled = results[0].accuracy * 6
    assert abs(scaled - round(scaled)) < 1e-6
