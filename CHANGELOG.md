# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- README: Qiskit Ecosystem badge, now that QuantumUQ has been officially
  accepted ([Qiskit/ecosystem#1331](https://github.com/Qiskit/ecosystem/pull/1331)).
- "Open in Colab" badges on all 9 example notebooks, each followed by a
  `%pip install` cell so the notebook is self-contained with no local setup.
  PennyLane/Qiskit are pinned to the exact versions verified throughout
  development (`pennylane==0.42.3`, `qiskit==2.3.1`) rather than left
  open-ended -- an unpinned `qiskit` install can land on `qiskit==2.5.2`,
  which has an upstream bug unrelated to this project that breaks entirely
  on Python 3.10. Verified by executing every notebook end-to-end in a
  genuinely fresh virtual environment (only the packages Colab ships by
  default preinstalled) so the `%pip install` cell is exercised for real,
  not just added on faith.

## [0.4.1] - 2026-08-28

### Changed

- Capped `qiskit` (`<3`) and `qiskit-aer` (`<1`) in the `dev`/`aer` optional
  dependencies, requested by a Qiskit Ecosystem reviewer on the QuantumUQ
  submission ([Qiskit/ecosystem#1331](https://github.com/Qiskit/ecosystem/pull/1331)).
  Not hypothetical: earlier this session a fresh venv grabbed qiskit 2.5.2
  and hit an unrelated upstream bug in that release's own `passmanager`
  module.

## [0.4.0] - 2026-08-28

### Added

- `quantumuq.benchmarks`: a reproducible benchmark suite.
  - Datasets: `load_moons` (no extra dependency), `load_iris` (binary
    subset), `load_breast_cancer` (PCA-reduced) -- all reduced to 2
    features so one small reference circuit applies to each. `load_iris`/
    `load_breast_cancer` require the new `quantumuq[benchmarks]` extra
    (scikit-learn).
  - Reference variational classifiers per backend:
    `train_pennylane_vqc` (gradient-trained) and `train_qiskit_vqc`
    (SPSA-trained, since Qiskit circuits aren't differentiable through
    this library).
  - `run_benchmark(dataset, backend, shots_list=...)`: trains once, sweeps
    shot count, and reports accuracy/`nll`/`ece`/`brier`/
    `predictive_entropy`/mean `ShotBootstrap` uncertainty per shot count.
    Datasets larger than `max_samples` (default 150) are subsampled so
    runtime stays small and consistent regardless of source dataset size.
    Automatically detects and corrects for a global class-label swap that
    a symmetric loss can otherwise let either backend's optimizer converge
    to (shows up as suspiciously-below-chance accuracy if uncorrected).
  - New `quantumuq-benchmark` console script:
    `quantumuq-benchmark --backend pennylane --dataset moons --shots 100,500,1000,10000`.
  - `tests/test_benchmarks.py`, all 6 dataset/backend combinations verified
    to converge to above-chance accuracy.
- `examples/notebooks/08_pennylane_community_demo.ipynb`: "How Confident
  Should You Be in a Quantum Classifier?", written for submission via
  PennyLane's Community Demo track. Sweeps shot count on a trained VQC and
  measures directly (rather than assuming) that `ShotBootstrap` uncertainty
  shrinks ~1/sqrt(shots) while calibration barely moves -- more shots
  reduce measurement noise but don't by themselves fix calibration.

### Fixed

- Notebooks 00 and 02 used `pnp.eye(2)[y_train]` (fancy-indexing a
  `pennylane.numpy` tensor) inside a `qml.grad`-traced loss function, which
  silently collapses to a 0-d array under autograd tracing in this
  pennylane/autograd/numpy version combination, raising `AxisError`.
  `y_train` is a constant, not a differentiated quantity, so plain numpy
  indexing is correct and sidesteps the bug; both notebooks re-executed
  successfully after the fix.

## [0.3.0] - 2026-08-28

### Changed

- **Breaking**: `wrap_qiskit_sampler`/`wrap_qiskit_estimator` now require
  Qiskit V2 primitives (`BaseSamplerV2`/`BaseEstimatorV2` -- e.g.
  `StatevectorSampler`, `StatevectorEstimator`, `BackendSamplerV2`,
  `BackendEstimatorV2`). `qiskit>=2.0` removed the V1 primitives and the
  bare `BaseSampler`/`BaseEstimator` names this adapter previously
  imported, so it was unusable against current Qiskit regardless of
  anything else in the library. A V1-style object now raises a clear
  `TypeError` instead of failing at import time.
- Estimator `shots=` is converted to `precision = 1/sqrt(shots)`, the
  standard-error target V2 estimators use in place of a literal shot
  count.

### Added

- `tests/test_qiskit_adapter.py` (previously zero test coverage on the
  Qiskit adapter).
- `examples/notebooks/07_qiskit_v2_primitives.ipynb`, including the
  seeding gotcha where `StatevectorSampler(seed=<int>)` resets every
  call (zero shot-noise variance) versus `seed=<Generator instance>`
  (advances across calls).

### Fixed

- README's Qiskit quickstart used a circuit with zero free parameters
  (`qc.ry(0.0, 0)`) while its `feature_map` still supplied a binding
  value per call -- never actually ran. Fixed to use a real `Parameter`.
- README's PennyLane quickstart declared `n_classes=2` on a 2-qubit,
  4-outcome circuit with no `postprocess` to collapse them -- also never
  actually ran. Added the same 4-to-2 mapping notebook 00 already used.
- `examples/notebooks/06_uqmodel_persistence.ipynb`'s Qiskit section was
  illustrative-only (not executed) because the adapter didn't work
  against installed Qiskit; now a real, executed save/load round trip.

## [0.2.2] - 2026-08-28

### Fixed

- Placeholder author email in `pyproject.toml` (`ozgur.catak@example.com`)
  replaced with the correct contact address.
- Repo-wide `black` formatting drift cleared so CI passes.

### Added

- `CHANGELOG.md` (this file).
- `CITATION.cff`, so GitHub's "Cite this repository" button works.
- README badge row (PyPI, Python, Tests, Documentation, DOI, License,
  Downloads), a tagline block, a hero figure, and a proper Examples
  section linking all notebooks.

## [0.2.1] - 2026-08-28

### Fixed

- Published wheel was bundling an accidentally-committed dev virtualenv
  (`.venv-test`, 1768 files), bloating it to 654KB with irrelevant content
  including numpy's own internal test fixtures. `.venv-test` is now removed
  from version control, and `pyproject.toml`'s `[tool.hatch.build]` include
  patterns are anchored per-target so this can't recur: the wheel now ships
  only `quantumuq/` plus the script needed for the `quantumuq-smoke` console
  entry point (18.6KB total), while the sdist keeps the full `examples/`/
  `docs/` tree for reference.
- README referenced example notebooks as plain backtick text, which PyPI
  renders as inert (no repository context to resolve a relative path
  against). Added a proper Examples section linking all notebooks to their
  GitHub blob URLs.

## [0.2.0] - 2026-08-28

### Fixed

- `ece()` silently dropped predictions with confidence exactly `1.0` from
  the last calibration bin's accuracy and weight, instead of just binning
  them incorrectly. A model that was fully confident and always wrong
  reported `ECE = 0.0` instead of `~1.0`. This matters most in low-shot
  quantum measurement regimes, where probabilities saturate to exact 0/1
  values. See `examples/notebooks/05_ece_calibration_bugfix.ipynb`.

### Added

- `UQModel.save()` / `UQModel.load()`: checkpoint a fitted model's trained
  parameters and uncertainty-method configuration as JSON, for both
  PennyLane and Qiskit predictor backends, without pickling live backend
  objects or user-supplied closures. Supports `ShotBootstrap` and
  `NoiseProfile`; `DeepEnsemble` raises a clear error on save rather than
  silently doing the wrong thing, since each ensemble member is itself a
  live predictor. See `examples/notebooks/06_uqmodel_persistence.ipynb`.

## [0.1.0] - 2026-02-10

- Initial release: `ShotBootstrap`, `DeepEnsemble`, and `NoiseProfile`
  uncertainty methods; classification metrics (`nll`, `brier`, `ece`,
  `predictive_entropy`) and regression metrics (`rmse`, `gaussian_nll`);
  PennyLane and Qiskit adapters via `wrap_qnode`, `wrap_qiskit_sampler`,
  and `wrap_qiskit_estimator`.

[Unreleased]: https://github.com/ocatak/QuantumUQ/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/ocatak/QuantumUQ/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ocatak/QuantumUQ/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ocatak/QuantumUQ/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/ocatak/QuantumUQ/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ocatak/QuantumUQ/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ocatak/QuantumUQ/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ocatak/QuantumUQ/commit/bb34888
