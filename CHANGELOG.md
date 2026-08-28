# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/ocatak/QuantumUQ/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ocatak/QuantumUQ/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/ocatak/QuantumUQ/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ocatak/QuantumUQ/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ocatak/QuantumUQ/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ocatak/QuantumUQ/commit/bb34888
