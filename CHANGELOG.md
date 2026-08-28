# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/ocatak/QuantumUQ/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/ocatak/QuantumUQ/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ocatak/QuantumUQ/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ocatak/QuantumUQ/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ocatak/QuantumUQ/commit/bb34888
