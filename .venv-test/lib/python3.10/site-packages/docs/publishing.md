## Publishing QuantumUQ to PyPI

### 1. Build the distribution

```bash
python -m build  # or: hatch build
```

This creates `dist/*.tar.gz` and `dist/*.whl`.

### 2. TestPyPI upload (recommended first)

```bash
python -m pip install --upgrade twine
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Install from TestPyPI for a smoke test:

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple quantumuq
```

### 3. PyPI upload

Once satisfied with TestPyPI:

```bash
twine upload dist/*
```

### 4. GitHub Actions: trusted publishing

The `release.yml` workflow is configured to:

- Trigger on tags `v*` (e.g. `v0.1.0`).
- Build sdist + wheel.
- Upload to PyPI using trusted publishing (or a PyPI token if configured).

Refer to `release.yml` for details on secrets and configuration.

