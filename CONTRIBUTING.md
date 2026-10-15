# Contributing to DataPalette

Thanks for your interest in contributing! This document covers the development
workflow.

---

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/juan-garassino/data-palette.git
   cd dataPalette
   ```

2. Create a virtual environment and install in editable mode with dev
   dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. (Optional) Install the PyTorch extras if you need dataset loaders or the
   torch adapter:

   ```bash
   pip install -e ".[dev,torch]"
   ```

4. Install the pre-commit hooks:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

---

## Running Tests

Tests live in the `tests/` directory and are run with pytest:

```bash
pytest
```

With coverage:

```bash
pytest --cov=datapalette --cov-report=term-missing
```

---

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting:

```bash
ruff check .
ruff format --check .
```

To auto-fix issues:

```bash
ruff check --fix .
ruff format .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

---

## Type Checking

Static types are checked with [mypy](https://mypy-lang.org/):

```bash
mypy datapalette/
```


---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused and descriptive.
3. Ensure all checks pass locally:
   ```bash
   ruff check .
   mypy datapalette/
   pytest --cov
   ```
4. Open a pull request against `main` with a clear description of your
   changes.
5. A maintainer will review your PR. Address any feedback, then it will be
   merged.

---

## Code Style

- Target Python 3.9+.
- Maximum line length is 100 characters (configured in `pyproject.toml`).
- All public classes and functions should have docstrings with a
  `Parameters` section (NumPy-style).
- New transforms should subclass `ImageTransform` and implement `_apply`.
