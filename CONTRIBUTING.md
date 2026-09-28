# Contributing to DataPalette

Thanks for your interest in contributing! This document covers the development
workflow.

---

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/juan-garassino/data-palette.git
   cd data-palette
   ```

2. Install with [uv](https://docs.astral.sh/uv/) (creates `.venv` from `uv.lock`):

   ```bash
   uv sync --extra dev
   ```

3. (Optional) Add the PyTorch extra for the dataset loaders and the torch adapter:

   ```bash
   uv sync --extra dev --extra torch
   ```

4. Install the pre-commit hooks:

   ```bash
   uv tool install pre-commit
   pre-commit install
   ```

---

## Running Tests

Tests live in the `tests/` directory and are run with pytest:

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=datapalette --cov-report=term-missing
```

`tests/test_readme.py` executes every Python block in `README.md`, so keep examples runnable.

---

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

To auto-fix issues:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

---

## Type Checking

Static types are checked with [mypy](https://mypy-lang.org/):

```bash
uv run mypy datapalette/
```

Configuration is in `pyproject.toml` under `[tool.mypy]`.

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused and descriptive.
3. Ensure all checks pass locally:
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run mypy datapalette/
   uv run pytest --cov
   ```
4. Open a pull request against `main` with a clear description of your
   changes, and add an entry to `CHANGELOG.md`.
5. A maintainer will review your PR. Address any feedback, then it will be
   merged.

---

## Code Style

- Target Python 3.11+.
- Maximum line length is 110 characters (configured in `pyproject.toml`).
- All public classes and functions should have docstrings with a
  `Parameters` section (NumPy-style).
- New transforms should subclass `ImageTransform` (or `MultiOutputTransform` if they return
  several images) and implement `_apply`; random ones take `random_state` and draw from
  `self._random()`.
- New transforms must be added to the table in `README.md` (a test enforces it).
