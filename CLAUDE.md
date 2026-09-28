# dataPalette

Image-dataset preprocessing/augmentation library published on PyPI as `datapalette`.
Legacy flat layout (`datapalette/`, not `src/`) — keep it. uv + hatchling; version is
single-sourced from `datapalette/__init__.py`.

## Commands

- `uv sync --extra dev` (`--extra torch` for the adapter/MNIST tests)
- `uv run pytest` · `uv run ruff check . && uv run ruff format --check .` · `uv run mypy datapalette/`
- `uv build && uvx twine check dist/*`

## Module map

- `transforms/base.py` — `ImageTransform` (own estimator protocol: get/set_params, fit/transform,
  sklearn `__sklearn_is_fitted__`/`__sklearn_tags__` hooks; **no sklearn import at module level**),
  `MultiOutputTransform`, `Compose`, helpers (`interpolation_flag`, `keep_channel_axis`,
  `require_channels`, `minmax_to_uint8`). Random transforms take `random_state` and use `self._random()`.
- `transforms/{spatial,color,noise,edges,filters,frequency,dataset}.py` — the 23 transforms;
  `transforms/__init__.py` builds `REGISTRY` (name -> class) for the CLI and TOML configs.
- `pipelines/` — `TaskPipeline` (`.transforms` Compose, `load_images`, `transform_images`) and
  `GANPipeline`, `SegmentationPipeline` (nearest-neighbour `mask_transforms`, masks read unchanged),
  `DiffusionPipeline`.
- `io/batch.py` — `list_images`, `load_image`, `save_image`, `process_directory`, `write_outputs`;
  `io/video.py` — `extract_frames`, `n_croppings_from_frame`. All sizes are `(height, width)`.
- `datasets/` — MNIST/Fashion-MNIST loaders (torchvision, lazy) and synthetic mask generators.
- `adapters/torch.py` — `DataPaletteDataset` (lazy torch import; bare tensor without masks).
- `cli.py` — argparse subcommands `list-transforms`, `apply`, `showcase`, `run`; `parse_step`
  uses `ast` + `literal_eval` only. `utils/config.py` — TOML via stdlib `tomllib`.

## Rules

- Core deps stay `numpy` + `opencv-python-headless`. Anything else is an extra.
- Every transform appears in the README table and every README python block runs
  (`tests/test_readme.py`).
- Breaking changes go in `CHANGELOG.md` under the next version.

## Release runbook

1. Bump `__version__` in `datapalette/__init__.py`, date the CHANGELOG entry, merge to `main`.
2. First time only: create accounts on pypi.org and test.pypi.org (2FA), add a *pending trusted
   publisher* on both (project `datapalette`, owner `juan-garassino`, repo `data-palette`,
   workflow `release.yml`, environment `pypi`), and create the `pypi` environment on GitHub with a
   required reviewer.
3. Tag `vX.Y.ZrcN` -> TestPyPI; smoke it with
   `uv run --with datapalette==X.Y.ZrcN --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match python -c "import datapalette"`.
4. Tag `vX.Y.Z` -> PyPI.
