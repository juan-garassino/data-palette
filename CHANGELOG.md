# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — unreleased

First release on PyPI. The public API (everything exported from `datapalette`,
`datapalette.adapters` and the `datapalette` CLI) is now stable and follows semantic versioning.

### Breaking changes
- **scikit-learn is no longer a dependency.** `ImageTransform` implements the estimator protocol
  itself, and transforms still work inside `sklearn.pipeline.Pipeline`. `TaskPipeline.sklearn_pipeline`
  is replaced by `TaskPipeline.transforms` (a `Compose`).
- `TaskPipeline._load_images_from_dir` / `_transform_images` are public as `load_images` /
  `transform_images`.
- **CLI rewritten** as subcommands: `apply`, `showcase`, `run {gan,segmentation,diffusion}`,
  `list-transforms`. Configs are TOML (`[[steps]]` tables) instead of YAML.
- `BrightnessContrast` defaults: `brightness_range=(-0.2, 0.2)` as a fraction of 255 (was an
  additive 0.5–1.5 on a 0–255 scale, i.e. a no-op) and `contrast_range=(0.8, 1.2)`.
- `PCAColorAugmentation.transform` without `fit` uses per-image colour statistics and stores nothing
  (it used to fit itself on the first image it saw).
- `extract_frames(output_size=)` and `n_croppings_from_frame(crop_size=)` take `(height, width)`,
  like every other size in the package.
- `DataPaletteDataset` without `masks_dir` returns an image tensor, not `(image, None)`.
- `process_directory` lost `batch_size`/`desc`; `utils.setup_logging` is removed.
- Python 3.11+ (stdlib `tomllib`); core dependencies are only `numpy` and
  `opencv-python-headless` (tqdm, pyyaml, scikit-learn and the GUI OpenCV build are gone).

### Added
- `Compose` for chaining transforms (named or unnamed steps), `MultiOutputTransform` base.
- `random_state` on every random transform (`RandomCrop`, `GaussianNoise`, `SaltPepperNoise`,
  `BrightnessContrast`, `PCAColorAugmentation`, `n_croppings_from_frame`).
- `Resize(interpolation=)` and `Rotate(interpolation=, expand=)`; exact quarter turns.
- `datapalette list-transforms`, `datapalette apply`, safe `--step "Name(key=value)"` parsing.
- `datapalette.adapters.DataPaletteDataset` export; README examples are executed by the test suite.

### Fixed
- scikit-learn >= 1.8 raised `NotFittedError` for every pipeline built from datapalette transforms.
- Segmentation masks were resized/rotated bilinearly (inventing label values) and read as 3-channel
  colour images; instance mode skipped the transforms entirely.
- CLI: default output directory pointed inside site-packages; GAN results were discarded;
  `--input-type single_image` crashed; config values were silently ignored; tiling kept only the
  first tile; `--num-crops` did nothing.
- Transforms that need colour or gray input now raise a clear `ValueError` instead of crashing
  inside OpenCV; single-channel `H x W x 1` images keep their channel axis.
- Temporary video-frame directories are cleaned up.

## [0.1.0]

- sklearn-based transform library, task pipelines, MNIST loaders, PyTorch adapter, CLI.
