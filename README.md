# DataPalette

Image dataset preprocessing and augmentation toolkit with sklearn-compatible transforms.

DataPalette provides a library of composable image transforms that plug directly
into scikit-learn `Pipeline` objects, plus task-specific pipelines for GANs,
segmentation, and diffusion models.

---

## Installation

```bash
git clone https://github.com/juan-garassino/data-palette.git
cd data-palette
pip install -e .
```

With PyTorch/torchvision support (required for built-in datasets and the
`DataPaletteDataset` adapter):

```bash
pip install -e ".[torch]"
```

For development (testing, linting, type checking):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

Every transform is an sklearn-compatible estimator, so you can compose them
with `sklearn.pipeline.Pipeline`:

```python
from sklearn.pipeline import Pipeline
from datapalette import Rotate, GaussianNoise

pipe = Pipeline(
    [
        ("rotate", Rotate(angle=15)),
        ("noise", GaussianNoise(amount=0.03)),
    ]
)

import cv2

image = cv2.imread("photo.jpg")  # BGR uint8, shape (H, W, 3)
augmented = pipe.fit_transform(image)
cv2.imwrite("augmented.jpg", augmented)
```

`fit_transform` is used because recent scikit-learn releases raise
`NotFittedError` when `Pipeline.transform` is called before `fit`
(see [Known issues](#known-issues)).

---

## Built-in Datasets

DataPalette ships convenience loaders for MNIST and Fashion-MNIST that return
plain NumPy arrays (requires the `[torch]` extra):

```python
from datapalette.datasets import load_mnist, load_fashion_mnist

images, labels = load_mnist(train=True)  # (60000, 28, 28, 1) uint8
images, labels = load_fashion_mnist(train=True)
```

---

## Task Pipelines

Pre-configured pipelines bundle common transforms for specific deep-learning
tasks and return `(X, y)` pairs ready for training.

> These snippets are not verified: with scikit-learn 1.9.1 the
> `load_and_transform` calls currently raise `NotFittedError`
> (see [Known issues](#known-issues)). The `dataset="mnist"` /
> `"fashion_mnist"` variants also need the `[torch]` extra and a download.

### GANPipeline

```python
from datapalette import GANPipeline

pipe = GANPipeline(dataset="mnist", size=(64, 64))
X, _ = pipe.load_and_transform()
```

### SegmentationPipeline

```python
from datapalette import SegmentationPipeline

pipe = SegmentationPipeline(dataset=None, mode="binary")
X, y = pipe.load_and_transform(images_dir="images/", masks_dir="masks/")
```

### DiffusionPipeline

```python
from datapalette import DiffusionPipeline

pipe = DiffusionPipeline(dataset="fashion_mnist", size=(64, 64))
X, _ = pipe.load_and_transform()
```

---

## PyTorch Adapter

`DataPaletteDataset` wraps any pipeline into a lazy-loading PyTorch `Dataset`:

```python
from datapalette.adapters.torch import DataPaletteDataset
from datapalette import GANPipeline
from torch.utils.data import DataLoader

pipe = GANPipeline(dataset=None, size=(256, 256))
ds = DataPaletteDataset("images/", pipeline=pipe)
loader = DataLoader(ds, batch_size=16, shuffle=True)

for images, labels in loader:
    ...  # images: (B, C, H, W) float32 tensors
```

---

## CLI Usage

DataPalette includes a command-line interface for batch processing:

```bash
# Showcase mode: save every intermediate step (verified; writes
# 01_rotation.png, 02_mirror.png, 03_brightness_contrast.png)
datapalette --pipeline showcase --input-type single_image --input photo.jpg \
    --rotate --mirror --brightness-contrast --output_dir ./showcase
```

The other pipelines are available through the same flags (`--pipeline gan`,
`unet`, `diffusion`, `custom`; `--input-type video|images|single_image`; see
`datapalette --help`), but they currently fail on scikit-learn 1.9.1, and
`--pipeline custom` with `--input-type single_image` passes a file where
`process_directory` expects a directory. An example config is in
[`examples/config.yaml`](examples/config.yaml).

---

## Transforms

| Class | Module | Parameters |
|---|---|---|
| `Rotate` | `spatial` | `angle: float = 90.0` |
| `Mirror` | `spatial` | `mode: str = "horizontal"` (`horizontal`, `vertical`, `both`) |
| `RandomCrop` | `spatial` | `crop_size: tuple[int, int] = (224, 224)` |
| `Tile` | `spatial` | `tile_size: tuple[int, int] = (256, 256)`, `overlap: float = 0.0` |
| `Resize` | `spatial` | `size: tuple[int, int] = (256, 256)` |
| `PCAColorAugmentation` | `color` | `alpha_std: float = 0.1` |
| `ConvertColorSpace` | `color` | `target: str = "hsv"` (`hsv`, `lab`, `gray`) |
| `EnhanceGreen` | `color` | *(none)* |
| `Multispectral` | `color` | *(none)* |
| `GaussianNoise` | `noise` | `amount: float = 0.05` |
| `SaltPepperNoise` | `noise` | `amount: float = 0.05`, `salt_ratio: float = 0.5` |
| `BrightnessContrast` | `noise` | `brightness_range: tuple = (0.5, 1.5)`, `contrast_range: tuple = (0.5, 1.5)` |
| `FourierTransform` | `frequency` | *(none)* |
| `GradientChannels` | `edges` | *(none)* |
| `EdgeChannels` | `edges` | `low_threshold: int = 100`, `high_threshold: int = 200` |
| `Emboss` | `filters` | *(none)* |
| `Sharpen` | `filters` | *(none)* |
| `CustomKernel` | `filters` | `kernel: np.ndarray \| None = None` (3x3 sharpen kernel when `None`) |
| `AspectResize` | `dataset` | `max_size: int = 256` |
| `SquareCrop` | `dataset` | `h_align: str = "center"` (`left`/`center`/`right`), `v_align: str = "center"` (`top`/`center`/`bottom`) |
| `SquarePad` | `dataset` | `border_type: str = "constant"` (`constant`/`reflect`/`replicate`), `fill: tuple = (0, 0, 0)` |
| `ManySquares` | `dataset` | `ratio_threshold: float = 1.2` — multi-crop; returns a *list* |
| `DominantColor` | `dataset` | `palette: dict \| None = None`, `n_clusters: int = 5` — returns a colour *name* |

### GAN/diffusion dataset prep (`dataset` module)

The `dataset` transforms port the image-op vocabulary from
[`dvschultz/dataset-tools`](https://github.com/dvschultz/dataset-tools) — the
standard StyleGAN-community prep toolkit — as clean, typed, sklearn-compatible
transforms. The canonical pipeline turns arbitrary imagery into a square,
fixed-size training set:

```python
from sklearn.pipeline import Pipeline
from datapalette import AspectResize, SquareCrop  # or SquarePad to keep the whole frame

prep = Pipeline(
    [
        ("resize", AspectResize(max_size=512)),  # longest side -> 512, aspect preserved
        ("square", SquareCrop(h_align="center")),  # crop the longer side to square, no rescaling
    ]
)
```

A 160x120 (width x height) input comes out 384x384: the longest side is scaled to 512
(512x384), then the 512-pixel width is cropped to 384.

`ManySquares` multiplies dataset size by emitting top/bottom (or left/right)
crops from tall/wide sources, and `DominantColor` labels each image with its
nearest named colour (KMeans + CIELAB deltaE) for foldering a set by colour.

---

## Known issues

Checked with scikit-learn 1.9.1 on Python 3.12 (`pytest`: 73 passed, 5 failed,
5 skipped):

- `Pipeline.transform` before `fit` raises `NotFittedError`, which breaks
  `TaskPipeline` subclasses, the CLI `custom`/`gan`/`unet`/`diffusion` paths and
  `DataPaletteDataset`. [`ROADMAP.md`](ROADMAP.md) lists this as a v0.1.1 item.
- `mypy datapalette/` reports 6 errors (CI runs it).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR
guidelines.

---

## License

MIT -- see [LICENSE](LICENSE) for details.
