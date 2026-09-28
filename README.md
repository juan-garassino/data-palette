# DataPalette

Image dataset preprocessing and augmentation with composable, scikit-learn-compatible
transforms — and no scikit-learn dependency. Core install is just NumPy and OpenCV (headless).

- **23 transforms**: spatial, colour, noise, edges, filters, frequency, and GAN/diffusion
  dataset prep (aspect resize, square crop/pad, multi-crop, dominant colour).
- **Task pipelines** for GANs, segmentation (masks stay label-exact) and diffusion.
- **Reproducible**: every random transform takes `random_state`.
- **Works inside `sklearn.pipeline.Pipeline`** when you want it, and in PyTorch `DataLoader`s via
  an adapter.
- **CLI** for batch jobs: `datapalette apply photos/ -o out/ --step "Resize(size=(256, 256))"`.

## Installation

```bash
pip install datapalette            # numpy + opencv-python-headless
pip install "datapalette[torch]"   # + MNIST loaders and the PyTorch Dataset adapter
```

Python 3.11+.

## Quick start

Transforms take a `H x W x C` (or `H x W`) `uint8` array in OpenCV's BGR order, or a list of them.

```python
import numpy as np
from datapalette import Compose, GaussianNoise, Mirror, Resize

image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)

pipe = Compose(
    [
        Resize(size=(128, 128)),
        Mirror(mode="horizontal"),
        GaussianNoise(amount=0.03, random_state=0),
    ]
)
out = pipe.transform(image)
assert out.shape == (128, 128, 3)

batch = pipe.transform([image, image])  # lists in, lists out
assert len(batch) == 2
```

Transforms are plain estimators (`get_params`, `set_params`, `fit`, `transform`,
`fit_transform`), so they also drop into scikit-learn if it is installed:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from datapalette import Resize, Rotate

pipe = Pipeline([("resize", Resize(size=(64, 64))), ("rotate", Rotate(angle=90))])
pipe.set_params(rotate__angle=180)
out = pipe.transform(np.zeros((100, 80, 3), dtype=np.uint8))
assert out.shape == (64, 64, 3)
```

## GAN / diffusion dataset prep

The `dataset` transforms port the image-op vocabulary of
[dvschultz/dataset-tools](https://github.com/dvschultz/dataset-tools), the StyleGAN-community
prep toolkit, to turn arbitrary imagery into a square, fixed-size training set:

```python
import numpy as np
from datapalette import AspectResize, Compose, ManySquares, SquareCrop

wide = np.zeros((300, 900, 3), dtype=np.uint8)
prep = Compose([AspectResize(max_size=512), SquareCrop(h_align="center")])
assert prep.transform(wide).shape == (171, 171, 3)  # longest side 512, then a centred square

crops = ManySquares().transform(wide)  # left + right squares from a wide image
assert len(crops) == 2
```

`DominantColor` labels an image with its nearest named colour (k-means + CIELAB ΔE), handy for
foldering a dataset by colour.

## Segmentation: masks stay exact

Image and mask get the same geometry, but masks are resized and rotated with nearest-neighbour
interpolation and read from disk unchanged, so no label values are invented at the edges:

```python
import os

import cv2
import numpy as np
from datapalette import SegmentationPipeline

for d in ("images", "masks"):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(0)
mask = np.zeros((60, 80), dtype=np.uint8)
mask[10:50, 20:60] = 3  # class 3
cv2.imwrite("images/a.png", rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
cv2.imwrite("masks/a.png", mask)

X, y = SegmentationPipeline(dataset=None, size=(64, 64)).load_and_transform(
    images_dir="images", masks_dir="masks"
)
assert set(np.unique(y[0])) <= {0, 3}
```

`GANPipeline` (resize, rotate, mirror, brightness/contrast; also accepts a video) and
`DiffusionPipeline` (resize, PCA colour, random crop) work the same way and return `(X, None)`.
With the `[torch]` extra, `dataset="mnist"` or `"fashion_mnist"` loads a built-in dataset instead
of a directory.

## PyTorch

```python
import os

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from datapalette import GANPipeline
from datapalette.adapters import DataPaletteDataset

os.makedirs("frames", exist_ok=True)
for i in range(4):
    cv2.imwrite(f"frames/{i}.png", np.zeros((50, 50, 3), dtype=np.uint8))

ds = DataPaletteDataset("frames", pipeline=GANPipeline(dataset=None, size=(32, 32)))
batch = next(iter(DataLoader(ds, batch_size=4)))
assert batch.shape == (4, 3, 32, 32)  # float32 in [0, 1]; pass masks_dir= for (image, mask) pairs
```

## Command line

```bash
datapalette list-transforms                                     # every transform + defaults

datapalette apply photos/ -o out/ \
    --step "AspectResize(max_size=512)" --step "SquareCrop" --step "Mirror"

datapalette apply photos/ -o out/ --config examples/config.toml  # [[steps]] tables
datapalette showcase photo.jpg -o steps/ --step "Resize(size=(256, 256))" --step "Sharpen"

datapalette run gan --video clip.mp4 --fps 2 --size 128 128 -o gan_set/
datapalette run segmentation --images images/ --masks masks/ --size 256 256 -o seg_set/
```

`--step` values are parsed with Python's `ast` and `literal_eval` (literal arguments only; nothing is
evaluated) and override the config file's steps. Multi-output transforms such as `Tile` write
`name_000.png`, `name_001.png`, ...

## Transforms

| Class | Module | Parameters |
|---|---|---|
| `Rotate` | `spatial` | `angle=90.0`, `expand=False`, `interpolation="linear"` |
| `Mirror` | `spatial` | `mode="horizontal"` (`horizontal`, `vertical`, `both`) |
| `RandomCrop` | `spatial` | `crop_size=(224, 224)`, `random_state=None` |
| `Tile` | `spatial` | `tile_size=(256, 256)`, `overlap=0.0` — returns a list |
| `Resize` | `spatial` | `size=(256, 256)`, `interpolation="linear"` (`nearest`, `linear`, `cubic`, `area`) |
| `PCAColorAugmentation` | `color` | `alpha_std=0.1`, `random_state=None` — `fit` on a dataset for AlexNet-style stats |
| `ConvertColorSpace` | `color` | `target="hsv"` (`hsv`, `lab`, `gray`) |
| `EnhanceGreen` | `color` | — (CLAHE on the green channel) |
| `Multispectral` | `color` | — (7 simulated channels) |
| `GaussianNoise` | `noise` | `amount=0.05`, `random_state=None` |
| `SaltPepperNoise` | `noise` | `amount=0.05`, `salt_ratio=0.5`, `random_state=None` |
| `BrightnessContrast` | `noise` | `brightness_range=(-0.2, 0.2)` (fraction of 255), `contrast_range=(0.8, 1.2)`, `random_state=None` |
| `FourierTransform` | `frequency` | — (log-magnitude spectrum) |
| `GradientChannels` | `edges` | — (Sobel magnitude + direction, 2 channels) |
| `EdgeChannels` | `edges` | `low_threshold=100`, `high_threshold=200` (Canny + Laplacian, 2 channels) |
| `Emboss` | `filters` | — |
| `Sharpen` | `filters` | — |
| `CustomKernel` | `filters` | `kernel=None` (defaults to the sharpen kernel) |
| `AspectResize` | `dataset` | `max_size=256` |
| `SquareCrop` | `dataset` | `h_align="center"`, `v_align="center"` |
| `SquarePad` | `dataset` | `border_type="constant"`, `fill=(0, 0, 0)` |
| `ManySquares` | `dataset` | `ratio_threshold=1.2` — returns a list |
| `DominantColor` | `dataset` | `palette=None`, `n_clusters=5` — returns a colour name |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
