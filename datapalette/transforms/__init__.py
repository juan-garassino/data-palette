from __future__ import annotations

from datapalette.transforms.base import Compose, ImageTransform
from datapalette.transforms.color import (
    ConvertColorSpace,
    EnhanceGreen,
    Multispectral,
    PCAColorAugmentation,
)
from datapalette.transforms.dataset import (
    AspectResize,
    DominantColor,
    ManySquares,
    SquareCrop,
    SquarePad,
)
from datapalette.transforms.edges import EdgeChannels, GradientChannels
from datapalette.transforms.filters import CustomKernel, Emboss, Sharpen
from datapalette.transforms.frequency import FourierTransform
from datapalette.transforms.noise import BrightnessContrast, GaussianNoise, SaltPepperNoise
from datapalette.transforms.spatial import Mirror, RandomCrop, Resize, Rotate, Tile

__all__ = [
    "ImageTransform",
    "Compose",
    "REGISTRY",
    "Rotate",
    "Mirror",
    "RandomCrop",
    "Tile",
    "Resize",
    "PCAColorAugmentation",
    "ConvertColorSpace",
    "EnhanceGreen",
    "Multispectral",
    "GaussianNoise",
    "SaltPepperNoise",
    "BrightnessContrast",
    "FourierTransform",
    "GradientChannels",
    "EdgeChannels",
    "Emboss",
    "Sharpen",
    "CustomKernel",
    "AspectResize",
    "SquareCrop",
    "SquarePad",
    "ManySquares",
    "DominantColor",
]

#: Transform classes by name, for the CLI and config files.
REGISTRY = {
    name: obj
    for name, obj in sorted(globals().items())
    if isinstance(obj, type) and issubclass(obj, ImageTransform) and obj not in (ImageTransform, Compose)
}
