from __future__ import annotations

__version__ = "0.1.0"

from datapalette.io.batch import load_image, process_directory, save_image
from datapalette.pipelines import (
    DiffusionPipeline,
    GANPipeline,
    SegmentationPipeline,
    TaskPipeline,
)
from datapalette.transforms import (
    AspectResize,
    BrightnessContrast,
    ConvertColorSpace,
    CustomKernel,
    DominantColor,
    EdgeChannels,
    Emboss,
    EnhanceGreen,
    FourierTransform,
    GaussianNoise,
    GradientChannels,
    ImageTransform,
    ManySquares,
    Mirror,
    Multispectral,
    PCAColorAugmentation,
    RandomCrop,
    Resize,
    Rotate,
    SaltPepperNoise,
    Sharpen,
    SquareCrop,
    SquarePad,
    Tile,
)

__all__ = [
    "__version__",
    # Transforms
    "ImageTransform",
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
    # Pipelines
    "TaskPipeline",
    "GANPipeline",
    "SegmentationPipeline",
    "DiffusionPipeline",
    # I/O
    "process_directory",
    "load_image",
    "save_image",
]
