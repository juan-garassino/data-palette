"""Built-in dataset loaders and segmentation-mask generators."""

from __future__ import annotations

from datapalette.datasets.mnist import load_fashion_mnist, load_mnist
from datapalette.datasets.segmentation import (
    make_binary_masks,
    make_instance_masks,
    make_multiclass_masks,
)

__all__ = [
    "load_mnist",
    "load_fashion_mnist",
    "make_binary_masks",
    "make_multiclass_masks",
    "make_instance_masks",
]
