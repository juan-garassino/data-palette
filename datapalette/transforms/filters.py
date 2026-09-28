from __future__ import annotations

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform

__all__ = ["Emboss", "Sharpen", "CustomKernel"]


class Emboss(ImageTransform):
    """Apply an emboss convolution filter."""

    _KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

    def _apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self._KERNEL)


class Sharpen(ImageTransform):
    """Apply a sharpening convolution filter."""

    _KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    def _apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self._KERNEL)


class CustomKernel(ImageTransform):
    """Apply a user-supplied convolution kernel.

    Parameters
    ----------
    kernel : np.ndarray
        2-D convolution kernel.
    """

    def __init__(self, kernel: np.ndarray | None = None):
        self.kernel = kernel if kernel is not None else Sharpen._KERNEL

    def _apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self.kernel)
