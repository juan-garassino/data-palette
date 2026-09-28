from __future__ import annotations

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform, minmax_to_uint8

__all__ = ["GradientChannels", "EdgeChannels"]


class GradientChannels(ImageTransform):
    """Compute Sobel gradient magnitude and direction as a 2-channel image."""

    def _apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        magnitude = minmax_to_uint8(magnitude)
        direction = minmax_to_uint8(direction)

        return np.dstack((magnitude, direction))


class EdgeChannels(ImageTransform):
    """Compute Canny edges and Laplacian as a 2-channel image.

    Parameters
    ----------
    low_threshold : int
        Low threshold for Canny. Default ``100``.
    high_threshold : int
        High threshold for Canny. Default ``200``.
    """

    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        laplacian = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))

        return np.dstack((edges, laplacian))
