"""Noise and brightness/contrast augmentation transforms."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform

__all__ = ["GaussianNoise", "SaltPepperNoise", "BrightnessContrast"]


class GaussianNoise(ImageTransform):
    """Add Gaussian noise to an image.

    Parameters
    ----------
    amount : float
        Noise intensity as a fraction of 255. Default ``0.05``.
    """

    def __init__(self, amount: float = 0.05):
        self.amount = amount

    def _apply(self, image: np.ndarray) -> np.ndarray:
        sigma = self.amount * 255
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class SaltPepperNoise(ImageTransform):
    """Add salt-and-pepper noise to an image.

    Parameters
    ----------
    amount : float
        Fraction of pixels to corrupt. Default ``0.05``.
    salt_ratio : float
        Proportion of corrupted pixels that become salt (white). Default ``0.5``.
    """

    def __init__(self, amount: float = 0.05, salt_ratio: float = 0.5):
        self.amount = amount
        self.salt_ratio = salt_ratio

    def _apply(self, image: np.ndarray) -> np.ndarray:
        noisy = image.copy()
        h, w = image.shape[:2]
        total_pixels = h * w

        num_salt = int(np.ceil(self.amount * total_pixels * self.salt_ratio))
        rows = np.random.randint(0, h, num_salt)
        cols = np.random.randint(0, w, num_salt)
        noisy[rows, cols] = 255

        num_pepper = int(np.ceil(self.amount * total_pixels * (1.0 - self.salt_ratio)))
        rows = np.random.randint(0, h, num_pepper)
        cols = np.random.randint(0, w, num_pepper)
        noisy[rows, cols] = 0

        return noisy


class BrightnessContrast(ImageTransform):
    """Randomly adjust brightness and contrast within given ranges.

    Parameters
    ----------
    brightness_range : tuple[float, float]
        ``(min, max)`` for the additive brightness term. Default ``(0.5, 1.5)``.
    contrast_range : tuple[float, float]
        ``(min, max)`` for the multiplicative contrast factor. Default ``(0.5, 1.5)``.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.5, 1.5),
        contrast_range: Tuple[float, float] = (0.5, 1.5),
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def _apply(self, image: np.ndarray) -> np.ndarray:
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
