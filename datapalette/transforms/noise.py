from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from datapalette.transforms.base import ImageTransform

__all__ = ["GaussianNoise", "SaltPepperNoise", "BrightnessContrast"]


class GaussianNoise(ImageTransform):
    """Add Gaussian noise to an image.

    Parameters
    ----------
    amount : float
        Noise intensity as a fraction of 255. Default ``0.05``.
    random_state : int or None
        Seed for reproducible noise. Default ``None``.
    """

    def __init__(self, amount: float = 0.05, random_state: Optional[int] = None):
        self.amount = amount
        self.random_state = random_state

    def _apply(self, image: np.ndarray) -> np.ndarray:
        sigma = self.amount * 255
        noise = self._random().normal(0, sigma, image.shape)
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
    random_state : int or None
        Seed for reproducible noise. Default ``None``.
    """

    def __init__(self, amount: float = 0.05, salt_ratio: float = 0.5, random_state: Optional[int] = None):
        self.amount = amount
        self.salt_ratio = salt_ratio
        self.random_state = random_state

    def _apply(self, image: np.ndarray) -> np.ndarray:
        noisy = image.copy()
        h, w = image.shape[:2]
        total_pixels = h * w

        # Salt
        num_salt = int(np.ceil(self.amount * total_pixels * self.salt_ratio))
        rng = self._random()
        rows = rng.integers(0, h, num_salt)
        cols = rng.integers(0, w, num_salt)
        noisy[rows, cols] = 255  # (row, col) pairs: every channel of the pixel

        # Pepper
        num_pepper = int(np.ceil(self.amount * total_pixels * (1.0 - self.salt_ratio)))
        rows = rng.integers(0, h, num_pepper)
        cols = rng.integers(0, w, num_pepper)
        noisy[rows, cols] = 0

        return noisy


class BrightnessContrast(ImageTransform):
    """Randomly adjust brightness and contrast: ``out = contrast * image + brightness * 255``.

    Parameters
    ----------
    brightness_range : tuple[float, float]
        ``(min, max)`` additive shift as a fraction of the full 0..255 range.
        Default ``(-0.2, 0.2)``.
    contrast_range : tuple[float, float]
        ``(min, max)`` multiplicative factor. Default ``(0.8, 1.2)``.
    random_state : int or None
        Seed for reproducible adjustments. Default ``None``.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        random_state: Optional[int] = None,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.random_state = random_state

    def _apply(self, image: np.ndarray) -> np.ndarray:
        rng = self._random()
        brightness = rng.uniform(*self.brightness_range) * 255.0
        contrast = rng.uniform(*self.contrast_range)
        out = image.astype(np.float32) * contrast + brightness
        return np.clip(out, 0, 255).astype(np.uint8)
