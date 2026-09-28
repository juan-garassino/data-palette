from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform, require_channels

__all__ = ["PCAColorAugmentation", "ConvertColorSpace", "EnhanceGreen", "Multispectral"]

logger = logging.getLogger(__name__)


class PCAColorAugmentation(ImageTransform):
    """PCA-based color augmentation (AlexNet-style).

    ``fit`` learns the colour statistics of a dataset (AlexNet computes them over all of
    ImageNet); ``transform`` then shifts each image along those principal colour axes.
    Without ``fit``, each image is perturbed along its *own* principal axes, and nothing
    is stored on the transform.

    Parameters
    ----------
    alpha_std : float
        Standard deviation used to sample perturbation magnitudes. Default ``0.1``.
    random_state : int or None
        Seed for reproducible perturbations. Default ``None``.
    """

    def __init__(self, alpha_std: float = 0.1, random_state: Optional[int] = None):
        self.alpha_std = alpha_std
        self.random_state = random_state

    @staticmethod
    def _stats(pixels: np.ndarray):
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        std[std == 0] = 1.0
        cov = np.cov((pixels - mean) / std, rowvar=False) + np.eye(3) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(cov)
        return mean, std, eigvals, eigvecs

    def fit(self, X: Any, y: Any = None) -> PCAColorAugmentation:
        """Learn colour statistics from one image or a list of images."""
        images = [X] if isinstance(X, np.ndarray) and X.ndim == 3 else list(X)
        for img in images:
            require_channels(img, (3,), type(self).__name__)
        pixels = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
        self.mean_, self.std_, self.eigvals_, self.eigvecs_ = self._stats(pixels)
        return self

    def _apply(self, image: np.ndarray) -> np.ndarray:
        require_channels(image, (3,), type(self).__name__)
        pixels = image.reshape(-1, 3).astype(np.float32)
        if hasattr(self, "eigvals_"):
            mean, std, eigvals, eigvecs = self.mean_, self.std_, self.eigvals_, self.eigvecs_
        else:
            mean, std, eigvals, eigvecs = self._stats(pixels)
        alphas = self._random().normal(0, self.alpha_std, 3)
        delta = eigvecs.dot(alphas * eigvals)
        result = ((pixels - mean) / std + delta) * std + mean
        return np.clip(result, 0, 255).reshape(image.shape).astype(np.uint8)


class ConvertColorSpace(ImageTransform):
    """Convert an image to a different colour space.

    Parameters
    ----------
    target : str
        Target colour space: ``'hsv'``, ``'lab'``, or ``'gray'``. Default ``'hsv'``.
    """

    _CODES = {
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2LAB,
        "gray": cv2.COLOR_BGR2GRAY,
    }

    def __init__(self, target: str = "hsv"):
        self.target = target

    def _apply(self, image: np.ndarray) -> np.ndarray:
        require_channels(image, (3,), type(self).__name__)
        code = self._CODES.get(self.target)
        if code is None:
            raise ValueError(f"Unknown target '{self.target}'. Choose from {list(self._CODES)}")
        result = cv2.cvtColor(image, code)
        if result.ndim == 2:
            result = result[:, :, np.newaxis]
        return result


class EnhanceGreen(ImageTransform):
    """Enhance the green channel using CLAHE."""

    def _apply(self, image: np.ndarray) -> np.ndarray:
        require_channels(image, (3,), type(self).__name__)
        b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enhanced = clahe.apply(g)
        return cv2.merge([b, g_enhanced, r])


class Multispectral(ImageTransform):
    """Simulate 7-channel multispectral imaging from a BGR image.

    Output channels: B, G, R, Yellow, Cyan, Magenta, simulated NIR.
    """

    def _apply(self, image: np.ndarray) -> np.ndarray:
        require_channels(image, (3,), type(self).__name__)
        b, g, r = cv2.split(image)
        yellow = cv2.addWeighted(g, 0.5, r, 0.5, 0)
        cyan = cv2.addWeighted(b, 0.5, g, 0.5, 0)
        magenta = cv2.addWeighted(b, 0.5, r, 0.5, 0)
        nir = np.clip(r.astype(np.int16) + 50, 0, 255).astype(np.uint8)  # saturating, like cv2.add
        return np.dstack((b, g, r, yellow, cyan, magenta, nir))
