from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform

__all__ = ["PCAColorAugmentation", "ConvertColorSpace", "EnhanceGreen", "Multispectral"]

logger = logging.getLogger(__name__)


class PCAColorAugmentation(ImageTransform):
    """PCA-based color augmentation (AlexNet-style).

    This is a *stateful* transform: ``fit`` computes eigen-decomposition on the
    provided images, and ``transform`` applies random perturbations along the
    principal colour axes.

    Parameters
    ----------
    alpha_std : float
        Standard deviation used to sample perturbation magnitudes. Default ``0.1``.
    """

    def __init__(self, alpha_std: float = 0.1):
        self.alpha_std = alpha_std
        self.eigvals_: np.ndarray | None = None
        self.eigvecs_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> PCAColorAugmentation:
        """Compute eigen-decomposition from a set of images."""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            pixels: np.ndarray = X.reshape(-1, 3).astype(np.float32)
        else:
            pixels = np.vstack([img.reshape(-1, 3) for img in X]).astype(np.float32)

        self.mean_ = pixels.mean(axis=0)
        self.std_ = pixels.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        normed = (pixels - self.mean_) / self.std_
        cov = np.cov(normed, rowvar=False) + np.eye(3) * 1e-6
        self.eigvals_, self.eigvecs_ = np.linalg.eigh(cov)
        return self

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if self.eigvals_ is None:
            # Fall back: fit on this single image
            self.fit(image)

        assert self.eigvals_ is not None
        assert self.eigvecs_ is not None
        assert self.mean_ is not None
        assert self.std_ is not None

        pixels = image.reshape(-1, 3).astype(np.float32)
        normed = (pixels - self.mean_) / self.std_
        alphas = np.random.normal(0, self.alpha_std, 3)
        delta = self.eigvecs_.dot(alphas * self.eigvals_)
        result = (normed + delta) * self.std_ + self.mean_
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
        b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enhanced = clahe.apply(g)
        return cv2.merge([b, g_enhanced, r])


class Multispectral(ImageTransform):
    """Simulate 7-channel multispectral imaging from a BGR image.

    Output channels: B, G, R, Yellow, Cyan, Magenta, simulated NIR.
    """

    def _apply(self, image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        yellow = cv2.addWeighted(g, 0.5, r, 0.5, 0)
        cyan = cv2.addWeighted(b, 0.5, g, 0.5, 0)
        magenta = cv2.addWeighted(b, 0.5, r, 0.5, 0)
        nir = cv2.add(r, np.full_like(r, 50))  # saturating, same as the scalar form
        return np.dstack((b, g, r, yellow, cyan, magenta, nir))
