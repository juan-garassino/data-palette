from __future__ import annotations

from typing import Any, List, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ImageTransform"]


class ImageTransform(BaseEstimator, TransformerMixin):
    """Base class for all DataPalette image transforms.

    Subclasses must implement ``_apply(image)`` which operates on a single
    HxWxC ``numpy.ndarray`` and returns a transformed array.

    ``fit`` is a no-op by default (override for stateful transforms like PCA).
    ``transform`` dispatches a single image or a list of images to ``_apply``.
    """

    def fit(self, X: Any, y: Any = None) -> ImageTransform:
        """No-op fit. Override for stateful transforms."""
        return self

    def transform(
        self, X: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Apply the transform to a single image or list of images."""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return self._apply(X)
        return [self._apply(img) for img in X]

    def _apply(self, image: np.ndarray) -> np.ndarray:
        """Transform a single HxWxC image. Must be overridden by subclasses."""
        raise NotImplementedError


def minmax_to_uint8(x: np.ndarray) -> np.ndarray:
    """Linearly stretch an array to the full 0..255 range (constant arrays map to 0)."""
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)
