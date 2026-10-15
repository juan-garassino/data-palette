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

    def __sklearn_is_fitted__(self) -> bool:
        """Transforms are usable without ``fit`` (stateless, or self-fitting on first use).

        Recent scikit-learn versions call ``check_is_fitted`` from
        ``Pipeline.transform``; without this hook an unfitted pipeline of
        stateless transforms raises ``NotFittedError``.
        """
        return True

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
