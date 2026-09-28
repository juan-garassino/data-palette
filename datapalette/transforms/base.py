from __future__ import annotations

import inspect
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np

__all__ = ["ImageTransform", "Compose", "minmax_to_uint8"]

Images = Union[np.ndarray, List[np.ndarray]]


class ImageTransform:
    """Base class for all DataPalette image transforms.

    Subclasses implement ``_apply(image)`` for a single H x W (x C) ``numpy.ndarray``.
    ``transform`` accepts one image or a list of images; ``fit`` is a no-op unless a
    subclass learns something from data (see ``PCAColorAugmentation``).

    No scikit-learn dependency, but the estimator protocol is implemented
    (``get_params``/``set_params``/``fit``/``transform``/``fit_transform`` plus sklearn's
    tag and fitted hooks), so transforms still work inside ``sklearn.pipeline.Pipeline``.
    """

    # -- parameters -----------------------------------------------------------------
    @classmethod
    def _param_names(cls) -> List[str]:
        sig = inspect.signature(cls.__init__)
        return [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self, deep: bool = True) -> dict:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> ImageTransform:
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    # -- estimator protocol -----------------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> ImageTransform:
        """No-op fit. Override for transforms that learn from data."""
        return self

    def transform(self, X: Images) -> Images:
        """Apply the transform to a single image or a list of images."""
        if isinstance(X, np.ndarray) and X.ndim in (2, 3):
            return self._apply(X)
        return [self._apply(img) for img in X]

    def fit_transform(self, X: Images, y: Any = None) -> Images:
        return self.fit(X, y).transform(X)

    __call__ = transform

    def _apply(self, image: np.ndarray) -> np.ndarray:
        """Transform a single image. Must be overridden by subclasses."""
        raise NotImplementedError

    # -- scikit-learn interop (only called when sklearn is installed) -----------------
    def __sklearn_is_fitted__(self) -> bool:
        return True

    def __sklearn_tags__(self):
        from sklearn.base import BaseEstimator, TransformerMixin

        class _Probe(TransformerMixin, BaseEstimator):
            pass

        tags = _Probe().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class Compose(ImageTransform):
    """Apply transforms in order. Accepts transforms or ``(name, transform)`` pairs.

    >>> Compose([Resize((64, 64)), ("flip", Mirror())])
    """

    def __init__(self, steps: Sequence[Union[ImageTransform, Tuple[str, ImageTransform]]]):
        self.steps = list(steps)

    @property
    def named_steps(self) -> dict:
        out = {}
        for i, step in enumerate(self.steps):
            name, t = (
                step if isinstance(step, tuple) else (f"{type(step).__name__.lower()}_{i}", step)
            )
            out[name] = t
        return out

    def _transforms(self) -> Iterable[ImageTransform]:
        return self.named_steps.values()

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, key: Union[int, str]) -> ImageTransform:
        return self.named_steps[key] if isinstance(key, str) else list(self._transforms())[key]

    def fit(self, X: Any, y: Any = None) -> Compose:
        for t in self._transforms():
            X = t.fit_transform(X, y)
        return self

    def transform(self, X: Images) -> Images:
        for t in self._transforms():
            X = t.transform(X)
        return X

    __call__ = transform

    def _apply(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image)  # type: ignore[return-value]


def minmax_to_uint8(x: np.ndarray) -> np.ndarray:
    """Linearly stretch an array to the full 0..255 range (constant arrays map to 0)."""
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)
