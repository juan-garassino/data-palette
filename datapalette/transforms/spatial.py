from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform, interpolation_flag, keep_channel_axis

__all__ = ["Rotate", "Mirror", "RandomCrop", "Tile", "Resize"]


class Rotate(ImageTransform):
    """Rotate an image by a fixed angle (degrees, counter-clockwise).

    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default 90.
    expand : bool
        Grow the canvas so nothing is cropped (e.g. a 90 degree turn of a non-square
        image swaps height and width). Default ``False`` keeps the input size.
    interpolation : str
        ``'linear'`` (default) for images, ``'nearest'`` for label masks.
    """

    def __init__(self, angle: float = 90.0, expand: bool = False, interpolation: str = "linear"):
        self.angle = angle
        self.expand = expand
        self.interpolation = interpolation

    def _apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.angle % 90 == 0 and (self.expand or h == w):
            # exact quarter turns: no interpolation, no half-pixel black seam
            return np.ascontiguousarray(np.rot90(image, k=int(self.angle // 90) % 4))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, 1)
        out_w, out_h = w, h
        if self.expand:
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            out_w, out_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
            M[0, 2] += out_w / 2 - w / 2
            M[1, 2] += out_h / 2 - h / 2
        flag = interpolation_flag(self.interpolation)
        return keep_channel_axis(image, cv2.warpAffine(image, M, (out_w, out_h), flags=flag))


class Mirror(ImageTransform):
    """Flip an image horizontally, vertically, or both.

    Parameters
    ----------
    mode : str
        One of ``'horizontal'``, ``'vertical'``, or ``'both'``. Default ``'horizontal'``.
    """

    _FLIP_CODES = {"horizontal": 1, "vertical": 0, "both": -1}

    def __init__(self, mode: str = "horizontal"):
        self.mode = mode

    def _apply(self, image: np.ndarray) -> np.ndarray:
        code = self._FLIP_CODES.get(self.mode)
        if code is None:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from {list(self._FLIP_CODES)}")
        return keep_channel_axis(image, cv2.flip(image, code))


class RandomCrop(ImageTransform):
    """Extract a random crop from an image.

    If the image is smaller than ``crop_size``, it is resized up first.

    Parameters
    ----------
    crop_size : tuple[int, int]
        ``(height, width)`` of the crop. Default ``(224, 224)``.
    random_state : int or None
        Seed for reproducible crops. Default ``None``.
    """

    def __init__(self, crop_size: Tuple[int, int] = (224, 224), random_state: Optional[int] = None):
        self.crop_size = crop_size
        self.random_state = random_state

    def _apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ch, cw = self.crop_size

        # Guard: resize if image is smaller than crop
        if h < ch or w < cw:
            image = keep_channel_axis(image, cv2.resize(image, (max(w, cw), max(h, ch))))
            h, w = image.shape[:2]

        rng = self._random()
        top = int(rng.integers(0, h - ch + 1))
        left = int(rng.integers(0, w - cw + 1))
        return image[top : top + ch, left : left + cw]


class Tile(ImageTransform):
    """Split an image into overlapping tiles.

    ``transform`` always returns a *list* of tiles (even for a single image).

    Parameters
    ----------
    tile_size : tuple[int, int]
        ``(height, width)`` of each tile. Default ``(256, 256)``.
    overlap : float
        Fraction of tile size used as overlap, in ``[0, 1)``. Default ``0.0``.
    """

    def __init__(self, tile_size: Tuple[int, int] = (256, 256), overlap: float = 0.0):
        self.tile_size = tile_size
        self.overlap = overlap

    def transform(self, X: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return self._apply(X)
        tiles: List[np.ndarray] = []
        for img in X:
            tiles.extend(self._apply(img))
        return tiles

    def _apply(self, image: np.ndarray) -> List[np.ndarray]:  # type: ignore[override]
        h, w = image.shape[:2]
        th, tw = self.tile_size
        stride_h = max(1, int(th * (1 - self.overlap)))
        stride_w = max(1, int(tw * (1 - self.overlap)))

        tiles: List[np.ndarray] = []
        for y in range(0, h - th + 1, stride_h):
            for x in range(0, w - tw + 1, stride_w):
                tiles.append(image[y : y + th, x : x + tw])
        return tiles


class Resize(ImageTransform):
    """Resize an image to a fixed ``(height, width)``.

    Parameters
    ----------
    size : tuple[int, int]
        Target ``(height, width)``.
    interpolation : str
        ``'linear'`` (default) for images, ``'nearest'`` for label masks, ``'area'`` for
        downscaling.
    """

    def __init__(self, size: Tuple[int, int] = (256, 256), interpolation: str = "linear"):
        self.size = size
        self.interpolation = interpolation

    def _apply(self, image: np.ndarray) -> np.ndarray:
        flag = interpolation_flag(self.interpolation)
        return keep_channel_axis(image, cv2.resize(image, (self.size[1], self.size[0]), interpolation=flag))
