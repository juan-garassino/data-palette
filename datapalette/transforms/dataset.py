"""GAN/diffusion dataset-prep transforms.

Ported (rewritten as clean, typed ``ImageTransform`` subclasses) from the
StyleGAN-community ``dvschultz/dataset-tools`` operation vocabulary: aspect-
preserving resize with interpolation picking, aligned square cropping, square
padding with border control, multi-crop augmentation, and dominant-colour
naming (from ``sort-color.py``). These are the standard building blocks for
turning a folder of arbitrary images into a square, fixed-size training set.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform

__all__ = [
    "AspectResize",
    "SquareCrop",
    "SquarePad",
    "ManySquares",
    "DominantColor",
]


def _pick_interp(src_shape: Sequence[int], target_hw: Tuple[int, int]) -> int:
    """Pick the interpolation flag for a resize.

    ``INTER_AREA`` is sharper/faster when shrinking; ``INTER_CUBIC`` when
    upscaling. Mirrors ``dataset-tools.pick_interp``.
    """
    h, w = src_shape[:2]
    th, tw = target_hw
    if th * tw < h * w:
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


class AspectResize(ImageTransform):
    """Resize so the longest side equals ``max_size``, preserving aspect ratio.

    Chooses ``INTER_AREA`` / ``INTER_CUBIC`` automatically based on whether the
    image is being shrunk or enlarged. This is the standard first step before
    square-cropping or padding a GAN dataset.

    Parameters
    ----------
    max_size : int
        Target length of the longest side, in pixels. Default ``256``.
    """

    def __init__(self, max_size: int = 256):
        self.max_size = max_size

    def _apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w >= h:
            r = self.max_size / float(w)
            dim = (self.max_size, max(1, int(round(h * r))))  # (w, h)
        else:
            r = self.max_size / float(h)
            dim = (max(1, int(round(w * r))), self.max_size)  # (w, h)
        return cv2.resize(image, dim, interpolation=_pick_interp(image.shape, (dim[1], dim[0])))


class SquareCrop(ImageTransform):
    """Crop the longer dimension so the image becomes square (no rescaling).

    Parameters
    ----------
    h_align : str
        Horizontal alignment when the image is wider than tall: ``'left'``,
        ``'center'``, or ``'right'``. Default ``'center'``.
    v_align : str
        Vertical alignment when the image is taller than wide: ``'top'``,
        ``'center'``, or ``'bottom'``. Default ``'center'``.
    """

    _H_ALIGN = {"left", "center", "right"}
    _V_ALIGN = {"top", "center", "bottom"}

    def __init__(self, h_align: str = "center", v_align: str = "center"):
        self.h_align = h_align
        self.v_align = v_align

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if self.h_align not in self._H_ALIGN:
            raise ValueError(
                f"Invalid h_align '{self.h_align}'. Choose from {sorted(self._H_ALIGN)}"
            )
        if self.v_align not in self._V_ALIGN:
            raise ValueError(
                f"Invalid v_align '{self.v_align}'. Choose from {sorted(self._V_ALIGN)}"
            )

        h, w = image.shape[:2]
        if w > h:
            if self.h_align == "left":
                left = 0
            elif self.h_align == "right":
                left = w - h
            else:
                left = (w - h) // 2
            return image[:, left : left + h]
        if h > w:
            if self.v_align == "top":
                top = 0
            elif self.v_align == "bottom":
                top = h - w
            else:
                top = (h - w) // 2
            return image[top : top + w, :]
        return image


class SquarePad(ImageTransform):
    """Pad the shorter dimension so the image becomes square (no cropping).

    Preserves the whole image, adding a border. Mirrors the padding behaviour
    of ``dataset-tools.makeSquare``.

    Parameters
    ----------
    border_type : str
        ``'constant'`` (solid ``fill`` colour), ``'reflect'``, or ``'replicate'``.
        Default ``'constant'``.
    fill : tuple[int, int, int]
        BGR fill colour used when ``border_type='constant'``. Default black.
    """

    _BORDERS = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }

    def __init__(
        self,
        border_type: str = "constant",
        fill: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.border_type = border_type
        self.fill = fill

    def _apply(self, image: np.ndarray) -> np.ndarray:
        border = self._BORDERS.get(self.border_type)
        if border is None:
            raise ValueError(
                f"Invalid border_type '{self.border_type}'. Choose from {sorted(self._BORDERS)}"
            )

        h, w = image.shape[:2]
        if h == w:
            return image

        diff = abs(h - w)
        pad_a = diff // 2
        pad_b = diff - pad_a  # absorbs the odd pixel so the result is exactly square

        if h > w:
            top, bottom, left, right = 0, 0, pad_a, pad_b
        else:
            top, bottom, left, right = pad_a, pad_b, 0, 0

        return cv2.copyMakeBorder(
            image, top, bottom, left, right, border, value=list(self.fill)
        )


class ManySquares(ImageTransform):
    """Multi-crop augmentation: derive several square crops from one image.

    For a portrait image, crops from the top and bottom; for a landscape image,
    crops from the left and right; near-square images yield a single centre
    crop. ``transform`` always returns a *list* of square crops (like ``Tile``),
    which multiplies dataset size from wide/tall source imagery. Mirrors
    ``dataset-tools.makeManySquares``.

    Parameters
    ----------
    ratio_threshold : float
        How far from square (ratio 1.0) an image must be before it is
        double-cropped. Default ``1.2`` (and its reciprocal ``0.833``).
    """

    def __init__(self, ratio_threshold: float = 1.2):
        self.ratio_threshold = ratio_threshold

    def transform(self, X: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return self._apply(X)
        crops: List[np.ndarray] = []
        for img in X:
            crops.extend(self._apply(img))
        return crops

    def _apply(self, image: np.ndarray) -> List[np.ndarray]:  # type: ignore[override]
        h, w = image.shape[:2]
        ratio = h / w if w else 1.0

        if ratio >= self.ratio_threshold:
            # Tall image: crop from top and bottom.
            return [image[0:w, 0:w], image[h - w : h, 0:w]]
        if ratio <= 1.0 / self.ratio_threshold:
            # Wide image: crop from left and right.
            return [image[0:h, 0:h], image[0:h, w - h : w]]
        # Near-square: single centre crop.
        return [SquareCrop()._apply(image)]


class DominantColor(ImageTransform):
    """Label an image with its nearest named colour (KMeans + LAB deltaE).

    A *describe* transform (not a pixel op): extracts the dominant colour via
    KMeans clustering, then matches it against a palette of named colours using
    perceptual CIELAB distance (deltaE CIE76). ``transform`` returns the colour
    *name* (or a list of names), which is the basis for foldering a dataset by
    colour. Ported from ``dataset-tools/sort-color.py`` (LAB conversion via
    OpenCV rather than scikit-image to avoid an extra dependency).

    Parameters
    ----------
    palette : dict[str, tuple[int, int, int]] | None
        Mapping of colour name -> RGB triple to match against. Defaults to a
        small W3C basic-colour set when ``None``.
    n_clusters : int
        Number of KMeans clusters used to find the dominant colour. Default ``5``.
    """

    _DEFAULT_PALETTE = {
        "red": (255, 0, 0),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "purple": (128, 0, 128),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
    }

    def __init__(
        self,
        palette: Optional[dict] = None,
        n_clusters: int = 5,
    ):
        self.palette = palette if palette is not None else dict(self._DEFAULT_PALETTE)
        self.n_clusters = n_clusters

    def transform(self, X: Union[np.ndarray, List[np.ndarray]]) -> Union[str, List[str]]:
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return self._label(X)
        return [self._label(img) for img in X]

    def dominant_rgb(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Return the dominant colour of a BGR image as an RGB triple."""
        from sklearn.cluster import KMeans

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)

        k = min(self.n_clusters, len(np.unique(pixels, axis=0)))
        k = max(1, k)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pixels)
        labels, counts = np.unique(km.labels_, return_counts=True)
        dominant = km.cluster_centers_[labels[int(np.argmax(counts))]]
        return tuple(int(round(c)) for c in dominant)  # type: ignore[return-value]

    @staticmethod
    def _rgb_to_lab(rgb: Sequence[int]) -> np.ndarray:
        """Convert a single RGB triple to CIELAB (float, L in [0,100])."""
        bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
        # OpenCV packs LAB into uint8: L*255/100, a+128, b+128. Undo the scaling
        # so deltaE is computed in true CIELAB units.
        return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0])

    def _label(self, image: np.ndarray) -> str:
        dom_lab = self._rgb_to_lab(self.dominant_rgb(image))

        best_name = ""
        best_diff = float("inf")
        for name, rgb in self.palette.items():
            diff = float(np.linalg.norm(dom_lab - self._rgb_to_lab(rgb)))  # deltaE CIE76
            if diff < best_diff:
                best_diff = diff
                best_name = name
        return best_name

    def _apply(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - not used
        raise NotImplementedError("DominantColor returns a label; use transform().")
