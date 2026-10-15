"""Synthetic segmentation-mask generators for digit images."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["make_binary_masks", "make_multiclass_masks", "make_instance_masks"]


def make_binary_masks(images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Create binary foreground/background masks from grayscale images.

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(N, H, W)`` or ``(N, H, W, 1)`` with values in ``[0, 255]``.
    threshold : float
        Fraction of 255 to binarize at. Default ``0.5``.

    Returns
    -------
    np.ndarray
        Binary masks of shape ``(N, H, W, 1)`` with values ``{0, 1}`` (uint8).
    """
    if images.ndim == 4:
        images = images[:, :, :, 0]
    masks = (images > threshold * 255).astype(np.uint8)
    return masks[:, :, :, np.newaxis]


def make_multiclass_masks(
    images: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Create pixel-wise class masks from digit images.

    Each foreground pixel is assigned the digit's class label (0-9).
    Background pixels are set to 255 (ignore label).

    Parameters
    ----------
    images : np.ndarray
        Shape ``(N, H, W)`` or ``(N, H, W, 1)``.
    labels : np.ndarray
        Shape ``(N,)`` integer class labels.

    Returns
    -------
    np.ndarray
        Masks of shape ``(N, H, W, 1)`` with pixel values in ``{0..9, 255}``.
    """
    if images.ndim == 4:
        images = images[:, :, :, 0]

    masks = np.full_like(images, 255, dtype=np.uint8)
    for i, label in enumerate(labels):
        masks[i][images[i] > 127] = int(label)

    return masks[:, :, :, np.newaxis]


def make_instance_masks(
    images: np.ndarray,
    labels: np.ndarray,
    n_per_image: int = 3,
    canvas_size: Tuple[int, int] = (64, 64),
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite multiple digit instances onto a canvas with overlap.

    Parameters
    ----------
    images : np.ndarray
        Shape ``(N, H, W)`` or ``(N, H, W, 1)``.
    labels : np.ndarray
        Shape ``(N,)``.
    n_per_image : int
        Number of digits to place on each composite. Default ``3``.
    canvas_size : tuple[int, int]
        ``(height, width)`` of the output canvas. Default ``(64, 64)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(composite_images, instance_masks)``
        — composites: ``(M, H, W, 1)`` uint8
        — instance_masks: ``(M, H, W, 1)`` uint8 with instance IDs (1-indexed, 0 = background)
    """
    if images.ndim == 4:
        images = images[:, :, :, 0]

    n = len(images)
    num_composites = n // n_per_image
    ch, cw = canvas_size
    ih, iw = images.shape[1], images.shape[2]

    composites = np.zeros((num_composites, ch, cw), dtype=np.uint8)
    masks = np.zeros((num_composites, ch, cw), dtype=np.uint8)

    for idx in range(num_composites):
        for inst in range(n_per_image):
            src_idx = idx * n_per_image + inst
            y = np.random.randint(0, max(1, ch - ih))
            x = np.random.randint(0, max(1, cw - iw))
            digit = images[src_idx]

            # Place digit
            y_end = min(y + ih, ch)
            x_end = min(x + iw, cw)
            region = digit[: y_end - y, : x_end - x]
            fg = region > 127

            composites[idx, y:y_end, x:x_end][fg] = region[fg]
            masks[idx, y:y_end, x:x_end][fg] = inst + 1

    return composites[:, :, :, np.newaxis], masks[:, :, :, np.newaxis]
