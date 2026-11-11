from __future__ import annotations

import logging
import os
from typing import Callable, List, Union

import cv2
import numpy as np

__all__ = ["load_image", "save_image", "process_directory"]

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
IMAGE_CHANNELS = (1, 3, 4)  # what PNG/JPG (cv2.imwrite) can hold


def is_writable_image(image: np.ndarray) -> bool:
    """True if ``image`` has 1, 3 or 4 channels and can be written as PNG/JPG."""
    channels = 1 if image.ndim == 2 else image.shape[2]
    return channels in IMAGE_CHANNELS


def write_array(path_stem: str, image: np.ndarray, output_format: str = "png") -> str:
    """Write ``image`` to ``path_stem.<output_format>``, or to ``path_stem.npy`` when it is not an image.

    Multi-channel outputs such as ``Multispectral`` (7 channels) or ``GradientChannels`` (2 channels)
    cannot be stored as PNG/JPG; they are saved losslessly with :func:`numpy.save` instead.
    """
    if not is_writable_image(image):
        path = f"{path_stem}.npy"
        np.save(path, image)
        return path
    path = f"{path_stem}.{output_format}"
    if not cv2.imwrite(path, image):
        raise OSError(f"OpenCV could not write {path}")
    return path


def list_images(directory: str) -> List[str]:
    """Sorted paths of the image files directly inside ``directory``."""
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        The loaded image (HxWxC, uint8).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be decoded as an image.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    """Write an image array to disk.

    Parameters
    ----------
    path : str
        Destination file path.
    image : np.ndarray
        Image array to save.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


def process_directory(
    input_dir: str,
    output_dir: str,
    process_func: Callable[[np.ndarray], Union[np.ndarray, List[np.ndarray]]],
    output_format: str = "png",
    log_every: int = 100,
) -> List[str]:
    """Apply *process_func* to every image in *input_dir* and write results to *output_dir*.

    A function that returns a list (e.g. ``Tile``) writes one file per output,
    suffixed ``_000``, ``_001``, ...

    Returns
    -------
    list[str]
        Paths to the saved output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = list_images(input_dir)
    saved: List[str] = []
    for n, path in enumerate(paths, start=1):
        img = cv2.imread(path)
        if img is None:
            logger.warning("Skipping unreadable file: %s", path)
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        saved.extend(write_outputs(process_func(img), output_dir, stem, output_format))
        if n % log_every == 0:
            logger.info("processed %d/%d images", n, len(paths))
    return saved


def write_outputs(
    result: Union[np.ndarray, List[np.ndarray]],
    output_dir: str,
    stem: str,
    output_format: str = "png",
) -> List[str]:
    """Write one image, or a list of images with numbered suffixes, and return the paths.

    Outputs with a channel count PNG/JPG cannot hold (not 1, 3 or 4) are saved as ``.npy``.
    """
    if isinstance(result, np.ndarray):
        items, names = [result], [stem]
    elif isinstance(result, list) and all(isinstance(r, np.ndarray) for r in result):
        items, names = result, [f"{stem}_{k:03d}" for k in range(len(result))]
    else:
        raise TypeError(f"expected an image or a list of images, got {type(result).__name__}")
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for image, name in zip(items, names):
        paths.append(write_array(os.path.join(output_dir, name), image, output_format))
    return paths
