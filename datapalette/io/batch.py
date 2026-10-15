"""Single-image and directory-level image I/O."""

from __future__ import annotations

import logging
import os
from typing import Callable, List

import cv2
import numpy as np
from tqdm import tqdm

__all__ = ["load_image", "save_image", "process_directory"]

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


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
    process_func: Callable[[np.ndarray], np.ndarray],
    batch_size: int = 32,
    output_format: str = "png",
    desc: str = "Processing images",
) -> List[str]:
    """Apply *process_func* to every image in *input_dir* and write results to *output_dir*.

    Parameters
    ----------
    input_dir : str
        Source directory containing images.
    output_dir : str
        Destination directory for processed images.
    process_func : callable
        ``(np.ndarray) -> np.ndarray`` applied to each image.
    batch_size : int
        Cosmetic batch size for the progress bar.
    output_format : str
        Output file extension (without dot). Default ``'png'``.
    desc : str
        Progress bar description.

    Returns
    -------
    list[str]
        Paths to the saved output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    saved: List[str] = []
    for i in tqdm(range(0, len(image_files), batch_size), desc=desc, unit="batch"):
        batch = image_files[i : i + batch_size]
        for img_file in batch:
            img = cv2.imread(os.path.join(input_dir, img_file))
            if img is None:
                logger.warning("Skipping unreadable file: %s", img_file)
                continue
            processed = process_func(img)
            out_name = f"{os.path.splitext(img_file)[0]}.{output_format}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, processed)
            saved.append(out_path)

    return saved
