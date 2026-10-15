"""Video frame extraction and random-crop helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

__all__ = ["extract_frames", "n_croppings_from_frame"]

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 1,
    output_size: Optional[Tuple[int, int]] = None,
) -> int:
    """Extract frames from a video file at the requested rate.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : str
        Directory where extracted frames are saved.
    fps : int
        Target frames per second. Default ``1``.
    output_size : tuple[int, int] or None
        Optional ``(width, height)`` to resize each frame.

    Returns
    -------
    int
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = max(1, int(video_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                if output_size:
                    frame = cv2.resize(frame, output_size)
                path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(path, frame)
                saved_count += 1
            frame_count += 1
            pbar.update(1)

    cap.release()
    logger.info("Extracted %d frames from %s", saved_count, video_path)
    return saved_count


def n_croppings_from_frame(
    input_frame: str,
    output_dir: str,
    num_crops: int,
    crop_size: Tuple[int, int],
) -> list[str]:
    """Generate *num_crops* random crops from a single image.

    Parameters
    ----------
    input_frame : str
        Path to the source image.
    output_dir : str
        Directory where crops are saved.
    num_crops : int
        Number of random crops to create.
    crop_size : tuple[int, int]
        ``(width, height)`` of each crop.

    Returns
    -------
    list[str]
        Paths to saved crop files.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame = cv2.imread(input_frame)
    if frame is None:
        raise ValueError(f"Could not read image: {input_frame}")

    h, w = frame.shape[:2]
    cw, ch = crop_size
    paths: list[str] = []

    for i in range(num_crops):
        max_x = max(0, w - cw)
        max_y = max(0, h - ch)
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        crop = frame[y : y + ch, x : x + cw]
        if crop.shape[0] != ch or crop.shape[1] != cw:
            crop = cv2.resize(crop, (cw, ch))
        name = f"crop_{i:03d}_{os.path.basename(input_frame)}"
        path = os.path.join(output_dir, name)
        cv2.imwrite(path, crop)
        paths.append(path)

    return paths
