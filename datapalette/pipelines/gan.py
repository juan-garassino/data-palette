from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from datapalette.pipelines.base import TaskPipeline
from datapalette.transforms.noise import BrightnessContrast
from datapalette.transforms.spatial import Mirror, Resize, Rotate

__all__ = ["GANPipeline"]

logger = logging.getLogger(__name__)


class GANPipeline(TaskPipeline):
    """Pipeline for preparing data for GANs.

    Returns ``(X, None)`` — unsupervised, X = augmented images.

    Parameters
    ----------
    dataset : str or None
        Built-in dataset name (``'mnist'`` or ``'fashion_mnist'``). Default ``'mnist'``.
    size : tuple[int, int]
        Target image size ``(height, width)``. Default ``(64, 64)``.
    angle : float
        Rotation angle in degrees. Default ``90.0``.
    """

    def __init__(
        self,
        dataset: Optional[str] = "mnist",
        size: Tuple[int, int] = (64, 64),
        angle: float = 90.0,
    ):
        steps = [
            ("resize", Resize(size=size)),
            ("rotate", Rotate(angle=angle)),
            ("mirror", Mirror(mode="horizontal")),
            ("brightness_contrast", BrightnessContrast()),
        ]
        super().__init__(steps=steps, dataset=dataset)
        self.size = size
        self.angle = angle

    def load_and_transform(
        self,
        images_dir: Optional[str] = None,
        video_path: Optional[str] = None,
        fps: int = 1,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], None]:
        """Load and augment images for GAN training.

        Parameters
        ----------
        images_dir : str or None
            Directory of custom images. If None, loads the built-in dataset.
        video_path : str or None
            Path to a video file to extract frames from.
        fps : int
            Frames per second for video extraction.

        Returns
        -------
        tuple[list[np.ndarray], None]
            ``(X, None)`` where X is a list of augmented images.
        """
        if video_path is not None:
            import tempfile

            from datapalette.io.video import extract_frames

            with tempfile.TemporaryDirectory() as tmp:
                extract_frames(video_path, tmp, fps=fps)
                images = self.load_images(tmp)
        elif images_dir is not None:
            images = self.load_images(images_dir)
        else:
            data, _ = self._load_dataset()
            # Convert (N, 28, 28, 1) grayscale to BGR for consistency
            images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in data]

        X = self.transform_images(images)
        return X, None
