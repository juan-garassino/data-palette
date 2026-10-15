"""Preprocessing pipeline for diffusion-model training data."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from datapalette.pipelines.base import TaskPipeline
from datapalette.transforms.color import PCAColorAugmentation
from datapalette.transforms.spatial import RandomCrop, Resize

__all__ = ["DiffusionPipeline"]

logger = logging.getLogger(__name__)


class DiffusionPipeline(TaskPipeline):
    """Pipeline for preparing diffusion model training data.

    Returns ``(X_clean, y_labels)`` — clean preprocessed images and class
    labels for conditional generation. Noise is added during training, not here.

    Parameters
    ----------
    dataset : str or None
        Built-in dataset name. Default ``'mnist'``.
    size : tuple[int, int]
        Target image size. Default ``(64, 64)``.
    """

    def __init__(
        self,
        dataset: Optional[str] = "mnist",
        size: Tuple[int, int] = (64, 64),
    ):
        steps = [
            ("resize", Resize(size=size)),
            ("pca_color", PCAColorAugmentation(alpha_std=0.1)),
            ("random_crop", RandomCrop(crop_size=size)),
        ]
        super().__init__(steps=steps, dataset=dataset)
        self.size = size

    def load_and_transform(
        self,
        images_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], Any]:
        """Load and preprocess images for diffusion model training.

        Parameters
        ----------
        images_dir : str or None
            Custom images directory. If None, uses built-in dataset.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray | None]
            ``(X_clean, y_labels)`` where y_labels are class labels (int) if
            using a built-in dataset, or None for custom images.
        """
        if images_dir is not None:
            images = self._load_images_from_dir(images_dir)
            X = self._transform_images(images)
            return X, None

        data, labels = self._load_dataset()
        images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in data[:, :, :, 0]]
        X = self._transform_images(images)
        return X, labels
