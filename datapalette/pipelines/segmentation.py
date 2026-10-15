"""Preprocessing pipeline for image/mask segmentation data."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from datapalette.pipelines.base import TaskPipeline
from datapalette.transforms.spatial import Mirror, Resize, Rotate

__all__ = ["SegmentationPipeline"]

logger = logging.getLogger(__name__)


class SegmentationPipeline(TaskPipeline):
    """Pipeline for preparing image segmentation data.

    Spatial transforms are applied identically to both images and masks.
    Returns ``(X_images, y_masks)``.

    Parameters
    ----------
    dataset : str or None
        Built-in dataset name. Default ``'mnist'``.
    mode : str
        Mask generation mode: ``'binary'``, ``'multiclass'``, or ``'instance'``.
        Default ``'binary'``.
    size : tuple[int, int]
        Target image size. Default ``(64, 64)``.
    """

    def __init__(
        self,
        dataset: Optional[str] = "mnist",
        mode: str = "binary",
        size: Tuple[int, int] = (64, 64),
    ):
        steps = [
            ("resize", Resize(size=size)),
            ("rotate", Rotate(angle=90.0)),
            ("mirror", Mirror(mode="horizontal")),
        ]
        super().__init__(steps=steps, dataset=dataset)
        self.mode = mode
        self.size = size

    def load_and_transform(
        self,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load and transform image/mask pairs.

        Parameters
        ----------
        images_dir : str or None
            Custom images directory. If None, uses built-in dataset.
        masks_dir : str or None
            Custom masks directory. Required if images_dir is provided.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            ``(X_images, y_masks)`` with matching spatial transforms applied.
        """
        if images_dir is not None:
            images = self._load_images_from_dir(images_dir)
            if masks_dir is None:
                raise ValueError("masks_dir is required when using custom images.")
            masks = self._load_images_from_dir(masks_dir)
        else:
            data, labels = self._load_dataset()
            images_raw = data[:, :, :, 0] if data.ndim == 4 else data

            from datapalette.datasets.segmentation import (
                make_binary_masks,
                make_instance_masks,
                make_multiclass_masks,
            )

            if self.mode == "binary":
                mask_arr = make_binary_masks(images_raw)
            elif self.mode == "multiclass":
                mask_arr = make_multiclass_masks(images_raw, labels)
            elif self.mode == "instance":
                comp, inst = make_instance_masks(images_raw, labels, canvas_size=self.size)
                # For instance mode, composites are the images
                images_list = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in comp[:, :, :, 0]]
                masks_list = [m for m in inst[:, :, :, 0]]
                return images_list, masks_list
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")

            images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images_raw]
            masks = [m for m in mask_arr[:, :, :, 0]]

        # Apply same spatial transforms to images and masks
        X = self._transform_images(images)
        y = self._transform_images(masks)
        return X, y
