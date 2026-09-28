from __future__ import annotations

import os
from typing import Any, Callable, Optional

import cv2
import numpy as np

__all__ = ["DataPaletteDataset"]


class DataPaletteDataset:
    """PyTorch Dataset adapter for DataPalette pipelines.

    Lazy-loads images on ``__getitem__`` and optionally applies a pipeline.

    Parameters
    ----------
    images_dir : str
        Directory containing source images.
    pipeline : object or None
        A ``TaskPipeline`` (or any ``ImageTransform``) applied to each image.
    masks_dir : str or None
        Optional directory of mask images (for segmentation).
    transform : callable or None
        Additional transform applied after the pipeline (e.g. torchvision).
    """

    def __init__(
        self,
        images_dir: str,
        pipeline: Any = None,
        masks_dir: Optional[str] = None,
        transform: Optional[Callable[..., Any]] = None,
    ):
        self.images_dir = images_dir
        self.pipeline = pipeline
        self.masks_dir = masks_dir
        self.transform = transform

        from datapalette.io.batch import list_images

        self._image_files = [os.path.basename(p) for p in list_images(images_dir)]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Any:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for DataPaletteDataset. Install with: pip install datapalette[torch]"
            ) from e

        img_path = os.path.join(self.images_dir, self._image_files[idx])
        image: Any = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Apply pipeline transforms
        if self.pipeline is not None:
            transform = getattr(self.pipeline, "transforms", self.pipeline)
            image = transform.transform(image)

        mask = None
        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, self._image_files[idx])
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            # HWC -> CHW, float32 [0,1]
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)

        if self.transform is not None:
            image = self.transform(image)

        if self.masks_dir is None:
            return image  # a bare tensor collates cleanly; (image, None) did not
        if mask is None:
            raise FileNotFoundError(f"no mask named {self._image_files[idx]} in {self.masks_dir}")
        return image, torch.from_numpy(mask.astype(np.int64))
