from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

__all__ = ["DataPaletteDataset"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


class DataPaletteDataset:
    """PyTorch Dataset adapter for DataPalette pipelines.

    Lazy-loads images on ``__getitem__`` and optionally applies a pipeline.

    Parameters
    ----------
    images_dir : str
        Directory containing source images.
    pipeline : object or None
        A ``TaskPipeline`` instance. If provided its sklearn pipeline is used
        for transforms.
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

        self._image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for DataPaletteDataset. "
                "Install with: pip install datapalette[torch]"
            ) from e

        img_path = os.path.join(self.images_dir, self._image_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Apply pipeline transforms
        if self.pipeline is not None:
            image = self.pipeline.sklearn_pipeline.transform(image)

        mask = None
        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, self._image_files[idx])
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Keep the mask spatially aligned with the image: pipelines that
                # define a mask_pipeline (e.g. SegmentationPipeline) apply the same
                # geometry with nearest-neighbour interpolation.
                mask_pipeline = getattr(self.pipeline, "mask_pipeline", None)
                if mask is not None and mask_pipeline is not None:
                    mask = mask_pipeline.transform(mask)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            # HWC -> CHW, float32 [0,1]
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)  # type: ignore[assignment]

        if self.transform is not None:
            image = self.transform(image)

        if mask is not None:
            label = torch.from_numpy(mask.astype(np.int64))
        else:
            label = None  # type: ignore[assignment]

        return image, label
