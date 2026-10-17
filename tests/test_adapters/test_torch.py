from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False


@pytest.mark.skipif(not torch_available, reason="torch not installed")
class TestDataPaletteDataset:
    def test_len(self, tmp_image_dir):
        from datapalette.adapters.torch import DataPaletteDataset

        ds = DataPaletteDataset(images_dir=tmp_image_dir)
        assert len(ds) == 3

    def test_getitem_returns_tensor(self, tmp_image_dir):
        from datapalette.adapters.torch import DataPaletteDataset

        ds = DataPaletteDataset(images_dir=tmp_image_dir)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        # CHW format, float32
        assert image.ndim == 3
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert label is None

    def test_with_masks(self, tmp_path):
        from datapalette.adapters.torch import DataPaletteDataset

        images_dir = str(tmp_path / "images")
        masks_dir = str(tmp_path / "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        for i in range(2):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            mask = np.random.randint(0, 5, (32, 32), dtype=np.uint8)
            cv2.imwrite(os.path.join(images_dir, f"img_{i:03d}.png"), img)
            cv2.imwrite(os.path.join(masks_dir, f"img_{i:03d}.png"), mask)

        ds = DataPaletteDataset(images_dir=images_dir, masks_dir=masks_dir)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.int64


@pytest.mark.skipif(not torch_available, reason="torch not installed")
class TestDataPaletteDatasetMaskAlignment:
    def test_mask_gets_same_spatial_transforms_as_image(self, tmp_path):
        """With a SegmentationPipeline the mask must be resized/rotated like the image."""
        from datapalette.adapters.torch import DataPaletteDataset
        from datapalette.pipelines import SegmentationPipeline

        images_dir = str(tmp_path / "images")
        masks_dir = str(tmp_path / "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)
        cv2.imwrite(os.path.join(images_dir, "a.png"), np.zeros((40, 60, 3), np.uint8))
        mask = np.zeros((40, 60), np.uint8)
        mask[:, :30] = 4
        cv2.imwrite(os.path.join(masks_dir, "a.png"), mask)

        pipe = SegmentationPipeline(dataset=None, size=(16, 16))
        ds = DataPaletteDataset(images_dir=images_dir, pipeline=pipe, masks_dir=masks_dir)
        image, label = ds[0]
        assert tuple(image.shape[1:]) == (16, 16)
        assert tuple(label.shape) == (16, 16)
        assert set(label.unique().tolist()) <= {0, 4}
