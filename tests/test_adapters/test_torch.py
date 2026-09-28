from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestDataPaletteDataset:
    def test_len(self, tmp_image_dir):
        from datapalette.adapters.torch import DataPaletteDataset

        ds = DataPaletteDataset(images_dir=tmp_image_dir)
        assert len(ds) == 3

    def test_getitem_returns_tensor(self, tmp_image_dir):
        from datapalette.adapters.torch import DataPaletteDataset

        ds = DataPaletteDataset(images_dir=tmp_image_dir)
        image = ds[0]
        assert isinstance(image, torch.Tensor)
        # CHW format, float32
        assert image.ndim == 3
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_dataloader_collates_without_masks(self, tmp_image_dir):
        from torch.utils.data import DataLoader

        from datapalette import GANPipeline
        from datapalette.adapters import DataPaletteDataset

        ds = DataPaletteDataset(images_dir=tmp_image_dir, pipeline=GANPipeline(dataset=None, size=(16, 16)))
        batch = next(iter(DataLoader(ds, batch_size=3)))
        assert batch.shape == (3, 3, 16, 16)

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
