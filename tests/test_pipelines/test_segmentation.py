from __future__ import annotations

import os

import cv2
import numpy as np

from datapalette.pipelines import SegmentationPipeline


class TestSegmentationPipeline:
    def test_load_and_transform_custom(self, tmp_path):
        images_dir = str(tmp_path / "images")
        masks_dir = str(tmp_path / "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (64, 64, 3), dtype=np.uint8) * 255
            cv2.imwrite(os.path.join(images_dir, f"img_{i:03d}.png"), img)
            cv2.imwrite(os.path.join(masks_dir, f"img_{i:03d}.png"), mask)

        pipe = SegmentationPipeline(dataset=None, mode="binary", size=(32, 32))
        X, y = pipe.load_and_transform(images_dir=images_dir, masks_dir=masks_dir)
        assert len(X) == 3
        assert len(y) == 3
        for img in X:
            assert img.shape == (32, 32, 3)
        for mask in y:
            assert mask.shape == (32, 32, 3)

    def test_raises_without_masks_dir(self, tmp_image_dir):
        pipe = SegmentationPipeline(dataset=None)
        try:
            pipe.load_and_transform(images_dir=tmp_image_dir, masks_dir=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "masks_dir" in str(e)

    def test_mask_values_preserved_through_spatial_transforms(self, tmp_path):
        """Masks must only contain label values present in the input (nearest-neighbour).

        Bilinear resize/rotate would blend e.g. class 3 and ignore-label 255 into
        invalid ids like 129.
        """
        images_dir = str(tmp_path / "images")
        masks_dir = str(tmp_path / "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        mask = np.full((64, 64), 255, dtype=np.uint8)
        mask[10:40, 5:30] = 3
        mask[35:60, 25:55] = 7
        cv2.imwrite(os.path.join(images_dir, "a.png"), np.zeros((64, 64, 3), np.uint8))
        cv2.imwrite(os.path.join(masks_dir, "a.png"), np.dstack([mask] * 3))

        pipe = SegmentationPipeline(dataset=None, mode="multiclass", size=(45, 45))
        X, y = pipe.load_and_transform(images_dir=images_dir, masks_dir=masks_dir)
        assert y[0].shape[:2] == (45, 45)
        assert set(np.unique(y[0]).tolist()) <= {0, 3, 7, 255}

    def test_image_pipeline_still_uses_linear_interpolation(self):
        pipe = SegmentationPipeline(dataset=None)
        assert pipe.sklearn_pipeline.named_steps["resize"].interpolation == cv2.INTER_LINEAR
