from __future__ import annotations

import numpy as np

from datapalette.datasets.segmentation import (
    make_binary_masks,
    make_instance_masks,
    make_multiclass_masks,
)


class TestMakeBinaryMasks:
    def test_output_shape_and_values(self):
        images = np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8)
        masks = make_binary_masks(images)
        assert masks.shape == (5, 28, 28, 1)
        assert masks.dtype == np.uint8
        assert set(np.unique(masks)).issubset({0, 1})

    def test_4d_input(self):
        images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        masks = make_binary_masks(images)
        assert masks.shape == (5, 28, 28, 1)


class TestMakeMulticlassMasks:
    def test_output_shape(self):
        images = np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8)
        labels = np.array([0, 3, 7, 2, 9], dtype=np.int64)
        masks = make_multiclass_masks(images, labels)
        assert masks.shape == (5, 28, 28, 1)
        assert masks.dtype == np.uint8

    def test_foreground_has_class_label(self):
        # Create an image that is all white (foreground)
        images = np.full((1, 28, 28), 200, dtype=np.uint8)
        labels = np.array([5], dtype=np.int64)
        masks = make_multiclass_masks(images, labels)
        # All pixels > 127 should have label 5
        assert np.all(masks[0, :, :, 0] == 5)


class TestMakeInstanceMasks:
    def test_output_shapes(self):
        images = np.random.randint(0, 255, (9, 28, 28), dtype=np.uint8)
        labels = np.arange(9, dtype=np.int64)
        composites, instance_masks = make_instance_masks(images, labels, n_per_image=3, canvas_size=(64, 64))
        # 9 images / 3 per composite = 3 composites
        assert composites.shape == (3, 64, 64, 1)
        assert instance_masks.shape == (3, 64, 64, 1)
        assert composites.dtype == np.uint8
        assert instance_masks.dtype == np.uint8

    def test_instance_ids_are_1_indexed(self):
        # Use high-contrast images to ensure foreground pixels
        images = np.full((3, 28, 28), 200, dtype=np.uint8)
        labels = np.array([0, 1, 2], dtype=np.int64)
        _, inst = make_instance_masks(images, labels, n_per_image=3, canvas_size=(64, 64))
        unique = np.unique(inst)
        # Should have 0 (background) and some of 1, 2, 3
        assert 0 in unique
        assert any(v > 0 for v in unique)
