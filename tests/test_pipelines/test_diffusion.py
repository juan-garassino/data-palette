from __future__ import annotations

import numpy as np

from datapalette.pipelines import DiffusionPipeline


class TestDiffusionPipeline:
    def test_load_and_transform_from_images_dir(self, tmp_image_dir):
        pipe = DiffusionPipeline(dataset=None, size=(32, 32))
        X, y = pipe.load_and_transform(images_dir=tmp_image_dir)
        assert y is None
        assert isinstance(X, list)
        assert len(X) == 3
        for img in X:
            assert img.shape == (32, 32, 3)
            assert img.dtype == np.uint8

    def test_transforms_accessible(self):
        pipe = DiffusionPipeline(dataset=None, size=(64, 64))
        sk_pipe = pipe.transforms
        assert sk_pipe is not None
        assert len(sk_pipe.steps) == 3
