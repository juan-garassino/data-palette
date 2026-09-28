from __future__ import annotations

import cv2
import numpy as np

from datapalette import SegmentationPipeline


def test_masks_keep_integer_labels_through_the_pipeline(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(images / f"{i}.png"), rng.integers(0, 255, (50, 70, 3), dtype=np.uint8))
        m = np.zeros((50, 70), dtype=np.uint8)
        m[10:40, 5:35] = 1
        m[20:45, 30:65] = 2
        cv2.imwrite(str(masks / f"{i}.png"), m)
    X, y = SegmentationPipeline(dataset=None, size=(33, 33)).load_and_transform(
        images_dir=str(images), masks_dir=str(masks)
    )
    assert len(X) == len(y) == 3
    for mask in y:
        assert mask.shape == (33, 33)
        assert set(np.unique(mask)) <= {0, 1, 2}


def test_instance_mode_is_transformed_too(monkeypatch):
    pipe = SegmentationPipeline(dataset="mnist", mode="instance", size=(40, 40))
    digits = np.zeros((6, 28, 28, 1), dtype=np.uint8)
    digits[:, 8:20, 8:20, 0] = 255
    monkeypatch.setattr(pipe, "_load_dataset", lambda: (digits, np.arange(6)))
    X, y = pipe.load_and_transform()
    assert all(img.shape == (40, 40, 3) for img in X)
    labels = set(np.unique(np.concatenate([m.ravel() for m in y])))
    assert all(float(v).is_integer() for v in labels)
