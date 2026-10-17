from __future__ import annotations

import numpy as np

from datapalette.pipelines import GANPipeline


class TestGANPipeline:
    def test_load_and_transform_from_images_dir(self, tmp_image_dir):
        pipe = GANPipeline(dataset=None, size=(32, 32))
        X, y = pipe.load_and_transform(images_dir=tmp_image_dir)
        assert y is None
        assert isinstance(X, list)
        assert len(X) == 3
        for img in X:
            assert img.shape == (32, 32, 3)
            assert img.dtype == np.uint8

    def test_sklearn_pipeline_accessible(self):
        pipe = GANPipeline(dataset=None, size=(64, 64))
        sk_pipe = pipe.sklearn_pipeline
        assert sk_pipe is not None
        assert len(sk_pipe.steps) == 4


def test_video_input_cleans_up_temp_frames(tmp_path, monkeypatch):
    """Frames extracted from a video must not leak a temp directory per call."""
    import tempfile

    import cv2
    import numpy as np

    from datapalette.pipelines import GANPipeline

    video = str(tmp_path / "v.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(5):
        writer.write(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    writer.release()

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    X, _ = GANPipeline(dataset=None, size=(16, 16)).load_and_transform(video_path=video, fps=10)
    assert len(X) > 0
    assert list(scratch.iterdir()) == []
