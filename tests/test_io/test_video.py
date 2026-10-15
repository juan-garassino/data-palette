from __future__ import annotations

import os

import cv2
import numpy as np

from datapalette.io import extract_frames, n_croppings_from_frame


def _create_synthetic_video(path: str, num_frames: int = 10, fps: int = 10) -> None:
    """Create a tiny synthetic AVI video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, fps, (32, 32))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestExtractFrames:
    def test_extracts_frames(self, tmp_path):
        video_path = str(tmp_path / "test_video.avi")
        out_dir = str(tmp_path / "frames")
        _create_synthetic_video(video_path, num_frames=10, fps=10)

        count = extract_frames(video_path, out_dir, fps=10)
        assert count > 0
        frame_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(frame_files) == count

    def test_extract_with_resize(self, tmp_path):
        video_path = str(tmp_path / "test_video.avi")
        out_dir = str(tmp_path / "frames_resized")
        _create_synthetic_video(video_path, num_frames=5, fps=5)

        count = extract_frames(video_path, out_dir, fps=5, output_size=(16, 16))
        assert count > 0
        # Check a frame was resized
        sample = cv2.imread(os.path.join(out_dir, "frame_000000.png"))
        assert sample is not None
        assert sample.shape[:2] == (16, 16)


class TestNCroppingsFromFrame:
    def test_generates_crops(self, tmp_path):
        # Create a source image
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame_path = str(tmp_path / "source.png")
        cv2.imwrite(frame_path, img)
        out_dir = str(tmp_path / "crops")

        paths = n_croppings_from_frame(frame_path, out_dir, num_crops=5, crop_size=(16, 16))
        assert len(paths) == 5
        for p in paths:
            assert os.path.isfile(p)
            crop = cv2.imread(p)
            assert crop.shape == (16, 16, 3)
