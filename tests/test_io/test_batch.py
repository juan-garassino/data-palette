from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from datapalette.io import load_image, process_directory, save_image
from datapalette.io.batch import write_outputs
from datapalette.transforms import GradientChannels, Multispectral


class TestLoadImage:
    def test_loads_valid_image(self, tmp_path):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        loaded = load_image(path)
        assert loaded.shape == (32, 32, 3)
        assert loaded.dtype == np.uint8

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")


class TestSaveImage:
    def test_saves_and_creates_dirs(self, tmp_path):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / "subdir" / "out.png")
        save_image(path, img)
        assert os.path.isfile(path)
        loaded = cv2.imread(path)
        assert loaded is not None
        assert loaded.shape == (32, 32, 3)


class TestProcessDirectory:
    def test_processes_all_images(self, tmp_path):
        in_dir = str(tmp_path / "input")
        out_dir = str(tmp_path / "output")
        os.makedirs(in_dir)

        for i in range(3):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(in_dir, f"img_{i}.png"), img)

        def flip(img):
            return cv2.flip(img, 1)

        saved = process_directory(in_dir, out_dir, flip)
        assert len(saved) == 3
        for path in saved:
            assert os.path.isfile(path)

    def test_empty_directory(self, tmp_path):
        in_dir = str(tmp_path / "empty_in")
        out_dir = str(tmp_path / "empty_out")
        os.makedirs(in_dir)
        saved = process_directory(in_dir, out_dir, lambda img: img)
        assert len(saved) == 0


class TestWriteOutputs:
    def test_three_channel_image_is_written_in_requested_format(self, tmp_path):
        img = np.random.randint(0, 255, (16, 20, 3), dtype=np.uint8)
        (path,) = write_outputs(img, str(tmp_path), "rgb", "jpg")
        assert path.endswith("rgb.jpg") and cv2.imread(path).shape == (16, 20, 3)

    @pytest.mark.parametrize("transform", [Multispectral(), GradientChannels()], ids=["7ch", "2ch"])
    def test_non_image_channel_counts_are_saved_as_npy(self, tmp_path, transform):
        # PNG/JPG hold only 1, 3 or 4 channels; cv2.imwrite used to raise on these outputs.
        out = transform.transform(np.random.randint(0, 255, (16, 20, 3), dtype=np.uint8))
        (path,) = write_outputs(out, str(tmp_path), "multi", "png")
        assert path.endswith("multi.npy")
        np.testing.assert_array_equal(np.load(path), out)

    def test_list_of_multichannel_outputs_is_numbered(self, tmp_path):
        out = Multispectral().transform(np.zeros((8, 8, 3), dtype=np.uint8))
        paths = write_outputs([out, out], str(tmp_path), "ms")
        assert [os.path.basename(p) for p in paths] == ["ms_000.npy", "ms_001.npy"]
