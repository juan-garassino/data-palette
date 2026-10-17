from __future__ import annotations

import cv2
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from datapalette.transforms import Mirror, RandomCrop, Resize, Rotate, Tile


class TestRotate:
    def test_output_shape_and_dtype(self, sample_image):
        t = Rotate(angle=45.0)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_get_set_params(self):
        t = Rotate(angle=90.0)
        params = t.get_params()
        assert params["angle"] == 90.0
        t.set_params(angle=180.0)
        assert t.angle == 180.0


class TestMirror:
    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "both"])
    def test_output_shape_and_dtype(self, sample_image, mode):
        t = Mirror(mode=mode)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_invalid_mode_raises(self, sample_image):
        t = Mirror(mode="invalid")
        with pytest.raises(ValueError, match="Invalid mode"):
            t.transform(sample_image)


class TestRandomCrop:
    def test_output_shape(self, sample_image):
        t = RandomCrop(crop_size=(50, 50))
        result = t.transform(sample_image)
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_image_smaller_than_crop_resizes_up(self, small_image):
        """Regression: when the image is smaller than crop_size, it should
        be resized up and then cropped without error."""
        t = RandomCrop(crop_size=(32, 32))
        result = t.transform(small_image)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8


class TestTile:
    def test_returns_list(self, sample_image):
        t = Tile(tile_size=(50, 50), overlap=0.0)
        tiles = t.transform(sample_image)
        assert isinstance(tiles, list)
        assert len(tiles) > 0
        for tile in tiles:
            assert tile.shape == (50, 50, 3)

    def test_tile_count(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        t = Tile(tile_size=(50, 50), overlap=0.0)
        tiles = t.transform(img)
        # 100/50 = 2 steps in each dimension -> 4 tiles
        assert len(tiles) == 4

    def test_overlap(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        t = Tile(tile_size=(50, 50), overlap=0.5)
        tiles = t.transform(img)
        # stride = 25, so (100-50)//25 + 1 = 3 steps each dim -> 9 tiles
        assert len(tiles) > 4


class TestResize:
    def test_output_shape(self, sample_image):
        t = Resize(size=(64, 48))
        result = t.transform(sample_image)
        assert result.shape == (64, 48, 3)
        assert result.dtype == np.uint8


class TestSklearnPipelineComposition:
    def test_pipeline_compose(self, sample_image):
        pipe = Pipeline([
            ("resize", Resize(size=(64, 64))),
            ("rotate", Rotate(angle=90)),
            ("mirror", Mirror(mode="horizontal")),
        ])
        result = pipe.transform(sample_image)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_stateless_transforms_report_fitted(self):
        # sklearn>=1.6 Pipeline.transform calls check_is_fitted on the last step;
        # stateless transforms must not raise NotFittedError (G-001).
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(Resize(size=(8, 8)))
        check_is_fitted(Pipeline([("resize", Resize(size=(8, 8)))]))


class TestGrayscaleSingleImage:
    """A single 2-D (grayscale) image must be treated as one image, not a list of rows."""

    def test_resize_grayscale(self):
        gray = np.random.randint(0, 255, (40, 30), dtype=np.uint8)
        result = Resize(size=(20, 10)).transform(gray)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 10)

    def test_mirror_grayscale(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = Mirror(mode="horizontal").transform(gray)
        np.testing.assert_array_equal(result, gray[:, ::-1])

    def test_tile_grayscale(self):
        gray = np.zeros((8, 8), dtype=np.uint8)
        tiles = Tile(tile_size=(4, 4)).transform(gray)
        assert len(tiles) == 4
        assert all(t.shape == (4, 4) for t in tiles)


class TestInterpolation:
    def test_resize_nearest_keeps_labels(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:, 5:] = 200
        out = Resize(size=(7, 7), interpolation=cv2.INTER_NEAREST).transform(mask)
        assert set(np.unique(out).tolist()) <= {0, 200}

    def test_rotate_nearest_keeps_labels(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 9
        out = Rotate(angle=33.0, interpolation=cv2.INTER_NEAREST).transform(mask)
        assert set(np.unique(out).tolist()) <= {0, 9}


class TestRotateCenter:
    """Rotation must pivot on the pixel-grid centre ((w-1)/2, (h-1)/2).

    Pivoting on (w/2, h/2) shifted a 90-degree rotation by one pixel, dropping a
    row and filling it with 0 (a valid class id in multiclass masks).
    """

    @pytest.mark.parametrize("interp", [cv2.INTER_NEAREST, cv2.INTER_LINEAR])
    def test_square_90_matches_rot90(self, interp):
        img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        out = Rotate(angle=90.0, interpolation=interp).transform(img)
        np.testing.assert_array_equal(out, np.rot90(img))

    def test_square_180_matches_flip(self):
        img = np.random.randint(0, 255, (9, 9, 3), dtype=np.uint8)
        out = Rotate(angle=180.0).transform(img)
        np.testing.assert_array_equal(out, img[::-1, ::-1])

    def test_border_value_fills_uncovered_pixels(self):
        mask = np.full((10, 10), 3, dtype=np.uint8)
        out = Rotate(angle=45.0, interpolation=cv2.INTER_NEAREST, border_value=255).transform(mask)
        assert out[0, 0] == 255
        assert set(np.unique(out).tolist()) == {3, 255}


class TestTileValidation:
    @pytest.mark.parametrize("overlap", [-0.5, 1.0, 1.5])
    def test_invalid_overlap_raises(self, overlap):
        with pytest.raises(ValueError, match="overlap"):
            Tile(tile_size=(4, 4), overlap=overlap).transform(np.zeros((8, 8), np.uint8))

    def test_non_positive_tile_size_raises(self):
        with pytest.raises(ValueError, match="tile_size"):
            Tile(tile_size=(0, 4)).transform(np.zeros((8, 8), np.uint8))
