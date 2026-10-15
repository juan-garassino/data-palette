from __future__ import annotations

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
