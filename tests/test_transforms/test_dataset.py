from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from datapalette.transforms import (
    AspectResize,
    DominantColor,
    ManySquares,
    SquareCrop,
    SquarePad,
)


@pytest.fixture
def wide_image() -> np.ndarray:
    """200x100 (HxW) landscape BGR uint8 image."""
    return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def tall_image() -> np.ndarray:
    """200x100 (HxW) portrait BGR uint8 image."""
    return np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)


class TestAspectResize:
    def test_longest_side_and_ratio_preserved(self, wide_image):
        t = AspectResize(max_size=64)
        result = t.transform(wide_image)
        h, w = result.shape[:2]
        assert max(h, w) == 64  # longest side hits the target
        assert w == 64 and h == 32  # 200x100 landscape -> 64x32, aspect kept
        assert result.dtype == np.uint8

    def test_upscale(self, tall_image):
        t = AspectResize(max_size=400)
        result = t.transform(tall_image)
        assert max(result.shape[:2]) == 400


class TestSquareCrop:
    def test_wide_becomes_square(self, wide_image):
        result = SquareCrop().transform(wide_image)
        h, w = result.shape[:2]
        assert h == w == 100  # crop to the shorter side, no rescale

    def test_tall_becomes_square(self, tall_image):
        result = SquareCrop().transform(tall_image)
        h, w = result.shape[:2]
        assert h == w == 100

    def test_alignment_left_vs_right_differ(self):
        img = np.tile(np.arange(200, dtype=np.uint8)[None, :, None], (100, 1, 3))
        left = SquareCrop(h_align="left").transform(img)
        right = SquareCrop(h_align="right").transform(img)
        assert not np.array_equal(left, right)  # alignment actually shifts the window

    def test_invalid_align_raises(self, wide_image):
        with pytest.raises(ValueError, match="Invalid h_align"):
            SquareCrop(h_align="nope").transform(wide_image)


class TestSquarePad:
    def test_wide_padded_to_square(self, wide_image):
        result = SquarePad().transform(wide_image)
        h, w = result.shape[:2]
        assert h == w == 200  # pad the shorter side up to the longer

    def test_tall_padded_to_square(self, tall_image):
        result = SquarePad().transform(tall_image)
        h, w = result.shape[:2]
        assert h == w == 200

    def test_constant_fill_color(self):
        img = np.full((10, 20, 3), 200, dtype=np.uint8)  # wide -> pads top/bottom
        result = SquarePad(border_type="constant", fill=(0, 0, 0)).transform(img)
        assert result.shape == (20, 20, 3)
        assert result[0, 0].tolist() == [0, 0, 0]  # padded region is the fill color
        assert result[10, 10].tolist() == [200, 200, 200]  # original content intact

    def test_invalid_border_raises(self, wide_image):
        with pytest.raises(ValueError, match="Invalid border_type"):
            SquarePad(border_type="nope").transform(wide_image)


class TestManySquares:
    def test_tall_yields_two_square_crops(self, tall_image):
        crops = ManySquares().transform(tall_image)
        assert isinstance(crops, list)
        assert len(crops) == 2
        for c in crops:
            assert c.shape[0] == c.shape[1] == 100  # each is square, side = short dim

    def test_wide_yields_two_square_crops(self, wide_image):
        crops = ManySquares().transform(wide_image)
        assert len(crops) == 2
        for c in crops:
            assert c.shape[0] == c.shape[1] == 100

    def test_near_square_yields_single_crop(self):
        img = np.random.randint(0, 255, (100, 105, 3), dtype=np.uint8)
        crops = ManySquares().transform(img)
        assert len(crops) == 1
        assert crops[0].shape[0] == crops[0].shape[1]


class TestDominantColor:
    def test_pure_red_labeled_red(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # BGR red
        assert DominantColor().transform(img) == "red"

    def test_pure_blue_labeled_blue(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # BGR blue
        assert DominantColor().transform(img) == "blue"

    def test_list_input_returns_list(self):
        red = np.zeros((40, 40, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        black = np.zeros((40, 40, 3), dtype=np.uint8)
        labels = DominantColor().transform([red, black])
        assert labels == ["red", "black"]


class TestPipelineComposition:
    def test_resize_then_crop_produces_fixed_square(self, wide_image):
        pipe = Pipeline([
            ("resize", AspectResize(max_size=128)),
            ("crop", SquareCrop()),
        ])
        result = pipe.transform(wide_image)
        h, w = result.shape[:2]
        assert h == w  # square after crop
        assert result.dtype == np.uint8
