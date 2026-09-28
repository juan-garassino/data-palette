from __future__ import annotations

import numpy as np
import pytest

from datapalette.transforms import (
    ConvertColorSpace,
    EnhanceGreen,
    Multispectral,
    PCAColorAugmentation,
)


class TestPCAColorAugmentation:
    def test_fit_transform_single_image(self, sample_image):
        t = PCAColorAugmentation(alpha_std=0.1)
        t.fit(sample_image)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_fit_transform_list(self, sample_image):
        images = [sample_image, sample_image.copy()]
        t = PCAColorAugmentation(alpha_std=0.1)
        t.fit(images)
        results = t.transform(images)
        assert len(results) == 2
        for r in results:
            assert r.shape == sample_image.shape

    def test_transform_without_fit_uses_per_image_stats(self, sample_image):
        """Without fit, each image uses its own colour axes and nothing is stored."""
        t = PCAColorAugmentation(alpha_std=0.1)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert not hasattr(t, "eigvals_")


class TestConvertColorSpace:
    @pytest.mark.parametrize("target", ["hsv", "lab"])
    def test_3_channel_output(self, sample_image, target):
        t = ConvertColorSpace(target=target)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_gray_output(self, sample_image):
        t = ConvertColorSpace(target="gray")
        result = t.transform(sample_image)
        # Gray produces (H, W, 1) due to the newaxis in _apply
        assert result.shape == (100, 100, 1)
        assert result.dtype == np.uint8

    def test_invalid_target_raises(self, sample_image):
        t = ConvertColorSpace(target="xyz")
        with pytest.raises(ValueError, match="Unknown target"):
            t.transform(sample_image)


class TestEnhanceGreen:
    def test_output_shape_dtype(self, sample_image):
        t = EnhanceGreen()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestMultispectral:
    def test_outputs_7_channels(self, sample_image):
        t = Multispectral()
        result = t.transform(sample_image)
        assert result.shape == (100, 100, 7)
        assert result.dtype == np.uint8
