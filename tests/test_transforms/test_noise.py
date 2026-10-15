from __future__ import annotations

import numpy as np

from datapalette.transforms import BrightnessContrast, GaussianNoise, SaltPepperNoise


class TestGaussianNoise:
    def test_output_shape_dtype(self, sample_image):
        t = GaussianNoise(amount=0.05)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self, sample_image):
        t = GaussianNoise(amount=0.1)
        result = t.transform(sample_image)
        # Very unlikely to be identical with noise added
        assert not np.array_equal(result, sample_image)


class TestSaltPepperNoise:
    def test_output_shape_dtype(self, sample_image):
        t = SaltPepperNoise(amount=0.05)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_no_error_on_3_channel_image(self, sample_image):
        """Regression: SaltPepperNoise uses 2D coords (rows, cols) which
        sets all channels at once. Must not raise on 3-channel images."""
        t = SaltPepperNoise(amount=0.1, salt_ratio=0.5)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape

    def test_salt_pixels_present(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        t = SaltPepperNoise(amount=0.5, salt_ratio=1.0)
        result = t.transform(img)
        # With salt_ratio=1.0 and amount=0.5, many pixels should be 255
        assert np.any(result == 255)


class TestBrightnessContrast:
    def test_output_shape_dtype(self, sample_image):
        t = BrightnessContrast()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_range(self, sample_image):
        """Output should be clipped to [0, 255] by cv2.convertScaleAbs."""
        t = BrightnessContrast(brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5))
        result = t.transform(sample_image)
        assert result.min() >= 0
        assert result.max() <= 255
