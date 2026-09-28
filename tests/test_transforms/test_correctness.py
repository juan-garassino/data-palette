"""Regression tests for the 1.0 correctness fixes (each failed on 0.1.0)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from datapalette import (
    BrightnessContrast,
    ConvertColorSpace,
    EdgeChannels,
    GaussianNoise,
    GradientChannels,
    Mirror,
    PCAColorAugmentation,
    RandomCrop,
    Resize,
    Rotate,
    SaltPepperNoise,
)


@pytest.mark.parametrize(
    "make",
    [
        lambda s: RandomCrop((20, 20), random_state=s),
        lambda s: GaussianNoise(0.1, random_state=s),
        lambda s: SaltPepperNoise(0.1, random_state=s),
        lambda s: BrightnessContrast(random_state=s),
        lambda s: PCAColorAugmentation(random_state=s),
    ],
)
def test_random_transforms_are_reproducible(make, sample_image):
    a, b, c = (
        make(0).transform(sample_image),
        make(0).transform(sample_image),
        make(1).transform(sample_image),
    )
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_brightness_is_a_fraction_of_the_range():
    gray = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = BrightnessContrast(brightness_range=(0.2, 0.2), contrast_range=(1.0, 1.0)).transform(gray)
    assert int(out[0, 0, 0]) == 151  # 100 + 0.2 * 255
    darker = BrightnessContrast(brightness_range=(-1, -1), contrast_range=(1, 1)).transform(gray)
    assert darker.max() == 0


def test_default_brightness_contrast_actually_changes_images():
    gray = np.full((10, 10, 3), 128, dtype=np.uint8)
    t = BrightnessContrast(random_state=3)
    assert any(not np.array_equal(t.transform(gray), gray) for _ in range(5))


def test_pca_without_fit_does_not_mutate_and_fit_learns(sample_image):
    t = PCAColorAugmentation(random_state=0)
    t.transform(sample_image)
    assert not hasattr(t, "eigvals_")
    assert t.get_params() == {"alpha_std": 0.1, "random_state": 0}
    t.fit([sample_image, sample_image])
    assert t.eigvals_.shape == (3,)


def test_pca_rejects_non_rgb():
    with pytest.raises(ValueError, match="3 channel"):
        PCAColorAugmentation().transform(np.zeros((8, 8), dtype=np.uint8))


@pytest.mark.parametrize("t", [Resize((16, 12)), Rotate(30), Mirror(), RandomCrop((5, 5), random_state=0)])
def test_single_channel_images_keep_their_shape(t):
    two_d = np.random.randint(0, 255, (20, 24), dtype=np.uint8)
    one_ch = two_d[:, :, None]
    assert t.transform(two_d).ndim == 2
    out = t.transform(one_ch)
    assert out.ndim == 3 and out.shape[2] == 1


def test_rotate_expand_swaps_sides_without_cropping():
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    out = Rotate(90, expand=True).transform(img)
    assert out.shape[:2] == (40, 20)
    assert out.mean() > 250  # nothing cropped, nothing padded


def test_nearest_resize_keeps_label_values():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, 20:] = 2
    mask[10:30, 10:30] = 7
    out = Resize((23, 17), interpolation="nearest").transform(mask)
    assert set(np.unique(out)) <= {0, 2, 7}
    assert set(np.unique(Rotate(33, interpolation="nearest").transform(mask))) <= {0, 2, 7}


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError):
        Resize((4, 4), interpolation="bogus").transform(np.zeros((8, 8), np.uint8))


def test_channel_checks_give_clear_errors():
    two_channels = np.zeros((8, 8, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="EdgeChannels expects"):
        EdgeChannels().transform(two_channels)
    with pytest.raises(ValueError, match="ConvertColorSpace expects"):
        ConvertColorSpace("hsv").transform(np.zeros((8, 8), np.uint8))


def test_gradient_then_edge_chain_no_longer_crashes_on_gray_input():
    gray = cv2.cvtColor(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    assert GradientChannels().transform(gray).shape == (16, 16, 2)
    assert EdgeChannels().transform(gray[:, :, None]).shape == (16, 16, 2)
