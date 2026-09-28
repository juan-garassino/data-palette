from __future__ import annotations

import cv2
import numpy as np

from datapalette.transforms.base import ImageTransform, minmax_to_uint8

__all__ = ["FourierTransform"]


class FourierTransform(ImageTransform):
    """Compute the Fourier magnitude spectrum per channel.

    Returns a single-channel (grayscale) or multi-channel magnitude image of
    the same spatial size as the input.
    """

    @staticmethod
    def _process_channel(channel: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        return minmax_to_uint8(magnitude)

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or image.shape[2] == 1:
            ch = image[:, :, 0] if image.ndim == 3 else image
            result = self._process_channel(ch)
            return result[:, :, np.newaxis] if image.ndim == 3 else result

        channels = cv2.split(image)
        processed = [self._process_channel(c) for c in channels]
        return cv2.merge(processed)
