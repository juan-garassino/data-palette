"""Prepare image/mask pairs for a segmentation model; masks keep their integer labels.

uv run python examples/segmentation_pairs.py images/ masks/ out/
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from datapalette import SegmentationPipeline


def main(images: str, masks: str, dst: str) -> None:
    X, y = SegmentationPipeline(dataset=None, size=(256, 256)).load_and_transform(
        images_dir=images, masks_dir=masks
    )
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "masks"), exist_ok=True)
    for i, (image, mask) in enumerate(zip(X, y)):
        cv2.imwrite(os.path.join(dst, "images", f"{i:06d}.png"), image)
        cv2.imwrite(os.path.join(dst, "masks", f"{i:06d}.png"), mask)
    labels = np.unique(np.concatenate([m.ravel() for m in y])) if y else []
    print(f"wrote {len(X)} pairs to {dst}; mask labels: {list(labels)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
