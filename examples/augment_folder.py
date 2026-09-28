"""Build a square, augmented training set from a folder of photos.

uv run python examples/augment_folder.py photos/ out/
"""

from __future__ import annotations

import sys

from datapalette import AspectResize, BrightnessContrast, Compose, Mirror, SquareCrop
from datapalette.io.batch import process_directory


def main(src: str, dst: str) -> None:
    pipe = Compose(
        [
            AspectResize(max_size=256),
            SquareCrop(),
            Mirror(mode="horizontal"),
            BrightnessContrast(random_state=0),
        ]
    )
    paths = process_directory(src, dst, pipe.transform)
    print(f"wrote {len(paths)} images to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
