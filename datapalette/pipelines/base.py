from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from datapalette.transforms.base import Compose, ImageTransform

__all__ = ["TaskPipeline"]


class TaskPipeline:
    """Base class for task-aware pipelines that return ``(X, y)``.

    Parameters
    ----------
    steps : list[tuple[str, ImageTransform]]
        Named transform steps, chained with :class:`~datapalette.transforms.base.Compose`.
    dataset : str or None
        Name of a built-in dataset (e.g. ``'mnist'``, ``'fashion_mnist'``).
    """

    def __init__(
        self,
        steps: Optional[List[Tuple[str, ImageTransform]]] = None,
        dataset: Optional[str] = None,
    ):
        self.steps = steps or []
        self.dataset = dataset
        self.transforms = Compose(self.steps)

    def load_and_transform(
        self,
        images_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], Any]:
        """Load images, apply transforms, return ``(X, y)``.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the configured built-in dataset."""
        if self.dataset is None:
            raise ValueError("No dataset configured and no images_dir provided.")

        from datapalette.datasets.mnist import load_fashion_mnist, load_mnist

        loaders = {
            "mnist": load_mnist,
            "fashion_mnist": load_fashion_mnist,
        }
        loader = loaders.get(self.dataset)
        if loader is None:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Choose from {list(loaders)}")
        return loader(train=True)

    def load_images(self, images_dir: str) -> List[np.ndarray]:
        """Load all images from a directory."""
        import os

        from datapalette.io.batch import IMAGE_EXTENSIONS, load_image

        files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
        return [load_image(os.path.join(images_dir, f)) for f in files]

    def transform_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the configured transforms to a list of images."""
        return self.transforms.transform(images)  # type: ignore[return-value]
