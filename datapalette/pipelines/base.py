from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from datapalette.transforms.base import ImageTransform

__all__ = ["TaskPipeline"]


class TaskPipeline:
    """Base class for task-aware pipelines that return ``(X, y)``.

    Parameters
    ----------
    steps : list[tuple[str, ImageTransform]]
        Named transform steps, passed directly to ``sklearn.pipeline.Pipeline``.
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
        self.pipeline = Pipeline(self.steps) if self.steps else None

    def load_and_transform(
        self,
        images_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], Any]:
        """Load images, apply transforms, return ``(X, y)``.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def sklearn_pipeline(self) -> Pipeline:
        """Access the underlying sklearn Pipeline."""
        if self.pipeline is None:
            raise ValueError("No pipeline steps configured.")
        return self.pipeline

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

    def _load_images_from_dir(self, images_dir: str) -> List[np.ndarray]:
        """Load all images from a directory."""
        import os

        from datapalette.io.batch import IMAGE_EXTENSIONS, load_image

        files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
        return [load_image(os.path.join(images_dir, f)) for f in files]

    def _transform_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the sklearn pipeline to a list of images."""
        if self.pipeline is None:
            return images
        return self.pipeline.transform(images)
