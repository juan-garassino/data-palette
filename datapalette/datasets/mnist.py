"""MNIST and Fashion-MNIST loaders returning NumPy arrays (requires torchvision)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["load_mnist", "load_fashion_mnist"]


def _torch_dataset_to_numpy(dataset: object) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision dataset to numpy arrays.

    Returns ``(images, labels)`` where images is ``(N, 28, 28, 1)`` uint8
    and labels is ``(N,)`` int64.
    """
    images = dataset.data.numpy()  # type: ignore[attr-defined]
    labels = dataset.targets.numpy()  # type: ignore[attr-defined]
    # Ensure (N, H, W, 1)
    if images.ndim == 3:
        images = images[:, :, :, np.newaxis]
    return images.astype(np.uint8), labels.astype(np.int64)


def load_mnist(train: bool = True, root: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST as numpy arrays via torchvision.

    Requires the ``[torch]`` optional dependency.

    Parameters
    ----------
    train : bool
        Load training set (True) or test set (False). Default True.
    root : str
        Download directory. Default ``'./data'``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(images, labels)`` — images shape ``(N, 28, 28, 1)`` uint8,
        labels shape ``(N,)`` int64.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:
        raise ImportError(
            "torchvision is required for built-in datasets. "
            "Install with: pip install datapalette[torch]"
        ) from e

    ds = MNIST(root=root, train=train, download=True)
    return _torch_dataset_to_numpy(ds)


def load_fashion_mnist(train: bool = True, root: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST as numpy arrays via torchvision.

    Requires the ``[torch]`` optional dependency.

    Parameters
    ----------
    train : bool
        Load training set (True) or test set (False). Default True.
    root : str
        Download directory. Default ``'./data'``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(images, labels)`` — images shape ``(N, 28, 28, 1)`` uint8,
        labels shape ``(N,)`` int64.
    """
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as e:
        raise ImportError(
            "torchvision is required for built-in datasets. "
            "Install with: pip install datapalette[torch]"
        ) from e

    ds = FashionMNIST(root=root, train=train, download=True)
    return _torch_dataset_to_numpy(ds)
