from __future__ import annotations

import importlib.util

import pytest

torch_available = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchvision"))


@pytest.mark.skipif(not torch_available, reason="torch/torchvision not installed")
class TestLoadMnist:
    def test_import_works(self):
        from datapalette.datasets import load_mnist

        assert callable(load_mnist)

    def test_import_fashion_works(self):
        from datapalette.datasets import load_fashion_mnist

        assert callable(load_fashion_mnist)


class TestLoadMnistWithoutTorch:
    def test_raises_import_error_without_torch(self):
        """If torch is missing, load_mnist should raise ImportError."""
        if torch_available:
            pytest.skip("torch is installed; cannot test missing-torch path")
        from datapalette.datasets import load_mnist

        with pytest.raises(ImportError, match="torchvision"):
            load_mnist()
