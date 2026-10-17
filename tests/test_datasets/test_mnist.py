from __future__ import annotations

import pytest

torch_available = True
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:
    torch_available = False


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
