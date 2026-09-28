from __future__ import annotations

import numpy as np
import pytest

from datapalette import Mirror, Resize, Rotate
from datapalette.transforms.base import Compose, ImageTransform


def test_get_and_set_params():
    t = Rotate(angle=15)
    assert t.get_params() == {"angle": 15}
    assert t.set_params(angle=30) is t and t.angle == 30
    with pytest.raises(ValueError):
        t.set_params(nope=1)
    assert repr(t) == "Rotate(angle=30)"


def test_transform_is_callable_and_handles_lists(sample_image):
    t = Resize(size=(16, 24))
    assert t(sample_image).shape == (16, 24, 3)
    out = t.transform([sample_image, sample_image])
    assert isinstance(out, list) and len(out) == 2


def test_compose_named_and_unnamed_steps(sample_image):
    pipe = Compose([Resize((32, 32)), ("flip", Mirror(mode="horizontal"))])
    assert list(pipe.named_steps) == ["resize_0", "flip"]
    assert isinstance(pipe["flip"], Mirror) and isinstance(pipe[0], Resize)
    assert pipe.transform(sample_image).shape == (32, 32, 3)
    assert pipe.fit_transform([sample_image])[0].shape == (32, 32, 3)


def test_compose_fit_propagates_through_steps():
    class Recorder(ImageTransform):
        def __init__(self):
            self.seen = None

        def fit(self, X, y=None):
            self.seen = [x.shape for x in X]
            return self

        def _apply(self, image):
            return image

    rec = Recorder()
    Compose([Resize((8, 8)), rec]).fit([np.zeros((20, 20, 3), np.uint8)])
    assert rec.seen == [(8, 8, 3)]


class TestScikitLearnInterop:
    """Regression: sklearn >= 1.8 raised NotFittedError on unfitted pipelines of our transforms."""

    def test_unfitted_sklearn_pipeline_transforms(self, sample_image):
        Pipeline = pytest.importorskip("sklearn.pipeline").Pipeline
        pipe = Pipeline([("resize", Resize((32, 32))), ("rotate", Rotate(angle=90))])
        assert pipe.transform(sample_image).shape == (32, 32, 3)

    def test_clone_and_grid_params(self):
        base = pytest.importorskip("sklearn.base")
        pipeline = pytest.importorskip("sklearn.pipeline")
        pipe = pipeline.Pipeline([("rotate", Rotate(angle=90))])
        pipe.set_params(rotate__angle=45)
        cloned = base.clone(pipe)
        assert cloned.named_steps["rotate"].angle == 45
