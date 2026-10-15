"""Task-aware pipelines that return ``(X, y)`` pairs."""

from __future__ import annotations

from datapalette.pipelines.base import TaskPipeline
from datapalette.pipelines.diffusion import DiffusionPipeline
from datapalette.pipelines.gan import GANPipeline
from datapalette.pipelines.segmentation import SegmentationPipeline

__all__ = [
    "TaskPipeline",
    "GANPipeline",
    "SegmentationPipeline",
    "DiffusionPipeline",
]
