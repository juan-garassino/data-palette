"""Command-line interface (``datapalette``) for batch image processing."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, List, Tuple

import cv2

from datapalette.io.batch import process_directory
from datapalette.io.video import extract_frames
from datapalette.pipelines import DiffusionPipeline, GANPipeline, SegmentationPipeline
from datapalette.transforms import (
    BrightnessContrast,
    ConvertColorSpace,
    EdgeChannels,
    Emboss,
    EnhanceGreen,
    FourierTransform,
    GaussianNoise,
    GradientChannels,
    Mirror,
    Multispectral,
    PCAColorAugmentation,
    RandomCrop,
    Rotate,
    SaltPepperNoise,
    Sharpen,
    Tile,
)
from datapalette.utils.config import load_config
from datapalette.utils.logging import setup_logging

__all__ = ["main"]

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _showcase_pipeline(
    input_image: str, output_dir: str, steps: List[Tuple[str, Any]]
) -> None:
    """Process a single image through all steps, saving each intermediate result."""
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image}")

    for i, (step_name, transform) in enumerate(steps):
        logger.info("Applying step %d/%d: %s", i + 1, len(steps), step_name)
        img = transform.transform(img)
        out_path = os.path.join(output_dir, f"{i + 1:02d}_{step_name}.png")
        cv2.imwrite(out_path, img)


def _get_pipeline_steps(args: argparse.Namespace) -> List[Tuple[str, Any]]:
    """Build transform steps from CLI arguments."""
    steps: List[Tuple[str, Any]] = []
    if args.rotate:
        steps.append(("rotation", Rotate(angle=args.rotate_angles[0])))
    if args.mirror:
        steps.append(("mirror", Mirror()))
    if args.brightness_contrast:
        steps.append((
            "brightness_contrast",
            BrightnessContrast(
                brightness_range=tuple(args.brightness_range),
                contrast_range=tuple(args.contrast_range),
            ),
        ))
    if args.noise:
        if args.noise_type == "gaussian":
            steps.append(("noise", GaussianNoise(amount=args.noise_amount)))
        else:
            steps.append(("noise", SaltPepperNoise(amount=args.noise_amount)))
    if args.crop:
        steps.append(("crop", RandomCrop(crop_size=tuple(args.crop_size))))
    if args.pca:
        steps.append(("pca", PCAColorAugmentation(alpha_std=args.pca_alpha_std)))
    if args.hsv:
        steps.append(("hsv", ConvertColorSpace(target="hsv")))
    if args.lab:
        steps.append(("lab", ConvertColorSpace(target="lab")))
    if args.fourier:
        steps.append(("fourier", FourierTransform()))
    if args.gradient:
        steps.append(("gradient", GradientChannels()))
    if args.multispectral:
        steps.append(("multispectral", Multispectral()))
    if args.edge:
        steps.append(("edge", EdgeChannels()))
    if args.green:
        steps.append(("green", EnhanceGreen()))
    if args.custom_filter == "emboss":
        steps.append(("emboss", Emboss()))
    elif args.custom_filter == "sharpen":
        steps.append(("sharpen", Sharpen()))
    return steps


def _process_input(args: argparse.Namespace) -> str:
    """Resolve input path, extracting video frames if necessary."""
    if args.input_type == "video":
        if not os.path.isfile(args.input):
            raise ValueError("For video input, provide a path to a video file.")
        frames_dir = os.path.join(args.output_dir, "video_frames")
        logger.info("Extracting frames from video: %s", args.input)
        extract_frames(args.input, frames_dir, fps=args.fps)
        return frames_dir
    elif args.input_type == "images":
        if not os.path.isdir(args.input):
            raise ValueError("For image input, provide a directory path.")
        return str(args.input)
    elif args.input_type == "single_image":
        if not os.path.isfile(args.input):
            raise ValueError("For single image input, provide a file path.")
        return str(args.input)
    else:
        raise ValueError(f"Invalid input type: {args.input_type}")


def _execute_pipeline(args: argparse.Namespace, input_path: str) -> None:
    """Run the selected pipeline."""
    if args.pipeline == "gan":
        gan = GANPipeline(dataset=None, size=tuple(args.crop_size))
        if args.input_type == "video":
            gan.load_and_transform(video_path=input_path, fps=args.fps)
        else:
            gan.load_and_transform(images_dir=input_path)

    elif args.pipeline == "unet":
        seg = SegmentationPipeline(dataset=None, mode="binary")
        if os.path.isdir(input_path):
            # For CLI we don't have masks_dir — just transform images
            images = seg._load_images_from_dir(input_path)
            X = seg._transform_images(images)
            out_dir = os.path.join(args.output_dir, "segmentation")
            os.makedirs(out_dir, exist_ok=True)
            for i, img in enumerate(X):
                cv2.imwrite(os.path.join(out_dir, f"seg_{i:06d}.png"), img)

    elif args.pipeline == "diffusion":
        diff = DiffusionPipeline(dataset=None, size=tuple(args.crop_size))
        images = diff._load_images_from_dir(input_path)
        X = diff._transform_images(images)
        out_dir = os.path.join(args.output_dir, "diffusion")
        os.makedirs(out_dir, exist_ok=True)
        for i, img in enumerate(X):
            cv2.imwrite(os.path.join(out_dir, f"diff_{i:06d}.png"), img)

    elif args.pipeline in ("custom", "showcase"):
        steps = _get_pipeline_steps(args)
        if args.pipeline == "showcase":
            _showcase_pipeline(input_path, args.output_dir, steps)
        else:
            from sklearn.pipeline import Pipeline as SkPipeline

            pipe = SkPipeline(steps)
            process_directory(
                input_path,
                args.output_dir,
                lambda img: pipe.transform(img),
                desc="Custom pipeline",
            )
    else:
        raise ValueError(f"Invalid pipeline: {args.pipeline}")


def main() -> None:
    """CLI entry point for DataPalette."""
    parser = argparse.ArgumentParser(
        description="DataPalette: Image Dataset Preprocessing & Augmentation"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT_DIR, help="Input directory/file"
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--pipeline",
        choices=["gan", "unet", "diffusion", "custom", "showcase"],
        required=True,
    )
    parser.add_argument(
        "--input-type",
        choices=["video", "images", "single_image"],
        required=True,
    )
    parser.add_argument("--config", help="Path to YAML configuration file")

    # Video
    parser.add_argument("--fps", type=int, default=1)
    # Crop / GAN
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--num-crops", type=int, default=5)
    # Tile
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--tile-overlap", type=float, default=0.0)
    # Transforms
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-angles", nargs="+", type=float, default=[90, 180, 270])
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--brightness-contrast", action="store_true")
    parser.add_argument("--brightness-range", nargs=2, type=float, default=[0.5, 1.5])
    parser.add_argument("--contrast-range", nargs=2, type=float, default=[0.5, 1.5])
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--noise-type", choices=["gaussian", "salt_and_pepper"], default="gaussian")
    parser.add_argument("--noise-amount", type=float, default=0.05)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--pca-alpha-std", type=float, default=0.1)
    parser.add_argument("--hsv", action="store_true")
    parser.add_argument("--lab", action="store_true")
    parser.add_argument("--fourier", action="store_true")
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--multispectral", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--green", action="store_true")
    parser.add_argument("--custom-filter", choices=["emboss", "sharpen"])

    args = parser.parse_args()

    # Merge config file (with key normalization)
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "datapalette.log"))
    logger.info("Starting DataPalette with pipeline: %s", args.pipeline)

    try:
        input_path = _process_input(args)

        # Apply tiling if requested
        if args.tile and args.pipeline != "showcase":
            tile = Tile(tile_size=tuple(args.tile_size), overlap=args.tile_overlap)
            tiled_dir = os.path.join(args.output_dir, "tiled_input")
            process_directory(
                input_path, tiled_dir, lambda img: tile.transform(img)[0], desc="Tiling"
            )
            input_path = tiled_dir

        _execute_pipeline(args, input_path)
        logger.info("DataPalette execution completed successfully")
    except Exception:
        logger.exception("An error occurred during execution")
        raise


if __name__ == "__main__":
    main()
