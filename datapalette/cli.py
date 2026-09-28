"""datapalette command line: apply transform chains, run task pipelines, list transforms."""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import cv2

from datapalette import __version__
from datapalette.transforms import REGISTRY, Compose, ImageTransform

__all__ = ["main", "parse_step", "build_parser"]

logger = logging.getLogger(__name__)


def parse_step(spec: str) -> ImageTransform:
    """``"Resize(size=(64, 64))"`` or ``"Mirror"`` -> a transform instance.

    Parsed with :mod:`ast` and ``literal_eval`` — keyword arguments must be literals,
    nothing is ever evaluated.
    """
    try:
        node = ast.parse(spec.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"cannot parse step {spec!r}") from e
    if isinstance(node, ast.Name):
        name, kwargs = node.id, {}
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
        name = node.func.id
        try:
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except ValueError as e:
            raise ValueError(f"step {spec!r}: arguments must be literals") from e
    else:
        raise ValueError(f"step {spec!r}: expected Name or Name(key=value, ...)")
    return build_step(name, kwargs)


def build_step(name: str, params: Dict[str, Any]) -> ImageTransform:
    if name not in REGISTRY:
        raise ValueError(f"unknown transform {name!r}; see `datapalette list-transforms`")
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return REGISTRY[name](**params)


def steps_from_config(config: Dict[str, Any]) -> List[ImageTransform]:
    """``[[steps]]`` tables: ``transform = "Resize"`` plus that transform's parameters."""
    steps = []
    for table in config.get("steps", []):
        table = dict(table)
        steps.append(build_step(table.pop("transform"), table))
    return steps


def _resolve_steps(args: argparse.Namespace) -> List[ImageTransform]:
    """CLI --step flags win over the config file."""
    if args.step:
        return [parse_step(s) for s in args.step]
    if args.config:
        from datapalette.utils.config import load_config

        return steps_from_config(load_config(args.config))
    raise SystemExit("no steps: pass --step (repeatable) or --config with [[steps]] tables")


def _cmd_list(args: argparse.Namespace) -> int:
    for name, cls in REGISTRY.items():
        params = ", ".join(f"{k}={v!r}" for k, v in cls().get_params().items()) if name != "Compose" else ""
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name}({params})\n    {doc}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    from datapalette.io.batch import process_directory, write_outputs

    pipe = Compose(_resolve_steps(args))
    os.makedirs(args.output, exist_ok=True)
    if os.path.isdir(args.input):
        paths = process_directory(args.input, args.output, pipe.transform, output_format=args.format)
    else:
        image = cv2.imread(args.input)
        if image is None:
            raise SystemExit(f"cannot read image: {args.input}")
        stem = os.path.splitext(os.path.basename(args.input))[0]
        paths = write_outputs(pipe.transform(image), args.output, stem, args.format)
    logger.info("wrote %d image(s) to %s", len(paths), args.output)
    return 0


def _cmd_showcase(args: argparse.Namespace) -> int:
    image: Any = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    os.makedirs(args.output, exist_ok=True)
    for i, step in enumerate(_resolve_steps(args), start=1):
        image = step.transform(image)
        path = os.path.join(args.output, f"{i:02d}_{type(step).__name__.lower()}.png")
        cv2.imwrite(path, image)
        logger.info("%s -> %s", step, path)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from datapalette.io.batch import write_outputs
    from datapalette.pipelines import DiffusionPipeline, GANPipeline, SegmentationPipeline

    size = tuple(args.size)
    masks = None
    if args.pipeline == "gan":
        images, _ = GANPipeline(dataset=None, size=size).load_and_transform(
            images_dir=args.images, video_path=args.video, fps=args.fps
        )
    elif args.pipeline == "diffusion":
        images, _ = DiffusionPipeline(dataset=None, size=size).load_and_transform(images_dir=args.images)
    else:
        if not args.masks:
            raise SystemExit("the segmentation pipeline needs --masks")
        images, masks = SegmentationPipeline(dataset=None, size=size).load_and_transform(
            images_dir=args.images, masks_dir=args.masks
        )
    for i, image in enumerate(images):
        write_outputs(image, os.path.join(args.output, "images"), f"{i:06d}")
    for i, mask in enumerate(masks or []):
        write_outputs(mask, os.path.join(args.output, "masks"), f"{i:06d}")
    logger.info("wrote %d image(s) to %s", len(images), args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datapalette", description="DataPalette: image dataset preprocessing & augmentation"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-transforms", help="list available transforms and their parameters")
    p.set_defaults(func=_cmd_list)

    def add_steps(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--step", action="append", default=[], help='e.g. "Resize(size=(64, 64))" (repeatable)'
        )
        p.add_argument("--config", help="TOML file with [[steps]] tables (used when no --step is given)")

    p = sub.add_parser("apply", help="apply a transform chain to an image or a directory")
    p.add_argument("input", help="image file or directory")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("--format", default="png", help="output extension (default png)")
    add_steps(p)
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser("showcase", help="save every intermediate step for one image")
    p.add_argument("image")
    p.add_argument("-o", "--output", required=True)
    add_steps(p)
    p.set_defaults(func=_cmd_showcase)

    p = sub.add_parser("run", help="run a task pipeline (gan, segmentation, diffusion)")
    p.add_argument("pipeline", choices=["gan", "segmentation", "diffusion"])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="directory of images")
    src.add_argument("--video", help="video file (gan only)")
    p.add_argument("--masks", help="directory of masks, same filenames (segmentation)")
    p.add_argument("--fps", type=int, default=1, help="frames per second to extract from --video")
    p.add_argument("--size", nargs=2, type=int, default=[64, 64], metavar=("H", "W"))
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.cmd == "run" and args.video and args.pipeline != "gan":
        raise SystemExit("--video is only supported by the gan pipeline")
    try:
        return args.func(args)
    except ValueError as e:
        logger.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
