from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from datapalette import Mirror, Resize
from datapalette.cli import main, parse_step

EXAMPLE_CONFIG = Path(__file__).parents[1] / "examples" / "config.toml"


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(d / f"img{i}.png"), rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
    return d


def test_parse_step_variants():
    t = parse_step("Resize(size=(64, 32), interpolation='nearest')")
    assert isinstance(t, Resize) and t.size == (64, 32) and t.interpolation == "nearest"
    assert isinstance(parse_step("Mirror"), Mirror)
    for bad in ("Nope()", "Resize(size=open('x'))", "Resize((1, 2))", "1 + 2", "Resize("):
        with pytest.raises(ValueError):
            parse_step(bad)


def test_help_via_module():
    result = subprocess.run(
        [sys.executable, "-m", "datapalette.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "apply" in result.stdout and "run" in result.stdout


def test_list_transforms(capsys):
    assert main(["list-transforms"]) == 0
    out = capsys.readouterr().out
    assert "Resize(size=(256, 256), interpolation='linear')" in out and "Compose" not in out


def test_apply_directory_with_steps(images_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["apply", str(images_dir), "-o", str(out), "--step", "Resize(size=(16, 24))"]) == 0
    files = sorted(out.glob("*.png"))
    assert [f.name for f in files] == ["img0.png", "img1.png", "img2.png"]
    assert cv2.imread(str(files[0])).shape == (16, 24, 3)


def test_apply_single_image_with_multi_output_transform(images_dir, tmp_path):
    out = tmp_path / "tiles"
    main(
        [
            "apply",
            str(images_dir / "img0.png"),
            "-o",
            str(out),
            "--step",
            "Tile(tile_size=(20, 20))",
        ]
    )
    assert len(list(out.glob("img0_*.png"))) == 6  # every tile, not just the first


def test_apply_with_toml_config_and_cli_override(images_dir, tmp_path):
    out = tmp_path / "cfg"
    assert main(["apply", str(images_dir), "-o", str(out), "--config", str(EXAMPLE_CONFIG)]) == 0
    h, w = cv2.imread(str(next(out.glob("*.png")))).shape[:2]
    assert h == w  # SquareCrop from the config ran
    out2 = tmp_path / "override"
    main(
        [
            "apply",
            str(images_dir),
            "-o",
            str(out2),
            "--config",
            str(EXAMPLE_CONFIG),
            "--step",
            "Resize(size=(8, 9))",
        ]
    )
    assert cv2.imread(str(next(out2.glob("*.png")))).shape[:2] == (8, 9)


def test_unknown_transform_exits_2(images_dir, tmp_path):
    assert main(["apply", str(images_dir), "-o", str(tmp_path / "x"), "--step", "Bogus"]) == 2


@pytest.mark.parametrize("pipeline", ["gan", "diffusion"])
def test_run_writes_outputs(pipeline, images_dir, tmp_path):
    out = tmp_path / pipeline
    assert main(["run", pipeline, "--images", str(images_dir), "--size", "32", "32", "-o", str(out)]) == 0
    assert len(list((out / "images").glob("*.png"))) == 3  # results are written, not discarded


def test_run_segmentation_writes_masks(images_dir, tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    for f in images_dir.iterdir():
        m = np.zeros((40, 60), np.uint8)
        m[10:30, 10:50] = 1
        cv2.imwrite(str(masks / f.name), m)
    out = tmp_path / "seg"
    assert (
        main(
            [
                "run",
                "segmentation",
                "--images",
                str(images_dir),
                "--masks",
                str(masks),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    mask = cv2.imread(str(next((out / "masks").glob("*.png"))), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask)) <= {0, 1}
