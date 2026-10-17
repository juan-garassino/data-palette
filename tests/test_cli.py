from __future__ import annotations

import subprocess
import sys

from datapalette.cli import main


class TestCLI:
    def test_main_is_importable(self):
        assert callable(main)

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "datapalette.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "DataPalette" in result.stdout
        assert "--pipeline" in result.stdout


class TestConfigMerge:
    def test_config_overrides_argparse_defaults(self, tmp_path):
        """A YAML config must be able to set options that have non-empty defaults."""
        from datapalette.cli import _parse_args

        cfg = tmp_path / "c.yaml"
        cfg.write_text("fps: 7\ncrop-size: [32, 48]\nmirror: true\n")
        args = _parse_args(
            ["--pipeline", "custom", "--input-type", "images", "--config", str(cfg)]
        )
        assert args.fps == 7
        assert list(args.crop_size) == [32, 48]
        assert args.mirror is True

    def test_explicit_cli_flag_beats_config(self, tmp_path):
        from datapalette.cli import _parse_args

        cfg = tmp_path / "c.yaml"
        cfg.write_text("fps: 7\n")
        args = _parse_args(
            ["--pipeline", "custom", "--input-type", "images", "--config", str(cfg),
             "--fps", "3"]
        )
        assert args.fps == 3


class TestTileDirectory:
    def test_writes_every_tile(self, tmp_path):
        import cv2
        import numpy as np

        from datapalette.cli import _tile_directory
        from datapalette.transforms import Tile

        src = tmp_path / "in"
        src.mkdir()
        cv2.imwrite(str(src / "a.png"), np.zeros((8, 8, 3), np.uint8))
        out = _tile_directory(str(src), str(tmp_path / "out"), Tile(tile_size=(4, 4)))
        assert len(out) == 4
        assert all(cv2.imread(p).shape == (4, 4, 3) for p in out)
