from __future__ import annotations

import tomllib
from typing import Any, Dict

__all__ = ["load_config"]


def load_config(config_file: str) -> Dict[str, Any]:
    """Load a TOML configuration file (stdlib ``tomllib``; no extra dependency).

    Hyphens in top-level keys become underscores so they match ``argparse`` names.
    """
    with open(config_file, "rb") as f:
        raw = tomllib.load(f)
    return {key.replace("-", "_"): value for key, value in raw.items()}
