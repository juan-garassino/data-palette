"""YAML configuration loading."""

from __future__ import annotations

from typing import Any, Dict

import yaml

__all__ = ["load_config"]


def load_config(config_file: str) -> Dict[str, Any]:
    """Load a YAML configuration file and normalize keys.

    Hyphens in keys are replaced with underscores so that they match
    ``argparse`` attribute names.

    Parameters
    ----------
    config_file : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed and key-normalized configuration.
    """
    with open(config_file, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    return {key.replace("-", "_"): value for key, value in raw.items()}
