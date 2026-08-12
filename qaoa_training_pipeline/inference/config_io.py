"""Torch-free config loading for model bundles.

Split out of ``model_loader.py`` (which imports torch) so the ONNX runtime
path can read ``model_config.json`` without importing torch. ``model_loader``
re-exports these for backwards compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "model_config.json"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse a ``model_config.json`` from ``config_path``.

    Accepts either the config file directly or its enclosing directory.

    Raises:
        FileNotFoundError: If ``model_config.json`` doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    config_path = Path(config_path)
    if config_path.is_dir():
        config_file = config_path / CONFIG_FILENAME
    else:
        config_file = config_path

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}. " f"Expected at: {config_file.absolute()}"
        )

    with open(config_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_bundle_path(config_path: Path, relative: str) -> Path:
    """Resolve a path stored in a config relative to the config's directory.

    Used for the ``checkpoint`` and ``onnx`` config fields, which are stored
    relative to the bundle directory.
    """
    config_path = Path(config_path)
    config_dir = config_path if config_path.is_dir() else config_path.parent
    return config_dir / str(relative)
