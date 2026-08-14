"""Shared discovery of model_config bundles under the nested layout.

Bundles live at ``model_configs/<model>/p<p>/model_config.json``. A *bundle key*
is the relative directory ``"<model>/p<p>"`` (e.g. ``"gcn/p3"``); it is the
identifier the tooling and tests pass around, and it also joins cleanly onto
``MODEL_CONFIGS_DIR`` to reach the config/ONNX artifacts.
"""

from __future__ import annotations

from pathlib import Path


def discover_model_keys(model_configs_dir: Path) -> list[str]:
    """All bundle keys ``<model>/p<p>`` with a ``model_config.json``, sorted."""
    return sorted(
        p.parent.relative_to(model_configs_dir).as_posix()
        for p in model_configs_dir.glob("*/p*/model_config.json")
    )


def resolve_model_keys(arg: str, model_configs_dir: Path) -> list[str]:
    """Turn a ``--model`` argument into concrete bundle keys.

    Accepts ``"all"``, a full key such as ``"gcn/p3"``, or a bare model name
    such as ``"gcn"`` (expands to every ``p`` available for that architecture).
    """
    keys = discover_model_keys(model_configs_dir)
    if arg == "all":
        return keys
    if "/" in arg:
        return [arg]
    return [k for k in keys if k.split("/", 1)[0] == arg]


def baseline_filename(model_key: str) -> str:
    """Flatten a bundle key to a baseline filename, e.g. ``gcn/p3`` -> ``gcn_p3.json``."""
    return model_key.replace("/", "_") + ".json"
