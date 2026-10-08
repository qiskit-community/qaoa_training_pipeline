"""Shared discovery of model bundles, backed by the HF manifest.

A *bundle key* is ``"<model>/p<p>"`` (e.g. ``"gcn/p3"``); it is the identifier
the tooling and tests pass around. Bundles themselves live on the HuggingFace
Hub — ``qaoa_training_pipeline/inference/hf_setup.json`` maps each key to its
repo and pinned revision, and is the authoritative list of what exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the tools runnable from a bare checkout, without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qaoa_training_pipeline.inference.model_registry import (  # noqa: E402
    available_bundles,
)


def discover_model_keys() -> list[str]:
    """All bundle keys ``<model>/p<p>`` in the manifest, sorted."""
    return available_bundles()


def resolve_model_keys(arg: str) -> list[str]:
    """Turn a ``--model`` argument into concrete bundle keys.

    Accepts ``"all"``, a full key such as ``"gcn/p3"``, or a bare model name
    such as ``"gcn"`` (expands to every ``p`` available for that architecture).
    """
    keys = discover_model_keys()
    if arg == "all":
        return keys
    if "/" in arg:
        return [arg]
    return [k for k in keys if k.split("/", 1)[0] == arg]


def baseline_filename(model_key: str) -> str:
    """Flatten a bundle key to a baseline filename, e.g. ``gcn/p3`` -> ``gcn_p3.json``."""
    return model_key.replace("/", "_") + ".json"
