"""Resolution of ONNX model bundles from the HuggingFace Hub.

The model zoo lives entirely on the Hub: each bundle is a self-contained repo
holding ``model_config.json``, ``model.onnx`` and ``model.onnx.data`` at its
root. This package ships no model artifacts — only ``hf_setup.json``, which
maps a *bundle key* to the repo and the pinned revision it is served from.

A bundle key is ``<model>/p<p>`` (e.g. ``"gcn/p3"``): it is the stable public
identifier used by :class:`~qaoa_training_pipeline.inference.AIInference`, the
tests and the tooling, and is independent of how the repos happen to be named.

:func:`resolve_bundle` downloads a bundle with ``snapshot_download``, which
caches under ``~/.cache/huggingface``, so only the first use touches the
network. Call :func:`prefetch_bundles` to warm the cache ahead of an offline /
air-gapped run.

The repos are private while the models are unreleased, so downloads need a
token (``hf auth login``, or ``HF_TOKEN`` in the environment).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

hf_paths = Path(__file__).resolve().parent / "huggingface/hf_setup_local.json"

# Files that make up a bundle; passed as snapshot_download allow_patterns so a
# repo gaining unrelated files (a README, a license) does not enlarge the
# download. The .onnx references its .data sidecar by relative filename, so
# both must land in the same directory — a snapshot always co-locates them.
BUNDLE_FILES = ("model_config.json", "model.onnx", "model.onnx.data")


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    """Load and cache the HF bundle manifest."""
    if not hf_paths.is_file():
        raise FileNotFoundError(f"HF manifest not found: {hf_paths}")
    with open(hf_paths, "r", encoding="utf-8") as handle:
        return json.load(handle)


def available_bundles() -> list[str]:
    """All bundle keys the manifest knows about, sorted."""
    return sorted(load_manifest().get("bundles", {}))


def published_bundles() -> list[str]:
    """Bundle keys that are actually uploaded (i.e. have a pinned revision)."""
    bundles = load_manifest().get("bundles", {})
    return sorted(key for key, entry in bundles.items() if entry.get("revision"))


def bundle_entry(bundle_key: str) -> dict[str, Any]:
    """Return the manifest entry for ``bundle_key``.

    Raises:
        KeyError: If the key is not in the manifest.
        RuntimeError: If the bundle has no pinned revision (not yet uploaded).
    """
    bundles = load_manifest().get("bundles", {})
    if bundle_key not in bundles:
        raise KeyError(
            f"Unknown model bundle {bundle_key!r}. Available: {', '.join(available_bundles())}."
        )

    entry = bundles[bundle_key]
    if not entry.get("revision"):
        raise RuntimeError(
            f"Model bundle {bundle_key!r} is not published yet: its manifest entry "
            f"({hf_paths}) has no pinned revision. Published bundles: "
            f"{', '.join(published_bundles()) or 'none'}."
        )
    return entry


def _snapshot_download():
    """Import ``snapshot_download`` with a helpful error if the extra is missing."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Downloading model bundles from HuggingFace requires 'huggingface_hub'. "
            "Install the inference extra: pip install qaoa_training_pipeline[inference]."
        ) from exc
    return snapshot_download


def download_bundle(repo_id: str, revision: str = "main") -> Path:
    """Download a bundle repo and return the local directory holding its files.

    Escape hatch for a repo that is not in the manifest (a private export, a
    fork, an unreleased retrain). Prefer :func:`resolve_bundle` for the shipped
    zoo, which pins the revision for you.
    """
    manifest = load_manifest()
    snapshot_download = _snapshot_download()
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            repo_type=manifest.get("repo_type", "model"),
            allow_patterns=list(manifest.get("bundle_files", BUNDLE_FILES)),
        )
    )


def resolve_bundle(bundle_key: str) -> Path:
    """Download the bundle for ``bundle_key`` and return its local directory.

    The returned directory contains ``model_config.json`` next to the ONNX
    artifacts, which is what :class:`OnnxQAOAPredictor` expects.
    """
    entry = bundle_entry(bundle_key)
    return download_bundle(entry["repo_id"], revision=entry["revision"])


def prefetch_bundles(bundle_keys: list[str] | None = None) -> list[Path]:
    """Download and cache the given bundles (all published ones if ``None``).

    Useful to warm the cache before running in an air-gapped / offline setting.
    Returns the local bundle directories.
    """
    keys = bundle_keys if bundle_keys is not None else published_bundles()
    return [resolve_bundle(key) for key in keys]
