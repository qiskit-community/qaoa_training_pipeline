"""Torch-free resolution of ONNX weight bundles (local-first, HF fallback).

The tiny ``model_config.json`` files ship inside the package; the large
``model.onnx`` / ``model.onnx.data`` weights live on the HuggingFace Hub and are
downloaded lazily on first use, then cached. Resolution is **local-first**:

1. If the weight sits next to its config (bundled subset, a dev export, or a
   pre-fetched cache), it is used directly and no network access happens.
2. Otherwise it is downloaded from the Hub at the pinned revision recorded in
   ``model_configs/hf_manifest.json`` and integrity-checked against the
   manifest checksum.

This keeps offline / air-gapped and CI setups working provided the weights are
present locally (ship a subset, or run :func:`prefetch_bundles` ahead of time).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "model_configs"
MANIFEST_PATH = MODEL_CONFIGS_DIR / "hf_manifest.json"
WEIGHT_FILENAMES = ("model.onnx", "model.onnx.data")

# The manifest ships with this sentinel repo id until the weights are uploaded
# and it is pinned to a real repo + commit. We refuse to hit the network while
# it is still a placeholder so the failure is a clear message, not a 404.
_PLACEHOLDER_MARKER = "PLACEHOLDER"


def load_manifest() -> dict[str, Any]:
    """Load and cache the HF weight manifest."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"HF manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def bundle_key_for(config_path: Path) -> str:
    """Derive the ``<model>/p<p>`` bundle key from a config path or its dir."""
    config_path = Path(config_path)
    bundle_dir = config_path if config_path.is_dir() else config_path.parent
    return f"{bundle_dir.parent.name}/{bundle_dir.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: dict[str, Any]) -> None:
    actual = _sha256(path)
    if actual != expected["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {expected['sha256']}, got {actual}. "
            "The manifest and the downloaded weights are out of sync."
        )


def _download_bundle(bundle_key: str, manifest: dict[str, Any]) -> Path:
    """Download a bundle's weights from the Hub and return the local .onnx path."""
    repo_id = manifest["repo_id"]
    if _PLACEHOLDER_MARKER in repo_id:
        raise RuntimeError(
            f"ONNX weights for {bundle_key!r} are not present locally and the HF manifest "
            f"still points at the placeholder repo {repo_id!r}. Either ship the weights "
            "locally, or wait until the manifest is pinned to the uploaded HuggingFace repo."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Downloading ONNX weights from HuggingFace requires 'huggingface_hub'. "
            "Install the inference extra: pip install qaoa_training_pipeline[inference]."
        ) from exc

    files = manifest["bundles"][bundle_key]
    prefix = manifest.get("path_prefix", "").strip("/")
    onnx_local: Path | None = None
    # Both files must land in the same directory (the .onnx references its
    # .data sidecar by relative filename); hf_hub_download preserves the repo
    # layout in the cache, so co-location is guaranteed.
    for name in WEIGHT_FILENAMES:
        repo_filename = f"{prefix}/{bundle_key}/{name}" if prefix else f"{bundle_key}/{name}"
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                revision=manifest.get("revision"),
                repo_type=manifest.get("repo_type", "model"),
            )
        )
        _verify(cached, files[name])
        if name == "model.onnx":
            onnx_local = cached
    if onnx_local is None:  # pragma: no cover - WEIGHT_FILENAMES always includes model.onnx
        raise RuntimeError(f"No model.onnx entry for bundle {bundle_key!r} in the manifest.")
    return onnx_local


def ensure_onnx_local(config_path: Path, filename: str = "model.onnx") -> Path:
    """Return a local path to ``filename`` for the bundle at ``config_path``.

    Local-first: an existing file next to the config is returned untouched. If
    it is missing, the bundle is fetched from the Hub (per the manifest) and the
    cached path is returned.
    """
    config_path = Path(config_path)
    bundle_dir = config_path if config_path.is_dir() else config_path.parent
    local = bundle_dir / filename
    if local.is_file():
        return local

    manifest = load_manifest()
    bundle_key = bundle_key_for(config_path)
    if bundle_key not in manifest.get("bundles", {}):
        raise FileNotFoundError(
            f"ONNX weight {local} is missing and bundle {bundle_key!r} is not in the HF "
            f"manifest ({MANIFEST_PATH})."
        )
    return _download_bundle(bundle_key, manifest)


def prefetch_bundles(bundle_keys: list[str] | None = None) -> list[Path]:
    """Download and cache weights for the given bundles (all if ``None``).

    Useful to warm the cache before running in an air-gapped / offline setting.
    Returns the local ``model.onnx`` paths.
    """
    manifest = load_manifest()
    keys = bundle_keys if bundle_keys is not None else sorted(manifest.get("bundles", {}))
    return [_download_bundle(key, manifest) for key in keys]
