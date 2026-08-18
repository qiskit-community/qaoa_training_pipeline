"""Generate / refresh the HuggingFace weight manifest.

The torch-free ONNX runtime keeps only the tiny ``model_config.json`` files in
git; the ``model.onnx`` / ``model.onnx.data`` weights live on the HuggingFace
Hub and are lazily downloaded on first use (see
``qaoa_training_pipeline/inference/model_registry.py``). This script freezes the
mapping ``<model>/p<p>`` -> weight checksums + sizes into
``model_configs/hf_manifest.json`` so downloads can be integrity-checked and
pinned.

Run from the repository root after (re)exporting ONNX bundles, while the
weights are still present locally:

    python tools/inference/hf_manifest.py                 # refresh checksums
    python tools/inference/hf_manifest.py --repo-id org/name --revision <sha>

``--repo-id`` / ``--revision`` are optional; omit them to keep whatever the
manifest already records (they are pinned once the weights are uploaded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_keys import discover_model_keys  # local sibling module

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS_DIR = REPO_ROOT / "qaoa_training_pipeline" / "inference" / "model_configs"
MANIFEST_PATH = MODEL_CONFIGS_DIR / "hf_manifest.json"
WEIGHT_FILENAMES = ("model.onnx", "model.onnx.data")

# Placeholder until the weights are uploaded and the manifest is pinned to a
# concrete repo + immutable commit. The runtime refuses to download while the
# repo id is still a placeholder (see model_registry.py).
PLACEHOLDER_REPO_ID = "PLACEHOLDER_ORG/qaoa-training-pipeline-models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repo_id: str, revision: str, path_prefix: str) -> dict:
    """Scan the local weight files and build the manifest dict."""
    bundles: dict[str, dict] = {}
    for key in discover_model_keys(MODEL_CONFIGS_DIR):
        bundle_dir = MODEL_CONFIGS_DIR / key
        files: dict[str, dict] = {}
        for name in WEIGHT_FILENAMES:
            weight = bundle_dir / name
            if not weight.is_file():
                raise FileNotFoundError(
                    f"Missing weight {weight}; export bundles before refreshing the manifest."
                )
            files[name] = {"sha256": _sha256(weight), "size": weight.stat().st_size}
        bundles[key] = files
    return {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": "model",
        # HF repo mirrors the in-repo layout under this prefix:
        #   <path_prefix>/<model>/p<p>/model.onnx[.data]
        "path_prefix": path_prefix,
        "bundles": bundles,
    }


def main() -> None:
    existing = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=existing.get("repo_id", PLACEHOLDER_REPO_ID))
    parser.add_argument("--revision", default=existing.get("revision", "main"))
    parser.add_argument("--path-prefix", default=existing.get("path_prefix", "model_configs"))
    args = parser.parse_args()

    manifest = build_manifest(args.repo_id, args.revision, args.path_prefix)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} "
        f"({len(manifest['bundles'])} bundles, repo_id={manifest['repo_id']!r}, "
        f"revision={manifest['revision']!r})"
    )


if __name__ == "__main__":
    main()
