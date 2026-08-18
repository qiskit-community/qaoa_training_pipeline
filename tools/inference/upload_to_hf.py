"""Upload the ONNX weight bundles to the HuggingFace Hub and pin the manifest.

One-time (or on-retrain) migration step. The tiny ``model_config.json`` files
stay in git; this pushes the large ``model.onnx`` / ``model.onnx.data`` weights
to the Hub so they can be dropped from the repo and lazily downloaded at runtime
(see ``qaoa_training_pipeline/inference/model_registry.py``).

Requires ``huggingface_hub`` and a logged-in token with write access
(``huggingface-cli login`` or ``HF_TOKEN``). Run from the repository root:

    python tools/inference/upload_to_hf.py --repo-id org/qaoa-training-pipeline-models

It uploads every bundle under ``<path_prefix>/<model>/p<p>/``, then rewrites the
committed manifest with the real repo id and the resulting commit sha so
downloads are pinned to an immutable revision. After it succeeds and the pinned
manifest is committed, remove the weights from git:

    git rm --cached qaoa_training_pipeline/inference/model_configs/*/*/model.onnx \\
                    qaoa_training_pipeline/inference/model_configs/*/*/model.onnx.data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hf_manifest import MANIFEST_PATH, MODEL_CONFIGS_DIR, WEIGHT_FILENAMES, build_manifest
from model_keys import discover_model_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="target HF repo, e.g. org/name")
    parser.add_argument("--path-prefix", default="model_configs")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument(
        "--commit-message", default="Upload QAOA ONNX weight bundles", dest="message"
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo, upload_folder

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Upload only the weights, mirroring the in-repo layout under path_prefix.
    keys = discover_model_keys(MODEL_CONFIGS_DIR)
    allow = [f"*/{name}" for name in WEIGHT_FILENAMES]
    print(f"Uploading {len(keys)} bundles to {args.repo_id!r} under {args.path_prefix!r} ...")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(MODEL_CONFIGS_DIR),
        path_in_repo=args.path_prefix,
        allow_patterns=allow,
        commit_message=args.message,
    )

    # Resolve the resulting commit sha and pin the manifest to it.
    api = HfApi()
    revision = api.model_info(args.repo_id).sha
    manifest = build_manifest(args.repo_id, revision, args.path_prefix)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Pinned {Path(MANIFEST_PATH).name}: repo_id={args.repo_id!r}, revision={revision!r}.\n"
        "Commit the manifest, then git rm --cached the weight files (see module docstring)."
    )


if __name__ == "__main__":
    main()
