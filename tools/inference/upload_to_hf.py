#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Upload ONNX model bundles to the HuggingFace Hub, one repo per bundle.

Each bundle becomes a self-contained repo holding ``model_config.json``,
``model.onnx`` and ``model.onnx.data`` at its root, named after the setup
file's ``repo_id`` for that bundle key. After uploading, the resulting commit
sha is pinned into the setup file the runtime resolves (``$QAOA_HF_SETUP`` when
set, otherwise the packaged
``qaoa_training_pipeline/inference/huggingface/hf_setup.json``) so downloads are
reproducible.

The bundles are not in this repository (the runtime downloads them), so point
``--source`` at a local export tree laid out as ``<source>/<model>/p<p>/``.

Run from the repository root, against an installed package (``pip install -e .``):

    python tools/inference/upload_to_hf.py --source /path/to/exports
    python tools/inference/upload_to_hf.py --source /path/to/exports --only gcn
    python tools/inference/upload_to_hf.py --source /path/to/exports --dry-run

The uploaded ``model_config.json`` is stripped of its ``checkpoint`` field,
which records a path inside the private training-checkpoint tree and must not
be published.

Requires ``huggingface_hub`` and a token with write access (``hf auth login``
or ``HF_TOKEN``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from hf_setup import SETUP_PATH, read_setup, write_setup  # local sibling module
from model_keys import resolve_model_keys  # local sibling module

from qaoa_training_pipeline.inference.model_registry import (
    BUNDLE_FILES,
    PRIVATE_CONFIG_FIELDS,
)


def stage_bundle(bundle_dir: Path, staging: Path) -> Path:
    """Copy a bundle into ``staging`` with its config sanitized.

    Uploading from a staging copy keeps the local export tree untouched and
    guarantees only the three bundle files are pushed.
    """
    staged = staging / bundle_dir.name
    staged.mkdir(parents=True)

    for name in BUNDLE_FILES:
        src = bundle_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Bundle {bundle_dir} is missing {name}.")
        if name != "model_config.json":
            shutil.copy2(src, staged / name)
            continue

        config = json.loads(src.read_text(encoding="utf-8"))
        dropped = [f for f in PRIVATE_CONFIG_FIELDS if config.pop(f, None) is not None]
        if dropped:
            print(f"    stripped private field(s): {', '.join(dropped)}")
        (staged / name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return staged


def upload_bundle(
    repo_id: str, bundle_dir: Path, repo_type: str, private: bool, message: str
) -> str:
    """Create/update ``repo_id`` from ``bundle_dir`` and return the commit sha."""
    from huggingface_hub import HfApi, create_repo, upload_folder

    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_bundle(bundle_dir, Path(tmp))
        upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(staged),
            commit_message=message,
        )
    return HfApi().repo_info(repo_id, repo_type=repo_type).sha


def main() -> None:
    """Upload bundles of a local export tree and pin their revisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="local export tree laid out as <source>/<model>/p<p>/",
    )
    parser.add_argument(
        "--only",
        default="all",
        help="bundle key ('gcn/p3'), architecture ('gcn'), or 'all' (default)",
    )
    parser.add_argument(
        "--public", action="store_true", help="create repos public (default: private)"
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be uploaded")
    parser.add_argument("--commit-message", default="Upload QAOA ONNX bundle", dest="message")
    args = parser.parse_args()

    setup = read_setup()
    repo_type = setup.get("repo_type", "model")
    keys = [k for k in resolve_model_keys(args.only) if k in setup["bundles"]]
    if not keys:
        parser.error(f"--only {args.only!r} matched no bundle in {SETUP_PATH}")

    uploaded = skipped = 0
    for key in keys:
        repo_id = setup["bundles"][key]["repo_id"]
        bundle_dir = args.source / key
        print(f"[{key}] -> {repo_id}")
        if not (bundle_dir / "model_config.json").is_file():
            print(f"    SKIP: no bundle at {bundle_dir}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"    --dry-run: would upload {bundle_dir}")
            continue
        sha = upload_bundle(repo_id, bundle_dir, repo_type, not args.public, args.message)
        setup["bundles"][key]["revision"] = sha
        print(f"    pinned revision {sha}")
        uploaded += 1

    if uploaded:
        write_setup(setup)
        print(f"\nUploaded {uploaded} bundle(s); updated {SETUP_PATH}. Commit the setup file.")
    else:
        print("\nNothing uploaded.")
    if skipped:
        print(f"{skipped} bundle(s) were absent from --source and left unpinned.")


if __name__ == "__main__":
    main()
