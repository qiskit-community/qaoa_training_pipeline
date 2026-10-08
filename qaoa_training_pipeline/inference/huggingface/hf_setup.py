"""Refresh the pinned revisions in the HuggingFace bundle hf_setup.

The model zoo lives on the Hub: each bundle is a self-contained repo holding
``model_config.json`` + ``model.onnx`` + ``model.onnx.data``, and
``qaoa_training_pipeline/inference/hf_setup.json`` maps each bundle key
``<model>/p<p>`` to that repo plus the commit it is served from. Pinning to an
immutable commit is what makes a download reproducible.

This script re-reads the current head of every repo in the hf_setup and writes
it back as the pinned ``revision``. A bundle whose repo does not exist yet keeps
``revision: null``, which the runtime reports as "not published yet".

Run from the repository root:

    python tools/inference/hf_hf_setup.py              # refresh all
    python tools/inference/hf_hf_setup.py --only gcn   # one architecture
    python tools/inference/hf_hf_setup.py --dry-run

Requires ``huggingface_hub`` and, while the repos are private, a token with read
access (``hf auth login`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_keys import resolve_model_keys  # local sibling module

from qaoa_training_pipeline.inference.model_registry import (
    hf_paths,
)  # noqa: E402  (model_keys sets sys.path)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_hf_setup() -> dict:
    """Read the hf_setup straight from disk (not the runtime's cached copy)."""
    return json.loads(hf_paths.read_text(encoding="utf-8"))


def write_hf_setup(hf_setup: dict) -> None:
    """Write the hf_setup back, preserving the committed formatting."""
    hf_paths.write_text(json.dumps(hf_setup, indent=2) + "\n", encoding="utf-8")


def refresh_revisions(hf_setup: dict, keys: list[str]) -> tuple[int, int]:
    """Pin each key's revision to its repo's current head.

    Returns the ``(updated, missing)`` counts. A repo that cannot be reached is
    left as-is rather than clobbered, so a partial outage or a missing token
    never silently unpins a working bundle.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi()
    repo_type = hf_setup.get("repo_type", "model")
    updated = missing = 0

    for key in keys:
        entry = hf_setup["bundles"][key]
        repo_id = entry["repo_id"]
        try:
            sha = api.repo_info(repo_id, repo_type=repo_type).sha
        except (RepositoryNotFoundError, HfHubHTTPError) as exc:
            missing += 1
            print(f"  {key:40s} {repo_id}  UNAVAILABLE ({type(exc).__name__})")
            continue
        if entry.get("revision") == sha:
            print(f"  {key:40s} {repo_id}  unchanged")
            continue
        print(f"  {key:40s} {repo_id}  {entry.get('revision')} -> {sha}")
        entry["revision"] = sha
        updated += 1

    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="all",
        help="bundle key ('gcn/p3'), architecture ('gcn'), or 'all' (default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    hf_setup = ()
    keys = [k for k in resolve_model_keys(args.only) if k in hf_setup["bundles"]]
    if not keys:
        parser.error(f"--only {args.only!r} matched no bundle in {hf_paths}")

    print(f"Refreshing {len(keys)} bundle revision(s) from the Hub ...")
    updated, missing = refresh_revisions(hf_setup, keys)

    if updated and not args.dry_run:
        (hf_setup)
        print(f"\nWrote {hf_paths.relative_to(REPO_ROOT)} ({updated} updated).")
    elif updated:
        print(f"\n--dry-run: {updated} entries would change.")
    else:
        print("\nNothing to update.")

    unpinned = [k for k, e in hf_setup["bundles"].items() if not e.get("revision")]
    if unpinned:
        print(
            f"{len(unpinned)} bundle(s) still unpublished: {', '.join(sorted(unpinned))}\n"
            "Upload them with tools/inference/upload_to_hf.py, then re-run this script."
        )
    if missing:
        print(f"{missing} repo(s) were unreachable; their entries were left untouched.")


if __name__ == "__main__":
    main()
