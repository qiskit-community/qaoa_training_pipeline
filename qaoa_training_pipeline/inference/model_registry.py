"""Resolution of ONNX model bundles from the HuggingFace Hub.

The model zoo lives entirely on the Hub: each bundle is a self-contained repo
holding ``model_config.json``, ``model.onnx`` and ``model.onnx.data`` at its
root. This package ships no model artifacts — only ``hf_setup.json``, which
maps a *bundle key* to the repo and the pinned revision it is served from.

A bundle key is ``<model>/p<p>`` (e.g. ``"gcn/p3"``): it is the *only* way to
address a model — used by :class:`~qaoa_training_pipeline.inference.AIInference`,
the tests and the tooling — and is independent of how the repos happen to be
named. Keys are verified against :data:`VERIFIED_ARCHITECTURES`, so a typo
fails loudly instead of resolving to the wrong repo, and the revision is
pinned by the setup file, so a run is reproducible. A bundle outside the shipped
zoo is reached by listing it in your own setup file and pointing
``$QAOA_HF_SETUP`` at it, not by passing a raw repo id.

:func:`resolve_bundle` downloads a bundle with ``snapshot_download``, which
caches under ``~/.cache/huggingface``, so only the first use touches the
network. Call :func:`prefetch_bundles` to warm the cache ahead of an offline /
air-gapped run.

The repos are private while the models are unreleased, so downloads need a
token (``hf auth login``, or ``HF_TOKEN`` in the environment).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

HF_DIR = Path(__file__).resolve().parent / "huggingface"

# The setup file that ships with the package is the source of truth. The only
# way to use another one is to say so explicitly via $QAOA_HF_SETUP, so a
# checkout always resolves models the same way regardless of what else happens
# to be lying around in this directory.
TRACKED_SETUP = HF_DIR / "hf_setup_local.json"
SETUP_ENV_VAR = "QAOA_HF_SETUP"

# The seven released architectures, keyed by the name used in a bundle key.
# For each: the short name the HuggingFace repo id ends with, and the
# ``model_init.model_type`` its exported config must declare.
#
# This table is the verified contract. It exists because the three names drift
# apart -- `graph_isomorphism_network` is `gin` on the Hub, and `mlp` is really
# an `agg_transformer` -- so an unvalidated name silently resolves to the wrong
# repo or the wrong input builder. Every name outside this table is rejected.
#
# Verified: the model_type column against the p1-p5 exported configs (all seven
# architectures, each loaded and run); the short-name column against the
# published `gnn` repo. The other six short names follow the same
# `qaoa_angles.max_cut.p{p}.{short}` scheme but have no repo on the Hub yet.
VERIFIED_ARCHITECTURES = {
    "diffusion_transformer": ("diffusion_transformer", "diffusion_transformer"),
    "edge_transformer": ("edge_transformer", "edge_transformer"),
    "gcn": ("gcn", "gcn"),
    "graph_isomorphism_network": ("gin", "graph_isomorphism_network"),
    "graph_neural_network": ("gnn", "graph_neural_network"),
    "graph_transformer": ("graph_transformer", "graph_transformer"),
    "mlp": ("mlp", "agg_transformer"),
}

# The released QAOA depths. A bundle key is <architecture>/p<p>.
P_VALUES = (1, 2, 3, 4, 5)

# Short name -> architecture, so a user who types the Hub's name for a model
# ("gnn/p1") gets told the key to use instead of a bare "unknown model".
_SHORT_TO_ARCHITECTURE = {short: arch for arch, (short, _) in VERIFIED_ARCHITECTURES.items()}

# The bundle layout, fixed by contract: every repo in the zoo holds exactly
# these three files at its root, under these names.
CONFIG_FILENAME = "model_config.json"
ONNX_FILENAME = "model.onnx"
ONNX_WEIGHTS_FILENAME = "model.onnx.data"

# Passed as snapshot_download allow_patterns so a repo gaining unrelated files
# (a README, a license) does not enlarge the download. The .onnx references its
# .data sidecar by relative filename, so both must land in the same directory —
# a snapshot always co-locates them.
BUNDLE_FILES = (CONFIG_FILENAME, ONNX_FILENAME, ONNX_WEIGHTS_FILENAME)

# Config fields that must not leave the private training environment. Stripped
# on upload, and again on load: a bundle published before the upload-side strip
# existed still carries them, and re-publishing cannot unpublish the old commit.
PRIVATE_CONFIG_FIELDS = ("checkpoint",)


def setup_path() -> Path:
    """The setup file in use: $QAOA_HF_SETUP if set, else the packaged one."""
    override = os.environ.get(SETUP_ENV_VAR)
    if not override:
        return TRACKED_SETUP

    path = Path(override).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{SETUP_ENV_VAR} points at a missing file: {path}")
    return path


def parse_bundle_key(bundle_key: str) -> tuple[str, int]:
    """Split a bundle key into its architecture and QAOA depth, rejecting unknown names.

    Raises:
        ValueError: If the key is malformed, names an architecture outside
            :data:`VERIFIED_ARCHITECTURES`, or names an unreleased depth.
    """
    architecture, _, depth = str(bundle_key).partition("/")
    if not architecture or not depth:
        raise ValueError(
            f"Malformed bundle key {bundle_key!r}. Expected '<architecture>/p<p>', "
            f"e.g. 'graph_neural_network/p1'. Available: {', '.join(available_bundles())}."
        )

    if architecture not in VERIFIED_ARCHITECTURES:
        hint = ""
        if architecture in _SHORT_TO_ARCHITECTURE:
            # The Hub repo names use short forms; the bundle key does not.
            hint = (
                f" {architecture!r} is the HuggingFace short name for "
                f"{_SHORT_TO_ARCHITECTURE[architecture]!r} -- did you mean "
                f"'{_SHORT_TO_ARCHITECTURE[architecture]}/{depth}'?"
            )
        raise ValueError(
            f"Unverified model architecture {architecture!r}. Verified architectures: "
            f"{', '.join(sorted(VERIFIED_ARCHITECTURES))}.{hint}"
        )

    if not (depth.startswith("p") and depth[1:].isdigit() and int(depth[1:]) in P_VALUES):
        raise ValueError(
            f"Unverified QAOA depth {depth!r} in bundle key {bundle_key!r}. "
            f"Released depths: {', '.join('p' + str(p) for p in P_VALUES)}."
        )
    return architecture, int(depth[1:])


def expected_model_type(bundle_key: str) -> str:
    """The ``model_init.model_type`` the bundle's config must declare."""
    architecture, _ = parse_bundle_key(bundle_key)
    return VERIFIED_ARCHITECTURES[architecture][1]


def expected_repo_suffix(bundle_key: str) -> str:
    """The trailing ``p<p>.<short>`` a bundle's repo id must carry."""
    architecture, depth = parse_bundle_key(bundle_key)
    return f"p{depth}.{VERIFIED_ARCHITECTURES[architecture][0]}"


def validate_setup(setup: dict[str, Any], source: Path) -> None:
    """Reject a setup file whose keys or repo names are not the verified ones.

    A typo here is not a loud failure by itself -- it is a download of some
    other repo, or of nothing -- so the names are checked up front rather than
    at the point of use.

    Raises:
        ValueError: On a malformed key, an unverified architecture or depth, or
            a repo id that disagrees with the key about which model it holds.
    """
    bundles = setup.get("bundles", {})
    if not bundles:
        raise ValueError(f"HF setup file {source} declares no bundles.")

    problems = []
    for bundle_key, entry in sorted(bundles.items()):
        try:
            suffix = expected_repo_suffix(bundle_key)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        repo_id = str(entry.get("repo_id", ""))
        if not repo_id.endswith("." + suffix):
            problems.append(
                f"bundle {bundle_key!r} maps to repo {repo_id!r}, which does not end in "
                f"{'.' + suffix!r} -- the key and the repo name disagree about the model."
            )

    if problems:
        raise ValueError(f"HF setup file {source} is invalid:\n  - " + "\n  - ".join(problems))


@lru_cache(maxsize=1)
def load_setup() -> dict[str, Any]:
    """Load, validate and cache the HF bundle setup file."""
    source = setup_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"HF setup file not found: {source}. The packaged setup file is "
            f"{TRACKED_SETUP.name}; set ${SETUP_ENV_VAR} to use another one."
        )
    with open(source, "r", encoding="utf-8") as handle:
        setup = json.load(handle)
    validate_setup(setup, source)
    return setup


def available_bundles() -> list[str]:
    """All bundle keys the setup file lists, sorted."""
    return sorted(load_setup().get("bundles", {}))


def published_bundles() -> list[str]:
    """Bundle keys that are actually uploaded (i.e. have a pinned revision)."""
    bundles = load_setup().get("bundles", {})
    return sorted(key for key, entry in bundles.items() if entry.get("revision"))


def bundle_entry(bundle_key: str) -> dict[str, Any]:
    """Return the setup entry for ``bundle_key``.

    Raises:
        ValueError: If the key does not name a verified architecture and depth.
        KeyError: If the key is not in the setup file.
        RuntimeError: If the bundle has no pinned revision (not yet uploaded).
    """
    # Validate the name before touching the setup file, so a wrong name fails
    # here with the verified set rather than as a download of the wrong repo.
    parse_bundle_key(bundle_key)

    bundles = load_setup().get("bundles", {})
    if bundle_key not in bundles:
        raise KeyError(
            f"Model bundle {bundle_key!r} is verified but absent from the setup file "
            f"({setup_path()}). Listed: {', '.join(available_bundles())}."
        )

    entry = bundles[bundle_key]
    if not entry.get("revision"):
        raise RuntimeError(
            f"Model bundle {bundle_key!r} is not published yet: its setup entry "
            f"({setup_path()}) has no pinned revision. Published bundles: "
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

    The download step of :func:`resolve_bundle`, which is what callers use: it
    takes the repo and revision from the setup file, so a bundle is always
    addressed by its key rather than by a raw repo id. To use a bundle outside
    the shipped zoo, list it in your own setup file and point ``$QAOA_HF_SETUP``
    at it.
    """
    setup = load_setup()
    snapshot_download = _snapshot_download()
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            repo_type=setup.get("repo_type", "model"),
            allow_patterns=list(setup.get("bundle_files", BUNDLE_FILES)),
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
