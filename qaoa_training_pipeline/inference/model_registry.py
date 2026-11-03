#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Resolution of ONNX model bundles from the HuggingFace Hub.

The model zoo lives entirely on the Hub: each bundle is a self-contained repo
holding ``model_config.json``, ``model.onnx`` and ``model.onnx.data`` at its
root. This package ships no model artifacts — only ``hf_setup.json``, which
maps a *bundle key* to the repo and the pinned revision it is served from.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_DIR = Path(__file__).resolve().parent / "huggingface"

TRACKED_SETUP = HF_DIR / "hf_setup.json"
SETUP_ENV_VAR = "QAOA_HF_SETUP"
SETUP_SCHEMA = 2

# The seven released architectures, keyed by the name used in a bundle key.
VERIFIED_ARCHITECTURES = {
    "diffusion_transformer": ("dit", "diffusion_transformer"),
    "edge_transformer": ("et", "edge_transformer"),
    "gcn": ("gcn", "gcn"),
    "graph_isomorphism_network": ("gin", "graph_isomorphism_network"),
    "graph_neural_network": ("gnn", "graph_neural_network"),
    "graph_transformer": ("gt", "graph_transformer"),
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
# (a README, a license) does not enlarge the download.
BUNDLE_FILES = (CONFIG_FILENAME, ONNX_FILENAME, ONNX_WEIGHTS_FILENAME)

# Config fields that must not leave the private training environment.
PRIVATE_CONFIG_FIELDS = ("checkpoint",)


def setup_path() -> Path:
    """The setup file in use: $QAOA_HF_SETUP if set, else the packaged one."""
    override = os.environ.get(SETUP_ENV_VAR)
    if not override:
        return TRACKED_SETUP.resolve()

    path = Path(override).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{SETUP_ENV_VAR} points at a missing file: {path}")
    return path.resolve()


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

    # A full match on p<digits> with no leading zero, so "p01" and "p1x" are
    # rejected rather than resolving to p1
    match = re.fullmatch(r"p([1-9][0-9]*)", depth)
    if not match or int(match.group(1)) not in P_VALUES:
        raise ValueError(
            f"Unverified QAOA depth {depth!r} in bundle key {bundle_key!r}. "
            f"Released depths: {', '.join('p' + str(p) for p in P_VALUES)}."
        )
    return architecture, int(match.group(1))


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

    Raises:
        ValueError: On a malformed key, an unverified architecture or depth, a
            repo id that disagrees with the key about which model it holds, or a
            revision that is not a commit sha.
    """
    schema = setup.get("schema")
    if schema != SETUP_SCHEMA:
        raise ValueError(
            f"HF setup file {source} declares schema {schema!r}, but this version reads "
            f"schema {SETUP_SCHEMA}. The file was written for a different layout."
        )

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

        # The convention the bundle was trained under. Required: the package
        # keeps no default, so a missing or zero prefactor is caught here
        # rather than surfacing as wrong weights or a division by zero.
        if "prefactor" not in entry:
            problems.append(
                f"bundle {bundle_key!r} declares no prefactor. Every entry must state the "
                "cost-operator convention its model was trained under, "
                "H = prefactor * sum_e w_e Z_i Z_j (-0.5 for the shipped max-cut models)."
            )
        else:
            try:
                numeric = float(entry["prefactor"])
            except (TypeError, ValueError):
                numeric = 0.0
            if numeric == 0.0:
                problems.append(
                    f"bundle {bundle_key!r} declares prefactor {entry['prefactor']!r}, which "
                    "must be a non-zero number: it is what the Pauli coefficients are divided "
                    "by to recover the edge weights the model consumes."
                )

        # A pinned revision must be a commit sha
        revision = entry.get("revision")
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            problems.append(
                f"bundle {bundle_key!r} pins revision {revision!r}, which is not a 40-character "
                "commit sha. A branch or tag name is not reproducible; use the commit sha "
                "(huggingface_hub.HfApi().model_info(repo_id).sha)."
            )

    if problems:
        raise ValueError(f"HF setup file {source} is invalid:\n  - " + "\n  - ".join(problems))


@lru_cache(maxsize=4)
def _load_setup_file(source: Path, _stamp: tuple[int, int]) -> dict[str, Any]:
    """Load and validate one setup file, caching the result per path and stamp."""
    if not source.is_file():
        raise FileNotFoundError(
            f"HF setup file not found: {source}. The packaged setup file is "
            f"{TRACKED_SETUP.name}; set ${SETUP_ENV_VAR} to use another one."
        )
    with open(source, "r", encoding="utf-8") as handle:
        setup = json.load(handle)
    validate_setup(setup, source)
    return setup


def load_setup() -> dict[str, Any]:
    """Load, validate and cache the HF bundle setup file.

    The cache is keyed on the resolved path and the file's mtime and size, not
    on nothing. $QAOA_HF_SETUP can be set or changed after this module is
    imported, and a setup file can be rewritten in place -- by the publisher
    tooling pinning a revision, most often -- and a process-lifetime cache
    would then keep serving whichever bundles were read first.
    """
    path = setup_path()
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        # Let _load_setup_file raise the informative FileNotFoundError.
        stamp = (0, 0)
    return _load_setup_file(path, stamp)


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
    """Download a bundle repo and return the local directory holding its files."""
    setup = load_setup()
    snapshot_download = _snapshot_download()
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    logger.debug("Resolving bundle repo %s at revision %s", repo_id, revision)
    try:
        path = snapshot_download(
            repo_id,
            revision=revision,
            repo_type=setup.get("repo_type", "model"),
            allow_patterns=list(setup.get("bundle_files", BUNDLE_FILES)),
        )
    except (OfflineModeIsEnabled, LocalEntryNotFoundError) as exc:
        # Raised when the bundle is not cached and cannot be fetched
        raise RuntimeError(
            f"Bundle repo {repo_id} at revision {revision} is not in the local HuggingFace "
            "cache and could not be downloaded (offline mode, or no network). Resolve the "
            "bundles you need from a networked run first so they land in the cache, or allow "
            "network access (unset HF_HUB_OFFLINE)."
        ) from exc
    logger.debug("Bundle repo %s available at %s", repo_id, path)
    return Path(path)


def resolve_bundle(bundle_key: str) -> Path:
    """Download the bundle for ``bundle_key`` and return its local directory.

    The returned directory contains ``model_config.json`` next to the ONNX
    artifacts, which is what :class:`OnnxQAOAPredictor` expects.

    Raises:
        RuntimeError: If the bundle is unpublished, or is neither cached nor
            downloadable. The message names the bundle key, since that is what
            the caller passed -- the repo id is an implementation detail of the
            setup file.
    """
    entry = bundle_entry(bundle_key)
    try:
        return download_bundle(entry["repo_id"], revision=entry["revision"])
    except RuntimeError as exc:
        raise RuntimeError(f"Could not resolve model bundle {bundle_key!r}. {exc}") from exc
