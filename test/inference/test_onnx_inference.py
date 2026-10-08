# TODO: REDO ALL TEST CASES ONCE REMAINING INFERENCE PIPELINE IS READY.

#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""End-to-end tests for the torch-free ONNX inference path.

Covered entrypoint:
  * OnnxQAOAPredictor (qaoa_training_pipeline/inference/onnx_predictor.py)

Model bundles are ingested from the HuggingFace Hub, so the tests come in two
flavours:
  * Offline tests exercise the manifest and the bundle-resolution logic with a
    mocked ``snapshot_download``. They always run.
  * Model-running tests (including the frozen-baseline regression) need an
    actual bundle. They download it, which is opt-in: set ``QTP_HF_TESTS=1``, or
    they run automatically against bundles already in the local HF cache. While
    the repos are private a token is needed too (``hf auth login`` / ``HF_TOKEN``).

``onnxruntime`` is an optional dependency (the ``inference`` extra), so the whole
module skips if it is not importable.
"""

import importlib.util
import json
import math
import os
import unittest
from pathlib import Path
from unittest import mock

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference import model_registry

from ..training_pipeline_test_case import TrainingPipelineTestCase

HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
HAS_HF_HUB = importlib.util.find_spec("huggingface_hub") is not None

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# Model architectures in the zoo. The MLP bundle key is ``mlp`` (its
# ``model_type`` inside the config is still ``agg_transformer``).
MODEL_NAMES = [
    "diffusion_transformer",
    "edge_transformer",
    "gcn",
    "graph_isomorphism_network",
    "graph_neural_network",
    "graph_transformer",
    "mlp",
]

# QAOA depths per architecture; bundle keys are ``<model>/p<p>``.
P_VALUES = [1, 2, 3, 4, 5]

# The manifest is the authoritative list of bundles; MODEL_NAMES/P_VALUES above
# are cross-checked against it by test_manifest_covers_expected_zoo.
MODEL_KEYS = model_registry.available_bundles()

# Graph-consuming model exercised by the behavioral tests alongside the
# scalar-only mlp; each across all shipped depths.
BEHAVIORAL_MODELS = ["mlp", "graph_neural_network"]
BEHAVIORAL_KEYS = [f"{model}/p{p}" for model in BEHAVIORAL_MODELS for p in P_VALUES]

# Downloading is opt-in so CI and air-gapped runs stay green while the model
# repos are private.
HF_TESTS_ENABLED = os.environ.get("QTP_HF_TESTS") == "1"


def bundle_is_cached(model_key):
    """True if the bundle's config is already in the local HF cache.

    Lets the model-running tests run without QTP_HF_TESTS once a developer has
    pulled a bundle, while never reaching the network by itself.
    """
    if not HAS_HF_HUB:
        return False
    from huggingface_hub import try_to_load_from_cache

    entry = model_registry.load_manifest()["bundles"].get(model_key) or {}
    if not entry.get("revision"):
        return False
    cached = try_to_load_from_cache(
        entry["repo_id"], "model_config.json", revision=entry["revision"]
    )
    return isinstance(cached, str)


def skip_reason(model_key):
    """Why a model-running subtest cannot run, or None if it can."""
    entry = model_registry.load_manifest()["bundles"].get(model_key) or {}
    if not entry.get("revision"):
        return f"bundle {model_key!r} is not published on the Hub yet"
    if not (HF_TESTS_ENABLED or bundle_is_cached(model_key)):
        return f"set QTP_HF_TESTS=1 to download {model_key!r} from the Hub"
    return None


# The graph_transformer's Laplacian positional encoding relies on an
# eigensolver whose eigenvectors are sign-ambiguous; the model was trained with
# random sign-flip augmentation and tolerates the difference — hence a looser
# tolerance for it.
PARITY_ATOL = {"graph_transformer": 2e-3}
DEFAULT_PARITY_ATOL = 1e-4


# --- deterministic cost operators (mirrors tools/inference/bench_ops.py) ----


def _zz(num_qubits, i, j, weight=1.0):
    label = ["I"] * num_qubits
    label[i] = "Z"
    label[j] = "Z"
    return "".join(label), weight


def _ring(n, weight=1.0):
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, weight) for k in range(n)])


def _line(n, weight=1.0):
    return SparsePauliOp.from_list([_zz(n, k, k + 1, weight) for k in range(n - 1)])


def _complete(n, weight=1.0):
    return SparsePauliOp.from_list(
        [_zz(n, i, j, weight) for i in range(n) for j in range(i + 1, n)]
    )


def _weighted_ring(n):
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, 0.5 + 0.25 * k) for k in range(n)])


BENCH_OPS = {
    "triangle_3": _complete(3),
    "line_4": _line(4),
    "ring_4": _ring(4),
    "complete_4": _complete(4),
    "ring_6": _ring(6),
    "line_8": _line(8),
    "weighted_ring_6": _weighted_ring(6),
    "complete_5": _complete(5),
}


def _p_of(model_key):
    """QAOA depth encoded in a bundle key, e.g. ``gcn/p3`` -> 3."""
    return int(model_key.rsplit("/p", 1)[1])


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestOnnxInference(TrainingPipelineTestCase):
    """Torch-free ONNX predictor tests, run against bundles from the Hub.

    Each subtest skips individually when its bundle is unpublished or when
    downloading is not enabled (see ``skip_reason``).
    """

    def _predictor(self, model_key):
        """Download (or reuse the cached) bundle and wrap it in a predictor."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        return OnnxQAOAPredictor.from_bundle(model_key, device="cpu")

    def test_onnx_predictor_loads_and_predicts(self):
        """Every exported bundle loads via onnxruntime and produces 2*p angles."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in MODEL_KEYS:
            with self.subTest(model=model_key):
                reason = skip_reason(model_key)
                if reason:
                    self.skipTest(reason)
                angles = self._predictor(model_key).predict(op)
                self.assertIsInstance(angles, list)
                self.assertEqual(len(angles), 2 * _p_of(model_key))  # [betas..., gammas...]
                self.assertTrue(all(isinstance(a, float) and math.isfinite(a) for a in angles))

    def test_onnx_predictor_is_deterministic(self):
        """Inference is deterministic: same input -> identical output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in BEHAVIORAL_KEYS:
            with self.subTest(model=model_key):
                reason = skip_reason(model_key)
                if reason:
                    self.skipTest(reason)
                predictor = self._predictor(model_key)
                self.assertEqual(predictor.predict(op), predictor.predict(op))

    def test_onnx_predictor_reacts_to_input(self):
        """Different problem graphs yield different predicted angles."""
        op_triangle = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        op_line4 = SparsePauliOp.from_list([("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0)])
        for model_key in BEHAVIORAL_KEYS:
            with self.subTest(model=model_key):
                reason = skip_reason(model_key)
                if reason:
                    self.skipTest(reason)
                predictor = self._predictor(model_key)
                self.assertNotEqual(predictor.predict(op_triangle), predictor.predict(op_line4))

    def test_onnx_predictor_raw_vs_denormalized_differ(self):
        """denormalize=False returns the raw (unscaled) model output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in BEHAVIORAL_KEYS:
            with self.subTest(model=model_key):
                reason = skip_reason(model_key)
                if reason:
                    self.skipTest(reason)
                predictor = self._predictor(model_key)
                raw = predictor.predict(op, denormalize=False)
                scaled = predictor.predict(op, denormalize=True)
                # config output_scale = pi/2, so scaled == raw * pi/2 for the betas
                # (first half); the gammas additionally undergo the rescale_a
                # division, so compare only the betas here (p=1 -> index 0).
                p = predictor.output_dim // 2
                for raw_beta, scaled_beta in zip(raw[:p], scaled[:p]):
                    self.assertAlmostEqual(scaled_beta, raw_beta * (math.pi / 2), places=5)

    def test_onnx_matches_baseline(self):
        """The ONNX predictor reproduces the frozen baseline for every op.

        Baselines are committed under ``test/inference/baselines/`` and were
        frozen from the original predictor at export time.
        """
        for model_key in MODEL_KEYS:
            with self.subTest(model=model_key):
                reason = skip_reason(model_key)
                if reason:
                    self.skipTest(reason)
                baseline_file = BASELINE_DIR / f"{model_key.replace('/', '_')}.json"
                if not baseline_file.is_file():
                    self.skipTest(f"no baseline for {model_key!r}")

                baseline = json.loads(baseline_file.read_text())
                predictor = self._predictor(model_key)
                atol = PARITY_ATOL.get(model_key.split("/", 1)[0], DEFAULT_PARITY_ATOL)

                for case_name, expected in baseline["cases"].items():
                    got = predictor.predict(BENCH_OPS[case_name])
                    for got_angle, expected_angle in zip(got, expected):
                        self.assertAlmostEqual(
                            got_angle,
                            expected_angle,
                            delta=atol,
                            msg=f"{model_key}/{case_name}: ONNX drifted from baseline",
                        )


class TestHfManifest(TrainingPipelineTestCase):
    """Offline checks on the manifest that addresses the model zoo."""

    def setUp(self):
        super().setUp()
        self.manifest = model_registry.load_manifest()
        self.bundles = self.manifest["bundles"]

    def test_manifest_covers_expected_zoo(self):
        """Every architecture x depth of the zoo has exactly one manifest entry."""
        expected = {f"{model}/p{p}" for model in MODEL_NAMES for p in P_VALUES}
        self.assertEqual(set(self.bundles), expected)

    def test_repo_ids_are_unique(self):
        """No two bundles point at the same repo (a copy-paste guard)."""
        repo_ids = [entry["repo_id"] for entry in self.bundles.values()]
        self.assertEqual(len(repo_ids), len(set(repo_ids)))

    def test_repo_id_depth_matches_bundle_key(self):
        """The repo name's ``p<p>`` component agrees with the bundle key's depth."""
        for key, entry in self.bundles.items():
            with self.subTest(bundle=key):
                self.assertIn(f".p{_p_of(key)}.", entry["repo_id"])

    def test_revisions_are_commit_shas(self):
        """A published bundle is pinned to an immutable commit, not a branch."""
        for key in model_registry.published_bundles():
            with self.subTest(bundle=key):
                revision = self.bundles[key]["revision"]
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_bundle_entry_rejects_unknown_key(self):
        """An unknown bundle key fails with the available keys listed."""
        with self.assertRaises(KeyError) as ctx:
            model_registry.bundle_entry("no_such_model/p1")
        self.assertIn("no_such_model/p1", str(ctx.exception))

    def test_bundle_entry_rejects_unpublished_bundle(self):
        """An unpinned bundle reports that it is not published, not a 404."""
        unpublished = sorted(set(self.bundles) - set(model_registry.published_bundles()))
        if not unpublished:
            self.skipTest("every bundle in the manifest is published")
        with self.assertRaises(RuntimeError) as ctx:
            model_registry.bundle_entry(unpublished[0])
        self.assertIn("not published yet", str(ctx.exception))


@unittest.skipUnless(HAS_HF_HUB, "huggingface_hub not installed (install the 'inference' extra)")
class TestBundleResolution(TrainingPipelineTestCase):
    """Bundle download plumbing, with the Hub mocked out."""

    def test_resolve_bundle_pins_revision_and_filters_files(self):
        """A zoo bundle downloads at its pinned revision, fetching only its files."""
        published = model_registry.published_bundles()
        if not published:
            self.skipTest("no published bundle in the manifest")
        key = published[0]
        entry = model_registry.bundle_entry(key)

        with mock.patch(
            "huggingface_hub.snapshot_download", return_value="/tmp/snapshot"
        ) as download:
            resolved = model_registry.resolve_bundle(key)

        self.assertEqual(resolved, Path("/tmp/snapshot"))
        download.assert_called_once()
        args, kwargs = download.call_args
        self.assertEqual(args[0], entry["repo_id"])
        self.assertEqual(kwargs["revision"], entry["revision"])
        self.assertEqual(kwargs["repo_type"], "model")
        # Only the three bundle files, so an unrelated file added to a repo
        # (README, license) never enlarges the download.
        self.assertEqual(sorted(kwargs["allow_patterns"]), sorted(model_registry.BUNDLE_FILES))

    def test_download_bundle_defaults_to_main(self):
        """The manifest-free path defaults to the repo's main branch."""
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value="/tmp/snapshot"
        ) as download:
            model_registry.download_bundle("org/some-bundle")
        self.assertEqual(download.call_args.kwargs["revision"], "main")

    def test_prefetch_defaults_to_published_bundles(self):
        """Prefetching without arguments warms exactly the published bundles."""
        with mock.patch.object(
            model_registry, "download_bundle", return_value=Path("/tmp/snapshot")
        ) as download:
            paths = model_registry.prefetch_bundles()
        self.assertEqual(len(paths), len(model_registry.published_bundles()))
        self.assertEqual(download.call_count, len(model_registry.published_bundles()))
