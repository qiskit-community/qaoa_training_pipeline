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
  * Offline tests exercise the bundle-resolution logic with a mocked
    ``snapshot_download``. They always run. (The setup file itself is checked
    in test_model_names.py.)
  * Model-running tests need an actual bundle. Downloading is opt-in: set
    ``QTP_HF_TESTS=1``, or they run automatically against bundles already in the
    local HF cache. While the repos are private a token is needed too
    (``hf auth login`` / ``HF_TOKEN``).

``onnxruntime`` is an optional dependency (the ``inference`` extra), so the whole
module skips if it is not importable.
"""

import importlib.util
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

# The registry's verified table is the single source of truth for which
# architectures and depths exist; the setup file maps each key to a repo.
# See test_model_names.py for the checks on that table and the setup file.
MODEL_NAMES = sorted(model_registry.VERIFIED_ARCHITECTURES)
P_VALUES = list(model_registry.P_VALUES)
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

    entry = model_registry.load_setup()["bundles"].get(model_key) or {}
    if not entry.get("revision"):
        return False
    cached = try_to_load_from_cache(
        entry["repo_id"], "model_config.json", revision=entry["revision"]
    )
    return isinstance(cached, str)


def skip_reason(model_key):
    """Why a model-running subtest cannot run, or None if it can."""
    entry = model_registry.load_setup()["bundles"].get(model_key) or {}
    if not entry.get("revision"):
        return f"bundle {model_key!r} is not published on the Hub yet"
    if not (HF_TESTS_ENABLED or bundle_is_cached(model_key)):
        return f"set QTP_HF_TESTS=1 to download {model_key!r} from the Hub"
    return None


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


@unittest.skipUnless(HAS_HF_HUB, "huggingface_hub not installed (install the 'inference' extra)")
class TestBundleResolution(TrainingPipelineTestCase):
    """Bundle download plumbing, with the Hub mocked out."""

    def test_resolve_bundle_pins_revision_and_filters_files(self):
        """A zoo bundle downloads at its pinned revision, fetching only its files."""
        published = model_registry.published_bundles()
        if not published:
            self.skipTest("no published bundle in the setup")
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
        """The download step defaults to main when a setup pins no revision."""
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
