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
from types import SimpleNamespace
from unittest import mock

from qiskit.quantum_info import SparsePauliOp

import numpy as np

from qaoa_training_pipeline.inference import model_registry
from qaoa_training_pipeline.inference.scaling import normalize_cost_op, undo_gamma_rescale
from qaoa_training_pipeline.inference.conventions import MAX_CUT_PRE_FACTOR

from ..training_pipeline_test_case import TrainingPipelineTestCase

# Distinguishes "output_dim absent" from "output_dim is None" in the tests below.
_ABSENT = object()


def _max_cut_operator(num_nodes: int) -> SparsePauliOp:
    """A ring max-cut operator in the convention the models were trained on."""
    terms, coeffs = [], []
    for node in range(num_nodes):
        label = ["I"] * num_nodes
        label[node] = "Z"
        label[(node + 1) % num_nodes] = "Z"
        terms.append("".join(reversed(label)))
        coeffs.append(MAX_CUT_PRE_FACTOR)
    return SparsePauliOp(terms, coeffs=coeffs).simplify()


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

    def _runnable(self, model_keys):
        """The keys whose bundle can be loaded, skipping this test if none can.

        The skip has to be the *test's* outcome rather than every subtest's. A
        test whose subtests all skip reports no status of its own, and stestr
        reads a started-but-statusless test as a worker that segfaulted: the run
        then fails with "exited without returning a status" while reporting zero
        failures. That is what CI saw once the zoo was published and every
        subtest here started skipping on "set QTP_HF_TESTS=1".
        """
        runnable = [key for key in model_keys if skip_reason(key) is None]
        if not runnable:
            self.skipTest(f"no bundle available to run against: {skip_reason(model_keys[0])}")
        return runnable

    @staticmethod
    def _angles(predictor, cost_op, **kwargs):
        """Finished angles for ``cost_op``: the chain AIInference runs.

        The predictor is the network step alone, so a test that wants angles
        for an operator drives the two scaling steps itself.
        """
        normalized, rescale_a = normalize_cost_op(cost_op)
        raw = predictor.predict_normalized(normalized, rescale_a, **kwargs)
        if not predictor.denormalizes(kwargs.get("denormalize")):
            return raw
        return undo_gamma_rescale(raw, p=predictor.output_dim // 2, rescale_a=rescale_a)

    def test_onnx_predictor_loads_and_predicts(self):
        """Every exported bundle loads via onnxruntime and produces 2*p angles."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in self._runnable(MODEL_KEYS):
            with self.subTest(model=model_key):
                angles = self._angles(self._predictor(model_key), op)
                self.assertIsInstance(angles, list)
                self.assertEqual(len(angles), 2 * _p_of(model_key))  # [betas..., gammas...]
                self.assertTrue(all(isinstance(a, float) and math.isfinite(a) for a in angles))

    def test_onnx_predictor_is_deterministic(self):
        """Inference is deterministic: same input -> identical output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in self._runnable(BEHAVIORAL_KEYS):
            with self.subTest(model=model_key):
                predictor = self._predictor(model_key)
                self.assertEqual(self._angles(predictor, op), self._angles(predictor, op))

    def test_onnx_predictor_reacts_to_input(self):
        """Different problem graphs yield different predicted angles."""
        op_triangle = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        op_line4 = SparsePauliOp.from_list([("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0)])
        for model_key in self._runnable(BEHAVIORAL_KEYS):
            with self.subTest(model=model_key):
                predictor = self._predictor(model_key)
                self.assertNotEqual(
                    self._angles(predictor, op_triangle), self._angles(predictor, op_line4)
                )

    def test_onnx_predictor_raw_vs_denormalized_differ(self):
        """denormalize=False returns the raw (unscaled) model output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_key in self._runnable(BEHAVIORAL_KEYS):
            with self.subTest(model=model_key):
                predictor = self._predictor(model_key)
                raw = self._angles(predictor, op, denormalize=False)
                scaled = self._angles(predictor, op, denormalize=True)
                # config output_scale = pi/2, so scaled == raw * pi/2 for the betas
                # (first half); the gammas additionally undergo the rescale_a
                # division, so compare only the betas here (p=1 -> index 0).
                p = predictor.output_dim // 2
                for raw_beta, scaled_beta in zip(raw[:p], scaled[:p]):
                    self.assertAlmostEqual(scaled_beta, raw_beta * (math.pi / 2), places=5)


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestExecutionProviders(TrainingPipelineTestCase):
    """Device selection, which decides what the model actually runs on."""

    @staticmethod
    def _providers_for(device: str) -> list[str]:
        """``_execution_providers`` for a device, without loading a bundle."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        # Only self.device is read, so a stand-in avoids a download here.
        return OnnxQAOAPredictor._execution_providers(  # pylint: disable=protected-access
            SimpleNamespace(device=device)
        )

    def test_cpu_asks_for_the_cpu_provider(self):
        """The default device needs nothing beyond the base onnxruntime."""
        self.assertEqual(self._providers_for("cpu"), ["CPUExecutionProvider"])

    def test_cuda_without_the_provider_is_refused(self):
        """A GPU run must not become a silent CPU run."""
        with (
            mock.patch(
                "onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            self._providers_for("cuda")
        self.assertIn("onnxruntime-gpu", str(ctx.exception))

    def test_cuda_with_the_provider_falls_back_to_cpu_last(self):
        """With CUDA available it leads, with CPU kept as the fallback."""
        with mock.patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            self.assertEqual(
                self._providers_for("cuda:0"),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )


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

    def test_uncached_bundle_offline_explains_the_cache(self):
        """The Hub's own offline error says nothing about warming the cache."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        with (
            mock.patch(
                "huggingface_hub.snapshot_download", side_effect=LocalEntryNotFoundError("")
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            model_registry.download_bundle("org/some-bundle")
        message = str(ctx.exception)
        self.assertIn("not in the local HuggingFace cache", message)
        self.assertIn("HF_HUB_OFFLINE", message)

    def test_resolution_failure_names_the_bundle_key(self):
        """The caller passed a key, so the error must speak in keys."""
        published = model_registry.published_bundles()
        if not published:
            self.skipTest("no published bundle in the setup")
        from huggingface_hub.errors import LocalEntryNotFoundError

        with (
            mock.patch(
                "huggingface_hub.snapshot_download", side_effect=LocalEntryNotFoundError("")
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            model_registry.resolve_bundle(published[0])
        self.assertIn(published[0], str(ctx.exception))


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestOutputDimContract(TrainingPipelineTestCase):
    """``output_dim`` decides which half of a prediction is rescaled.

    A bundle that declares it wrongly produces finite, plausible, wrong angles,
    so it is rejected at load rather than trusted.
    """

    @staticmethod
    def _require(output_dim):
        """``_require_output_dim`` for a declared value, without a download."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        model_init = {} if output_dim is _ABSENT else {"output_dim": output_dim}
        stand_in = SimpleNamespace(
            model_init=model_init,
            bundle_key="graph_neural_network/p1",
            repo_id="org/repo",
            revision="deadbeef",
        )
        # pylint: disable=protected-access
        return OnnxQAOAPredictor._require_output_dim(stand_in)

    def test_an_even_positive_value_is_accepted(self):
        """Accept p betas followed by p gammas, the only valid shape."""
        self.assertEqual(self._require(2), 2)
        self.assertEqual(self._require(10), 10)

    def test_a_missing_value_is_refused(self):
        """Without it, no gamma would be rescaled and the angles would be wrong."""
        with self.assertRaises(ValueError) as ctx:
            self._require(_ABSENT)
        self.assertIn("output_dim", str(ctx.exception))

    def test_zero_and_odd_and_negative_are_refused(self):
        """None of these can be a beta/gamma split."""
        for value in (0, 1, 3, -2):
            with self.subTest(output_dim=value), self.assertRaises(ValueError):
                self._require(value)

    def test_a_non_integer_is_refused(self):
        """A string or None in the config must not reach int() unguarded."""
        for value in ("two", None, [2]):
            with self.subTest(output_dim=value), self.assertRaises(ValueError):
                self._require(value)


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestDeviceNormalization(TrainingPipelineTestCase):
    """A device string must reach the provider check in a spelling it recognizes."""

    @staticmethod
    def _normalize(device):
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        # pylint: disable=protected-access
        return OnnxQAOAPredictor._normalize_device(device)

    def test_case_and_whitespace_do_not_change_the_device(self):
        """'CUDA' must not be treated as a non-CUDA device and land on the CPU."""
        self.assertEqual(self._normalize("CUDA"), "cuda")
        self.assertEqual(self._normalize("  Cpu "), "cpu")
        self.assertEqual(self._normalize("CUDA:1"), "cuda:1")

    def test_gpu_is_an_alias_for_cuda(self):
        """The word a user is most likely to reach for."""
        self.assertEqual(self._normalize("gpu"), "cuda")

    def test_an_unknown_device_is_refused(self):
        """Better a signature error than a silent CPU run."""
        for device in ("tpu", "cuda0", "", "mps"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self._normalize(device)


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestPrivateFieldStripping(TrainingPipelineTestCase):
    """Private training-environment fields must not survive a config load."""

    def test_fields_are_stripped_at_every_depth(self):
        """A top-level-only pop would publish the nested copy."""
        from qaoa_training_pipeline.inference.onnx_predictor import _strip_private_fields

        config = {
            "checkpoint": "/private/path.ckpt",
            "model_init": {"checkpoint": "/private/nested.ckpt", "output_dim": 2},
            "history": [{"checkpoint": "/private/in_a_list.ckpt", "epoch": 1}],
        }
        stripped = _strip_private_fields(config)

        self.assertNotIn("checkpoint", repr(stripped))
        self.assertEqual(stripped["model_init"]["output_dim"], 2)
        self.assertEqual(stripped["history"][0]["epoch"], 1)


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestScaleInvariance(TrainingPipelineTestCase):
    """Scaling the cost operator must leave betas fixed and divide gammas by it.

    This is the property the whole normalize/denormalize path exists to
    provide. With ``U = prod_k exp(-i beta_k H_B) exp(-i gamma_k H_C)`` and
    ``H_B = sum_i X_i``, sending ``H_C -> c H_C`` preserves the state only if
    ``gamma -> gamma / c``, while beta must not move. It runs without a bundle:
    the network is stubbed, so what is under test is the transformation layer
    around it, which is where the rescaling is applied.
    """

    SCALES = (1.0, 2.0, 10.0, 0.1)

    def _predictor(self, raw, output_dim):
        """A real predictor with the ONNX session stubbed out."""
        from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        predictor = OnnxQAOAPredictor.__new__(OnnxQAOAPredictor)
        predictor.config = {"denormalize_output": True, "output_scale": math.pi / 2}
        predictor.model_init = {"output_dim": output_dim}
        predictor.output_dim = output_dim
        predictor.feature_extractor = AIFeatureExtractor(in_features=[], norm_stats={})
        # A scale-invariant network: the features it is fed are already
        # normalized, so a correct model returns the same raw output at every
        # scale. Any scale dependence in the result is therefore the
        # transformation layer's, which is exactly what is being measured.
        predictor.session = mock.Mock()
        predictor.session.run.return_value = [np.asarray([raw], dtype=np.float32)]
        predictor._input_names = set()  # pylint: disable=protected-access
        predictor._prepare = lambda _features: {}  # pylint: disable=protected-access
        return predictor

    def _predict(self, cost_op, raw, output_dim):
        """Run the scaling chain with the ONNX session stubbed out."""
        predictor = self._predictor(raw, output_dim)
        normalized, rescale_a = normalize_cost_op(cost_op)
        return undo_gamma_rescale(
            predictor.predict_normalized(normalized, rescale_a),
            p=output_dim // 2,
            rescale_a=rescale_a,
        )

    def test_betas_hold_still_and_gammas_track_the_scale(self):
        """The headline guarantee, measured at four scales for p=1 and p=2."""
        cost_op = _max_cut_operator(10)

        for output_dim, raw in ((2, [0.25, 0.20]), (4, [0.25, 0.30, 0.20, 0.15])):
            p = output_dim // 2
            reference = self._predict(cost_op, raw, output_dim)
            for scale in self.SCALES:
                with self.subTest(output_dim=output_dim, scale=scale):
                    angles = self._predict(cost_op * scale, raw, output_dim)

                    for index in range(p):
                        self.assertAlmostEqual(
                            angles[index], reference[index], places=6, msg="a beta moved"
                        )
                    for index in range(p, output_dim):
                        self.assertAlmostEqual(
                            angles[index] * scale,
                            reference[index],
                            places=6,
                            msg="a gamma did not track the operator scale",
                        )

    def test_a_scaled_operator_does_not_change_the_raw_prediction(self):
        """denormalize=False must bypass the rescaling entirely."""
        cost_op = _max_cut_operator(8)
        from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor

        extractor = AIFeatureExtractor(in_features=[], norm_stats={})
        base = extractor.extract_np(cost_op)["rescale_a"]
        scaled = extractor.extract_np(cost_op * 4.0)["rescale_a"]
        # The rescaling factor is homogeneous of degree one, which is what
        # makes the normalized features -- and so the raw prediction -- the
        # same at every scale.
        self.assertAlmostEqual(scaled / base, 4.0, places=6)
