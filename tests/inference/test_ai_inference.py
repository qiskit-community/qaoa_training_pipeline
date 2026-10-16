#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for AIInference, the provider that fronts the ONNX model zoo.

Most of these run offline against a stub predictor: AIInference's own job is
argument validation, config round-tripping and the result
payload, none of which need a real graph. The one test that does run a model is
gated on an available bundle.
"""

import argparse
import importlib.util
import os
import unittest
from unittest import mock

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.ai_inference import AIInference
from qaoa_training_pipeline.inference.model_registry import (
    available_bundles,
    published_bundles,
)
from qaoa_training_pipeline.training.functions import FourierFunction, IdentityFunction

from ..training_pipeline_test_case import TrainingPipelineTestCase

HF_TESTS_ENABLED = os.environ.get("QTP_HF_TESTS") == "1"

# The provider itself is importable without the inference extra; only the ONNX
# predictor behind it is not. A test that reaches into that module has to say so.
HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None

# A key that parses but is deliberately not the one the stub reports, so a test
# can tell "what was asked for" from "what was loaded".
STUB_KEY = "graph_neural_network/p1"


class StubPredictor:
    """Stands in for OnnxQAOAPredictor: the surface AIInference actually uses."""

    def __init__(self, bundle_key=STUB_KEY, angles=(0.1, 0.2), output_dim=2):
        self.bundle_key = bundle_key
        self.angles = list(angles)
        self.output_dim = output_dim
        self.predict_calls = []

    def predict_normalized(self, cost_op, rescale_a, **kwargs):
        """What the provider calls: the network step, with the scale handed in."""
        self.predict_calls.append({"cost_op": cost_op, "rescale_a": rescale_a, **kwargs})
        return list(self.angles)

    def metadata(self):
        """Mimic the predictor's metadata payload."""
        return {"model_type": "graph_neural_network", "output_dim": self.output_dim}

    def source(self):
        """The bundle key is the only address a config may carry."""
        return {"model": self.bundle_key}


def _provider(stub=None, **kwargs):
    """Build an AIInference whose model is a stub, without touching the Hub."""
    stub = stub or StubPredictor()

    def fake_load(self):
        self.model = stub

    with mock.patch.object(AIInference, "load_model", fake_load):
        provider = AIInference(model=kwargs.pop("model", STUB_KEY), **kwargs)
    return provider, stub


def _ring(n):
    """A uniformly weighted ring cost operator on ``n`` nodes."""
    terms = []
    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        label[(i + 1) % n] = "Z"
        terms.append("".join(reversed(label)))
    return SparsePauliOp(terms, coeffs=[1.0] * n).simplify()


class TestModelAddressing(TrainingPipelineTestCase):
    """A model is addressed by bundle key, and only by a verified one."""

    def test_unverified_key_is_rejected_before_loading(self):
        """The name is checked at construction, not at download time."""
        with (
            mock.patch.object(AIInference, "load_model") as load_model,
            self.assertRaises(ValueError) as ctx,
        ):
            AIInference(model="not_a_model/p1")
        self.assertIn("Unverified model architecture", str(ctx.exception))
        load_model.assert_not_called()

    def test_hub_short_name_gets_a_hint(self):
        """'gnn/p1' is the repo's name, not the key; say which key to use."""
        with mock.patch.object(AIInference, "load_model"), self.assertRaises(ValueError) as ctx:
            AIInference(model="gnn/p1")
        self.assertIn("graph_neural_network/p1", str(ctx.exception))

    def test_unreleased_depth_is_rejected(self):
        """Depths outside the released set must not resolve."""
        with mock.patch.object(AIInference, "load_model"), self.assertRaises(ValueError) as ctx:
            AIInference(model="gcn/p9")
        self.assertIn("Unverified QAOA depth", str(ctx.exception))

    def test_model_is_required(self):
        """There is no default model: omitting it is a signature error."""
        with self.assertRaises(TypeError):
            AIInference()  # pylint: disable=no-value-for-parameter

    def test_no_second_addressing_argument(self):
        """A raw repo id or local path must not be reachable through __init__."""
        with mock.patch.object(AIInference, "load_model"):
            for argument in ("repo_id", "revision", "config_path", "model_path", "strict"):
                with self.subTest(argument=argument), self.assertRaises(TypeError):
                    AIInference(model=STUB_KEY, **{argument: "anything"})

    @unittest.skipUnless(
        HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)"
    )
    def test_load_model_passes_the_key_to_the_predictor(self):
        """load_model is a thin delegation to the bundle loader.

        Patching the predictor imports it, which needs onnxruntime -- hence the
        guard, unlike the rest of this class.
        """
        with mock.patch(
            "qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.from_bundle"
        ) as from_bundle:
            provider = AIInference(model=STUB_KEY, device="cpu")
        from_bundle.assert_called_once_with(STUB_KEY, device="cpu")
        self.assertIs(provider.model, from_bundle.return_value)


class TestConfigRoundTrip(TrainingPipelineTestCase):
    """The provider must survive to_config -> from_config on another machine."""

    def test_from_config_without_a_model_lists_the_options(self):
        """A config missing 'model' is a user error worth a helpful message."""
        with self.assertRaises(ValueError) as ctx:
            AIInference.from_config({"device": "cpu"})
        message = str(ctx.exception)
        self.assertIn("requires 'model'", message)
        keys = available_bundles()
        self.assertTrue(keys, "the setup file must list bundles for this test to mean anything")
        for key in keys:
            self.assertIn(key, message)

    def test_from_config_passes_options_through(self):
        """Every constructor option must be reachable from a config."""
        stub = StubPredictor()

        def fake_load(self):
            self.model = stub

        config = {
            "model": STUB_KEY,
            "device": "cuda",
            "qaoa_angles_function": "IdentityFunction",
        }
        with mock.patch.object(AIInference, "load_model", fake_load):
            provider = AIInference.from_config(config)

        self.assertEqual(provider.model_key, STUB_KEY)
        self.assertEqual(provider.device, "cuda")
        self.assertIsInstance(provider.qaoa_angles_function, IdentityFunction)

    def test_to_config_carries_the_bundle_key(self):
        """The config addresses the model the one supported way."""
        provider, _ = _provider()
        config = provider.to_config()
        self.assertEqual(config["model"], STUB_KEY)

    def test_to_config_leaks_no_local_path_or_checkpoint(self):
        """A cache path or a private checkpoint must never reach a config."""
        provider, _ = _provider()
        serialized = repr(provider.to_config())
        for leak in ("checkpoint", ".cache", "/Users", "huggingface"):
            self.assertNotIn(leak, serialized)

    def test_config_round_trips(self):
        """from_config(to_config()) must rebuild an equivalent provider."""
        provider, stub = _provider(device="cpu")

        def fake_load(self):
            self.model = stub

        with mock.patch.object(AIInference, "load_model", fake_load):
            rebuilt = AIInference.from_config(provider.to_config())

        self.assertEqual(rebuilt.model_key, provider.model_key)
        self.assertEqual(rebuilt.device, provider.device)
        self.assertIsInstance(rebuilt.qaoa_angles_function, IdentityFunction)

    def test_to_config_carries_the_angles_function_init(self):
        """An angles function's own arguments must survive the round-trip."""
        provider, stub = _provider(qaoa_angles_function=FourierFunction(depth=2))

        config = provider.to_config()
        self.assertEqual(config["qaoa_angles_function"], "FourierFunction")
        self.assertEqual(config["qaoa_angles_function_init"], {"depth": 2})

        def fake_load(self):
            self.model = stub

        with mock.patch.object(AIInference, "load_model", fake_load):
            rebuilt = AIInference.from_config(config)
        # pylint: disable=protected-access
        self.assertEqual(rebuilt.qaoa_angles_function._depth, 2)


class TestProvideParams(TrainingPipelineTestCase):
    """provide_params wraps the prediction in the framework's result type."""

    def setUp(self):
        super().setUp()
        self.cost_op = _ring(5)

    def test_returns_the_predicted_angles(self):
        """The provider passes the model's angles through unchanged."""
        provider, _ = _provider(StubPredictor(angles=(0.3, 0.4)))
        result = provider.provide_params(self.cost_op)
        self.assertEqual(list(result["optimized_params"]), [0.3, 0.4])

    def test_result_records_the_model_source(self):
        """The result must say which bundle produced the angles."""
        provider, _ = _provider()
        payload = provider.provide_params(self.cost_op)["ai_inference"]
        self.assertEqual(payload["model"], STUB_KEY)
        self.assertEqual(payload["device"], provider.device)
        self.assertIn("predictor_metadata", payload)

    def test_denormalization_is_always_left_to_the_config(self):
        """The provider must never override the config's own output scaling.

        It passes no ``denormalize`` argument at all: forcing it either way
        would overrule what the bundle declares, and suppressing it would hand
        the un-scaling step raw network output.
        """
        provider, stub = _provider()
        provider.provide_params(self.cost_op)
        self.assertNotIn("denormalize", stub.predict_calls[0])

    def test_unloaded_model_is_reported(self):
        """A provider without a model must fail loudly, not return garbage."""
        provider, _ = _provider()
        provider.model = None
        with self.assertRaises(RuntimeError):
            provider.provide_params(self.cost_op)


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestProviderScaleInvariance(TrainingPipelineTestCase):
    """Scaling the operator must leave betas fixed and divide gammas by it.

    ``TestScaleInvariance`` in ``test_onnx_inference.py`` measures this on the
    predictor. This measures it where a caller meets it: through
    ``provide_params``, which runs the two scaling steps around the predictor. The network is stubbed -- a scale-
    invariant model returns the same raw output at every scale -- so what is
    under test is the transformation layer, not the weights.
    """

    SCALES = (1.0, 2.0, 10.0, 0.1)

    @staticmethod
    def _max_cut_ring(n):
        """A ring in the training convention: negative quadratic coefficients."""
        return _ring(n) * -0.5

    def _provider_on_a_stubbed_network(self, raw, output_dim):
        """An AIInference whose model is a real predictor with a mock session."""
        from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        predictor = OnnxQAOAPredictor.__new__(OnnxQAOAPredictor)
        predictor.config = {"denormalize_output": True, "output_scale": np.pi / 2}
        predictor.model_init = {"output_dim": output_dim}
        predictor.output_dim = output_dim
        predictor.bundle_key = STUB_KEY
        predictor.feature_extractor = AIFeatureExtractor(in_features=[], norm_stats={})
        predictor.session = mock.Mock()
        predictor.session.run.return_value = [np.asarray([raw], dtype=np.float32)]
        # pylint: disable=protected-access
        predictor._input_names = set()
        predictor._prepare = lambda _features: {}

        def fake_load(self):
            self.model = predictor

        with mock.patch.object(AIInference, "load_model", fake_load):
            return AIInference(model=STUB_KEY)

    def test_betas_hold_still_and_gammas_track_the_scale(self):
        """The guarantee documented on AIInference, measured through it."""
        cost_op = self._max_cut_ring(10)

        for output_dim, raw in ((2, [0.25, 0.20]), (4, [0.25, 0.30, 0.20, 0.15])):
            depth = output_dim // 2
            provider = self._provider_on_a_stubbed_network(raw, output_dim)
            reference = list(provider.provide_params(cost_op)["optimized_params"])

            for scale in self.SCALES:
                with self.subTest(output_dim=output_dim, scale=scale):
                    angles = list(provider.provide_params(cost_op * scale)["optimized_params"])

                    for index in range(depth):
                        self.assertAlmostEqual(
                            angles[index], reference[index], places=6, msg="a beta moved"
                        )
                    for index in range(depth, output_dim):
                        self.assertAlmostEqual(
                            angles[index] * scale,
                            reference[index],
                            places=6,
                            msg="a gamma did not track the operator scale",
                        )


class TestOptionalDependency(TrainingPipelineTestCase):
    """Without the inference extra, constructing a provider must say so.

    The provider imports without onnxruntime or huggingface_hub -- the
    framework imports it, so it must stay importable on a core install -- and
    the missing dependency is reported when someone actually builds one.
    """

    def test_construction_without_the_extra_names_the_extra(self):
        """Patched rather than uninstalled: the flag is what the code branches on."""
        from qaoa_training_pipeline.inference import ai_inference

        with (
            mock.patch.object(ai_inference, "HAS_INFERENCE", False),
            self.assertRaises(ImportError) as ctx,
        ):
            AIInference(model=STUB_KEY)

        message = str(ctx.exception)
        self.assertIn("AIInference", message)
        self.assertIn("qaoa_training_pipeline[inference]", message)

    def test_the_flag_matches_the_installed_environment(self):
        """HAS_INFERENCE must track what is actually importable."""
        from qaoa_training_pipeline.inference.ai_inference import HAS_INFERENCE

        expected = all(
            importlib.util.find_spec(module) is not None
            for module in ("onnxruntime", "huggingface_hub")
        )
        self.assertEqual(HAS_INFERENCE, expected)


class TestPipelineRegistration(TrainingPipelineTestCase):
    """AIInference must be reachable from a pipeline config, like any provider."""

    def test_registered_as_a_problem_params_provider(self):
        """Being in this registry is what makes the pipeline pass the cost operator."""
        from qaoa_training_pipeline.framework.pipeline import PROBLEM_PROVIDERS

        self.assertIs(PROBLEM_PROVIDERS.get("AIInference"), AIInference)

    def test_a_config_naming_it_builds_a_provider(self):
        """The name in a pipeline config resolves without touching the Hub."""
        from qaoa_training_pipeline.framework.pipeline import Pipeline

        config = {
            "params_provider": {
                "provider_name": "AIInference",
                "provider_init": {"model": STUB_KEY},
            }
        }
        # The bundle itself is beside the point here: what is under test is
        # that the name resolves to this class and gets its init kwargs.
        with mock.patch.object(AIInference, "load_model", lambda _self: None):
            pipeline, _, _ = Pipeline.from_config(config, argparse.Namespace())
        # pylint: disable=protected-access
        self.assertIsInstance(pipeline._params_provider, AIInference)


@unittest.skipUnless(HF_TESTS_ENABLED, "set QTP_HF_TESTS=1 to run bundle-backed tests")
class TestAgainstARealBundle(TrainingPipelineTestCase):
    """End-to-end over a downloaded bundle: the path a user actually takes."""

    def test_predicts_finite_angles(self):
        """The full provider path must produce usable angles for every bundle."""
        keys = published_bundles()
        if not keys:
            self.skipTest("no published bundles in the setup file")

        for bundle_key in keys:
            with self.subTest(bundle_key=bundle_key):
                provider = AIInference(model=bundle_key)
                result = provider.provide_params(_ring(8))
                angles = list(result["optimized_params"])
                depth = int(bundle_key.rsplit("/p", 1)[1])
                self.assertEqual(len(angles), 2 * depth)
                self.assertTrue(np.all(np.isfinite(angles)))
                self.assertEqual(result["ai_inference"]["model"], bundle_key)


if __name__ == "__main__":
    unittest.main()
