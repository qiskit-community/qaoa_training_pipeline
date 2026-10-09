#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for AIInference, the provider that fronts the ONNX model zoo.

Most of these run offline against a stub predictor: AIInference's own job is
argument validation, the rescale hook, config round-tripping and the result
payload, none of which need a real graph. The one test that does run a model is
gated on an available bundle.
"""

import os
import unittest
from unittest import mock

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.ai_inference import AIInference
from qaoa_training_pipeline.inference.model_registry import published_bundles
from qaoa_training_pipeline.training.functions import IdentityFunction

from ..training_pipeline_test_case import TrainingPipelineTestCase

HF_TESTS_ENABLED = os.environ.get("QTP_HF_TESTS") == "1"

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

    def predict(self, cost_op, **kwargs):
        """Record how the provider called us and return fixed angles."""
        self.predict_calls.append({"cost_op": cost_op, **kwargs})
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
        with mock.patch.object(AIInference, "load_model") as load_model:
            with self.assertRaises(ValueError) as ctx:
                AIInference(model="not_a_model/p1")
        self.assertIn("Unverified model architecture", str(ctx.exception))
        load_model.assert_not_called()

    def test_hub_short_name_gets_a_hint(self):
        """'gnn/p1' is the repo's name, not the key; say which key to use."""
        with mock.patch.object(AIInference, "load_model"):
            with self.assertRaises(ValueError) as ctx:
                AIInference(model="gnn/p1")
        self.assertIn("graph_neural_network/p1", str(ctx.exception))

    def test_unreleased_depth_is_rejected(self):
        """Depths outside the released set must not resolve."""
        with mock.patch.object(AIInference, "load_model"):
            with self.assertRaises(ValueError) as ctx:
                AIInference(model="gcn/p9")
        self.assertIn("Unverified QAOA depth", str(ctx.exception))

    def test_model_is_required(self):
        """There is no default model: omitting it is a signature error."""
        with self.assertRaises(TypeError):
            AIInference()  # pylint: disable=no-value-for-parameter

    def test_no_second_addressing_argument(self):
        """A raw repo id or local path must not be reachable through __init__."""
        with mock.patch.object(AIInference, "load_model"):
            for argument in ("repo_id", "revision", "config_path", "model_path"):
                with self.subTest(argument=argument):
                    with self.assertRaises(TypeError):
                        AIInference(model=STUB_KEY, **{argument: "anything"})

    def test_load_model_passes_the_key_to_the_predictor(self):
        """load_model is a thin delegation to the bundle loader."""
        with mock.patch(
            "qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.from_bundle"
        ) as from_bundle:
            provider = AIInference(model=STUB_KEY, device="cpu", strict=False)
        from_bundle.assert_called_once_with(STUB_KEY, device="cpu", strict=False)
        self.assertIs(provider.model, from_bundle.return_value)


class TestConfigRoundTrip(TrainingPipelineTestCase):
    """The provider must survive to_config -> from_config on another machine."""

    def test_from_config_without_a_model_lists_the_options(self):
        """A config missing 'model' is a user error worth a helpful message."""
        with self.assertRaises(ValueError) as ctx:
            AIInference.from_config({"device": "cpu"})
        message = str(ctx.exception)
        self.assertIn("requires 'model'", message)
        for key in published_bundles():
            self.assertIn(key, message)

    def test_from_config_passes_options_through(self):
        """Every constructor option must be reachable from a config."""
        stub = StubPredictor()

        def fake_load(self):
            self.model = stub

        config = {
            "model": STUB_KEY,
            "device": "cuda",
            "strict": False,
            "validate_input_operator": False,
            "qaoa_angles_function": "IdentityFunction",
        }
        with mock.patch.object(AIInference, "load_model", fake_load):
            provider = AIInference.from_config(config)

        self.assertEqual(provider.model_key, STUB_KEY)
        self.assertEqual(provider.device, "cuda")
        self.assertFalse(provider.strict)
        self.assertFalse(provider.validate_input_operator)
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
        provider, stub = _provider(device="cpu", strict=True)

        def fake_load(self):
            self.model = stub

        with mock.patch.object(AIInference, "load_model", fake_load):
            rebuilt = AIInference.from_config(provider.to_config())

        self.assertEqual(rebuilt.model_key, provider.model_key)
        self.assertEqual(rebuilt.device, provider.device)
        self.assertEqual(rebuilt.strict, provider.strict)


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

    def test_denormalization_is_left_to_the_config_by_default(self):
        """With no hook the predictor applies its own rescaling."""
        provider, stub = _provider()
        provider.provide_params(self.cost_op)
        self.assertIsNone(stub.predict_calls[0]["denormalize"])

    def test_rescale_hook_replaces_denormalization(self):
        """A hook must see the raw output, not the already-rescaled angles."""
        provider, stub = _provider(
            StubPredictor(angles=(1.0, 2.0)), rescale=lambda a: [x * 3 for x in a]
        )
        result = provider.provide_params(self.cost_op)
        self.assertFalse(stub.predict_calls[0]["denormalize"])
        self.assertEqual(list(result["optimized_params"]), [3.0, 6.0])

    def test_rescale_hook_may_not_change_the_angle_count(self):
        """A hook that drops angles would silently corrupt the ansatz."""
        provider, _ = _provider(rescale=lambda a: list(a)[:1])
        with self.assertRaises(ValueError) as ctx:
            provider.provide_params(self.cost_op)
        self.assertIn("changed the number of angles", str(ctx.exception))

    def test_angle_count_is_checked_against_the_config(self):
        """A model emitting the wrong number of angles must not pass silently."""
        provider, _ = _provider(StubPredictor(angles=(0.1, 0.2), output_dim=4))
        with self.assertRaises(ValueError) as ctx:
            provider.provide_params(self.cost_op)
        self.assertIn("output_dim=4", str(ctx.exception))

    def test_angle_count_check_can_be_disabled(self):
        """validate_input_operator=False opts out of the cross-check."""
        provider, _ = _provider(
            StubPredictor(angles=(0.1, 0.2), output_dim=4), validate_input_operator=False
        )
        result = provider.provide_params(self.cost_op)
        self.assertEqual(len(result["optimized_params"]), 2)

    def test_unloaded_model_is_reported(self):
        """A provider without a model must fail loudly, not return garbage."""
        provider, _ = _provider()
        provider.model = None
        with self.assertRaises(RuntimeError):
            provider.provide_params(self.cost_op)


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
