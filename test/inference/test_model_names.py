#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Offline tests for model-name verification.

A bundle key names an architecture and a QAOA depth. An unverified name must
hard-fail rather than resolve to some other repo or to nothing, so these tests
pin the verified table, the parser, and the setup check that uses it. They
need no network and no onnxruntime.
"""

import importlib.util
import inspect
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from qaoa_training_pipeline.inference import ai_inference
from qaoa_training_pipeline.inference.model_registry import (
    P_VALUES,
    TRACKED_SETUP,
    VERIFIED_ARCHITECTURES,
    available_bundles,
    bundle_entry,
    expected_model_type,
    expected_repo_suffix,
    load_setup,
    published_bundles,
    setup_path,
    parse_bundle_key,
    validate_setup,
)


# The predictor imports onnxruntime, an optional dependency (the inference extra).
if importlib.util.find_spec("onnxruntime") is not None:
    from qaoa_training_pipeline.inference import onnx_predictor
else:  # pragma: no cover - exercised only without the inference extra
    onnx_predictor = None  # pylint: disable=invalid-name


class TestBundleKeyVerification(unittest.TestCase):
    """parse_bundle_key accepts exactly the verified names."""

    def test_every_verified_name_parses(self):
        """All architecture/depth combinations of the table are accepted."""
        for architecture in VERIFIED_ARCHITECTURES:
            for p in P_VALUES:
                with self.subTest(key=f"{architecture}/p{p}"):
                    self.assertEqual(parse_bundle_key(f"{architecture}/p{p}"), (architecture, p))

    def test_unverified_architecture_is_rejected(self):
        """An unknown architecture fails and names the verified set."""
        with self.assertRaises(ValueError) as ctx:
            parse_bundle_key("transformer_xl/p1")
        self.assertIn("Unverified model architecture", str(ctx.exception))
        self.assertIn("graph_neural_network", str(ctx.exception))

    def test_hub_short_name_suggests_the_bundle_key(self):
        """'gnn/p1' is the Hub's name for the model; the error says so."""
        with self.assertRaises(ValueError) as ctx:
            parse_bundle_key("gnn/p1")
        self.assertIn("graph_neural_network/p1", str(ctx.exception))

    def test_unreleased_depth_is_rejected(self):
        """A depth outside p1..p5 fails."""
        for key in ("gcn/p0", "gcn/p6", "gcn/p10"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as ctx:
                    parse_bundle_key(key)
                self.assertIn("Unverified QAOA depth", str(ctx.exception))

    def test_malformed_keys_are_rejected(self):
        """A key that is not '<architecture>/p<p>' fails."""
        for key in ("", "gcn", "/p1", "gcn/", "gcn p1"):
            with self.subTest(key=key):
                self.assertRaises(ValueError, parse_bundle_key, key)

    def test_expected_model_type_matches_the_table(self):
        """The declared model_type is the table's, not the architecture name."""
        self.assertEqual(expected_model_type("mlp/p3"), "agg_transformer")
        self.assertEqual(expected_model_type("graph_neural_network/p1"), "graph_neural_network")

    def test_expected_repo_suffix_uses_the_hub_short_name(self):
        """Repo ids end in p<p>.<short name>, which differs for two models."""
        self.assertEqual(expected_repo_suffix("graph_isomorphism_network/p2"), "p2.gin")
        self.assertEqual(expected_repo_suffix("graph_neural_network/p1"), "p1.gnn")
        self.assertEqual(expected_repo_suffix("gcn/p5"), "p5.gcn")

    def test_bundle_entry_rejects_an_unverified_name(self):
        """The lookup validates the name before consulting the setup."""
        with self.assertRaises(ValueError):
            bundle_entry("not_a_model/p1")


class TestSingleEntryPoint(unittest.TestCase):
    """A bundle key is the only way to address a model."""

    def test_raw_repo_id_is_not_an_address(self):
        """A HuggingFace repo id passed as a model key is refused."""
        with self.assertRaises(ValueError) as ctx:
            parse_bundle_key("ibm-research/qaoa_angles.max_cut.p1.gnn")
        self.assertIn("Unverified model architecture", str(ctx.exception))

    @unittest.skipIf(onnx_predictor is None, "onnxruntime not installed")
    def test_predictor_has_no_second_entry_point(self):
        """from_bundle is the only public loader on the predictor."""
        loaders = [
            name for name in vars(onnx_predictor.OnnxQAOAPredictor) if name.startswith("from_")
        ]
        self.assertEqual(loaders, ["from_bundle"])

    def test_ai_inference_takes_only_a_bundle_key(self):
        """AIInference exposes no repo_id/revision arguments to bypass the setup."""
        params = inspect.signature(ai_inference.AIInference.__init__).parameters
        self.assertIn("model", params)
        self.assertNotIn("repo_id", params)
        self.assertNotIn("revision", params)
        self.assertNotIn("config_path", params)


class TestSetupVerification(unittest.TestCase):
    """The shipped setups only ever list verified names."""

    def test_tracked_setup_is_valid(self):
        """The setup that ships with the package passes validation."""
        validate_setup(json.loads(TRACKED_SETUP.read_text(encoding="utf-8")), TRACKED_SETUP)

    def test_setup_defaults_to_the_packaged_one(self):
        """Without $QAOA_HF_SETUP, resolution uses the packaged setup only."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(setup_path(), TRACKED_SETUP)

    def test_env_override_selects_another_setup(self):
        """$QAOA_HF_SETUP is the one way to use a setup of your own."""
        with mock.patch.dict(os.environ, {"QAOA_HF_SETUP": str(TRACKED_SETUP)}):
            self.assertEqual(setup_path(), TRACKED_SETUP)
        with mock.patch.dict(os.environ, {"QAOA_HF_SETUP": "/nope/missing.json"}):
            self.assertRaises(FileNotFoundError, setup_path)

    def test_setup_lists_every_verified_bundle(self):
        """All architectures x depths are present, and nothing else."""
        expected = {f"{a}/p{p}" for a in VERIFIED_ARCHITECTURES for p in P_VALUES}
        self.assertEqual(set(available_bundles()), expected)

    def test_repo_id_disagreeing_with_the_key_is_rejected(self):
        """A key pointing at another model's repo is a hard failure."""
        setup = {
            "bundles": {
                "gcn/p1": {"repo_id": "ibm-research/qaoa_angles.max_cut.p1.gin"},
            }
        }
        with self.assertRaises(ValueError) as ctx:
            validate_setup(setup, Path("test.json"))
        self.assertIn("p1.gcn", str(ctx.exception))

    def test_unverified_key_in_a_setup_is_rejected(self):
        """An unverified name in the setup fails at load, not at use."""
        setup = {"bundles": {"gnn/p1": {"repo_id": "ibm-research/x.p1.gnn"}}}
        self.assertRaises(ValueError, validate_setup, setup, Path("test.json"))

    def test_empty_setup_is_rejected(self):
        """A setup with no bundles is invalid rather than silently empty."""
        self.assertRaises(ValueError, validate_setup, {"bundles": {}}, Path("test.json"))

    def test_repo_ids_are_unique(self):
        """No two bundles may point at the same repo (a copy-paste guard)."""
        repo_ids = [entry["repo_id"] for entry in load_setup()["bundles"].values()]
        self.assertEqual(len(repo_ids), len(set(repo_ids)))

    def test_revisions_are_commit_shas(self):
        """A published bundle is pinned to an immutable commit, not a branch."""
        bundles = load_setup()["bundles"]
        for key in published_bundles():
            with self.subTest(bundle=key):
                self.assertRegex(bundles[key]["revision"], r"^[0-9a-f]{40}$")

    def test_unpublished_bundle_is_reported_as_such(self):
        """An unpinned bundle says it is not published, rather than 404-ing."""
        unpublished = sorted(set(available_bundles()) - set(published_bundles()))
        if not unpublished:
            self.skipTest("every bundle in the setup is published")
        with self.assertRaises(RuntimeError) as ctx:
            bundle_entry(unpublished[0])
        self.assertIn("not published yet", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
