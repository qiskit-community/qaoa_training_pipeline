#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for the torch-free (numpy) path of AIFeatureExtractor.

These need neither torch nor a checkpoint. The numpy path (``extract_np`` /
``pack_features_np`` / ``extract_and_pack_np``) is what the default ONNX
inference backend uses, so it is the one exercised here.

The operators below use negative ``ZZ`` coefficients because that is what the
max-cut convention the models were trained on produces for positive edge
weights (see each bundle's declared ``prefactor``).
"""

import warnings

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.feature_extractor import (
    SCALAR_FEATURES,
    OUT_OF_DISTRIBUTION_SIGMA,
    AIFeatureExtractor,
)

from ..training_pipeline_test_case import TrainingPipelineTestCase

# The convention the shipped bundles declare; tests state it explicitly
# because the package keeps no default.
MAX_CUT = -0.5


# Identity-ish norm stats (mean 0, std 1) so normalized == raw for easy checks.
IDENTITY_STATS = {name: {"mean": 0.0, "std": 1.0} for name in sorted(SCALAR_FEATURES)}


def make_extractor(norm_stats=None):
    """Build an extractor over all scalar features."""
    return AIFeatureExtractor(
        in_features=sorted(SCALAR_FEATURES),
        norm_stats=norm_stats or IDENTITY_STATS,
        prefactor=MAX_CUT,
    )


class TestAIFeatureExtractor(TrainingPipelineTestCase):
    """Test the numpy feature-extraction path used by the ONNX backend."""

    def test_missing_norm_stats_raises(self):
        """A scalar feature without norm stats must fail fast at construction."""
        with self.assertRaises(KeyError):
            AIFeatureExtractor(
                in_features=sorted(SCALAR_FEATURES),
                norm_stats={"num_nodes": {"mean": 0.0, "std": 1.0}},
                prefactor=MAX_CUT,
            )

    def test_extract_triangle_graph_topology(self):
        """A 3-qubit triangle operator maps to a 3-node, 3-edge graph."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        feats = make_extractor().extract_np(op)
        self.assertEqual(feats["num_nodes"], 3)
        self.assertEqual(feats["num_edges"], 3)
        self.assertAlmostEqual(feats["edges_per_node"], 1.0)
        # every node in a triangle has degree 2
        self.assertAlmostEqual(feats["mean_degree"], 2.0)
        self.assertAlmostEqual(feats["std_degree"], 0.0)

    def test_extract_line_graph_topology(self):
        """A 4-qubit line operator maps to a 4-node, 3-edge path graph."""
        op = SparsePauliOp.from_list([("ZZII", -1.0), ("IZZI", -1.0), ("IIZZ", -1.0)])
        feats = make_extractor().extract_np(op)
        self.assertEqual(feats["num_nodes"], 4)
        self.assertEqual(feats["num_edges"], 3)
        # path degrees are [1, 2, 2, 1] -> mean 1.5
        self.assertAlmostEqual(feats["mean_degree"], 1.5)
        self.assertGreater(feats["std_degree"], 0.0)

    def test_pack_features_shape_and_order(self):
        """pack_features_np returns a (1, n_features) array in sorted-name order."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        ext = make_extractor()
        feats = ext.extract_np(op)
        x = ext.pack_features_np(feats)
        self.assertEqual(x.shape, (1, len(SCALAR_FEATURES)))
        self.assertEqual(x.dtype, np.float32)

        # With identity stats, packed values equal the raw features in sorted order.
        expected = [float(feats[name]) for name in sorted(SCALAR_FEATURES)]
        np.testing.assert_allclose(x.squeeze(0), expected, rtol=1e-6)

    def test_pack_features_applies_normalization(self):
        """(v - mean) / std is applied per feature."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        stats = {name: {"mean": 1.0, "std": 2.0} for name in SCALAR_FEATURES}
        ext = AIFeatureExtractor(
            in_features=sorted(SCALAR_FEATURES), norm_stats=stats, prefactor=MAX_CUT
        )
        feats = ext.extract_np(op)
        x = ext.pack_features_np(feats).squeeze(0)
        expected = [(float(feats[name]) - 1.0) / 2.0 for name in sorted(SCALAR_FEATURES)]
        np.testing.assert_allclose(x, expected, rtol=1e-6)

    def test_pack_features_zero_std_is_safe(self):
        """A zero std must not produce inf/nan (guarded to std=1)."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        stats = {name: {"mean": 0.0, "std": 0.0} for name in SCALAR_FEATURES}
        ext = AIFeatureExtractor(
            in_features=sorted(SCALAR_FEATURES), norm_stats=stats, prefactor=MAX_CUT
        )
        x = ext.pack_features_np(ext.extract_np(op))
        self.assertTrue(np.isfinite(x).all())

    def test_pack_features_missing_feature_raises(self):
        """Packing a feature dict missing a requested feature must raise."""
        ext = make_extractor()
        with self.assertRaises(KeyError):
            ext.pack_features_np({"num_nodes": 3})  # missing the rest

    def test_extract_and_pack_consistency(self):
        """extract_and_pack_np equals extract_np followed by pack_features_np."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        ext = make_extractor()
        x_vec, feats = ext.extract_and_pack_np(op)
        self.assertEqual(x_vec.shape, (1, len(SCALAR_FEATURES)))
        np.testing.assert_allclose(
            x_vec.squeeze(0),
            ext.pack_features_np(ext.extract_np(op)).squeeze(0),
            rtol=1e-6,
        )
        # graph-structure arrays are present for graph models
        self.assertIn("edges", feats)
        self.assertIn("node_count", feats)
        self.assertEqual(int(feats["node_count"][0]), 3)

        # edges array: batched (1, num_edges, 2), non-empty, and matching num_edges
        edges = feats["edges"]
        self.assertEqual(edges.dtype, np.int64)
        self.assertEqual(edges.ndim, 3)
        self.assertEqual(edges.shape[0], 1)  # batch dim
        self.assertEqual(edges.shape[2], 2)  # (src, dst) pairs
        self.assertEqual(edges.shape[1], feats["num_edges"])
        self.assertEqual(edges.shape[1], 3)
        self.assertGreater(edges.size, 0)
        # edge_weights align with the edges
        self.assertEqual(feats["edge_weights"].shape[:2], edges.shape[:2])


class TestCostOperatorConvention(TrainingPipelineTestCase):
    """The models consume edge weights, so the operator's convention matters."""

    def test_wrong_pre_factor_warns(self):
        """An all-positive quadratic operator is the pre_factor=1.0 mistake."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        with self.assertWarns(UserWarning):
            make_extractor().extract_np(op)

    def test_training_convention_does_not_warn(self):
        """The convention the models were trained on must pass silently."""
        op = SparsePauliOp.from_list([("ZZI", -1.0), ("IZZ", -1.0), ("ZIZ", -1.0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_extractor().extract_np(op)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_edge_weights_undo_the_pre_factor(self):
        """extract_np recovers graph weights by dividing out the pre-factor."""
        op = SparsePauliOp.from_list([("ZZI", -0.5), ("IZZ", -0.5), ("ZIZ", -0.5)])
        feats = make_extractor().extract_np(op)
        # The operator is normalized by rescaling_factor first, so compare the
        # weights to the normalized coefficients divided by the pre-factor.
        expected = (-0.5 / feats["rescale_a"]) / MAX_CUT
        np.testing.assert_allclose(feats["edge_weights"].squeeze(0), [expected] * 3, rtol=1e-6)


class TestPreFactorSignAndMagnitude(TrainingPipelineTestCase):
    """Normalization divides out the pre-factor's magnitude, not its sign.

    This is what decides whether the sign-convention warning is still needed
    once the operator is normalized. It is: two pre-factors of the same sign
    are indistinguishable after normalization, so the magnitude is free, while
    the opposite sign survives normalization as an exact negation and silently
    predicts for the wrong Hamiltonian.
    """

    @staticmethod
    def _weights(pre_factor):
        """Recovered edge weights for a uniform triangle at ``pre_factor``."""
        op = SparsePauliOp.from_list(
            [("ZZI", pre_factor), ("IZZ", pre_factor), ("ZIZ", pre_factor)]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return make_extractor().extract_np(op)["edge_weights"].squeeze(0)

    def test_magnitude_divides_out(self):
        """-0.5 and -2.0 are the same normalized problem, so the same features."""
        np.testing.assert_allclose(self._weights(-0.5), self._weights(-2.0), rtol=1e-6)

    def test_the_opposite_sign_survives_as_a_negation(self):
        """+1.0 normalizes to the negated operator -- the case worth warning about."""
        np.testing.assert_allclose(self._weights(1.0), -self._weights(-0.5), rtol=1e-6)

    def test_mixed_sign_weights_are_the_documented_blind_spot(self):
        """Neither spelling of a mixed-sign graph warns; the test pins that down.

        Nothing is wrong here -- it is why the docstring says the convention is
        the caller's to get right when weights have both signs.
        """
        for pre_factor in (-0.5, 1.0):
            op = SparsePauliOp.from_list(
                [("ZZI", pre_factor), ("IZZ", -pre_factor), ("ZIZ", 2 * pre_factor)]
            )
            with self.subTest(pre_factor=pre_factor):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    make_extractor().extract_np(op)
                self.assertEqual([str(w.message) for w in caught], [])


class TestDeclaredPrefactor(TrainingPipelineTestCase):
    """The convention comes from the bundle's setup entry, not from a constant."""

    OPERATOR = SparsePauliOp.from_list([("ZZI", -0.5), ("IZZ", -0.5), ("ZIZ", -0.5)])

    def _weights(self, prefactor):
        """Recovered edge weights when the bundle declares ``prefactor``."""
        extractor = AIFeatureExtractor(in_features=[], norm_stats={}, prefactor=prefactor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return extractor.extract_np(self.OPERATOR)["edge_weights"].squeeze(0)

    def test_the_declared_prefactor_divides_the_coefficients(self):
        """Halving the prefactor doubles the recovered weights."""
        np.testing.assert_allclose(self._weights(-0.25), self._weights(-0.5) * 2.0, rtol=1e-6)

    def test_the_prefactor_is_required(self):
        """There is no default convention to fall back on."""
        with self.assertRaises(TypeError):
            AIFeatureExtractor(in_features=[], norm_stats={})  # no prefactor

    def test_a_zero_prefactor_is_refused(self):
        """It is a divisor, so zero would make every weight infinite."""
        with self.assertRaises(ValueError) as ctx:
            AIFeatureExtractor(in_features=[], norm_stats={}, prefactor=0.0)
        self.assertIn("non-zero", str(ctx.exception))

    def test_the_warning_follows_the_declared_sign(self):
        """A positive-prefactor model must not be warned about positive coefficients."""
        positive = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0)])
        extractor = AIFeatureExtractor(in_features=[], norm_stats={}, prefactor=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extractor.extract_np(positive)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_the_prefactor_survives_a_config_round_trip(self):
        """A rebuilt extractor must recover the same weights."""
        extractor = AIFeatureExtractor(in_features=[], norm_stats={}, prefactor=-2.0)
        rebuilt = AIFeatureExtractor.from_config(extractor.to_config())
        self.assertEqual(rebuilt.prefactor, -2.0)


class TestExtractorInterface(TrainingPipelineTestCase):
    """AIFeatureExtractor is a BaseFeatureExtractor, so it answers that interface."""

    def test_features_are_reported_in_packing_order(self):
        """features() must be the order pack_features_np actually packs in."""
        extractor = AIFeatureExtractor(
            in_features=["num_nodes", "mean_degree"],
            norm_stats=IDENTITY_STATS,
            prefactor=MAX_CUT,
        )
        self.assertEqual(extractor.features(), ["mean_degree", "num_nodes"])

    def test_config_round_trip(self):
        """A rebuilt extractor packs the same vector as the original."""
        extractor = AIFeatureExtractor(
            in_features=sorted(SCALAR_FEATURES), norm_stats=IDENTITY_STATS, prefactor=MAX_CUT
        )
        rebuilt = AIFeatureExtractor.from_config(extractor.to_config())
        op = SparsePauliOp.from_list([("ZZI", -0.5), ("IZZ", -0.5), ("ZIZ", -0.5)])
        np.testing.assert_allclose(rebuilt(op)[0], extractor(op)[0], rtol=1e-6)

    def test_a_config_without_norm_stats_is_refused(self):
        """Rebuilding without stats would silently stop normalizing."""
        with self.assertRaises(KeyError) as ctx:
            AIFeatureExtractor.from_config({"in_features": ["num_nodes"]})
        self.assertIn("norm_stats", str(ctx.exception))


class TestOutOfDistributionWarning(TrainingPipelineTestCase):
    """A graph far outside the training set must not predict silently."""

    @staticmethod
    def _extractor():
        """An extractor whose training set was 'num_nodes around 10 +- 2'."""
        stats = {name: {"mean": 0.0, "std": 1.0} for name in SCALAR_FEATURES}
        stats["num_nodes"] = {"mean": 10.0, "std": 2.0}
        return AIFeatureExtractor(
            in_features=sorted(SCALAR_FEATURES), norm_stats=stats, prefactor=MAX_CUT
        )

    def test_in_distribution_is_silent(self):
        """A problem the model was trained on must not warn."""
        features = {name: 0.0 for name in SCALAR_FEATURES}
        features["num_nodes"] = 10.0 + OUT_OF_DISTRIBUTION_SIGMA * 2.0 - 0.1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._extractor().pack_features_np(features)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_extrapolation_warns_and_still_predicts(self):
        """Beyond the threshold the caller is told, but gets a vector anyway."""
        features = {name: 0.0 for name in SCALAR_FEATURES}
        features["num_nodes"] = 10.0 + OUT_OF_DISTRIBUTION_SIGMA * 2.0 + 1.0
        extractor = self._extractor()
        with self.assertWarns(UserWarning) as ctx:
            packed = extractor.pack_features_np(features)
        message = str(ctx.warning)
        self.assertIn("num_nodes", message)
        self.assertIn("outside the distribution", message)
        self.assertEqual(packed.shape, (1, len(SCALAR_FEATURES)))
        self.assertTrue(np.isfinite(packed).all())

    def test_a_feature_without_stats_cannot_trigger_it(self):
        """With no mean/std there is no distribution to be outside of."""
        extractor = AIFeatureExtractor(
            in_features=["rescale_a"], norm_stats=IDENTITY_STATS, prefactor=MAX_CUT
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extractor.pack_features_np({"rescale_a": 1e9})
        self.assertEqual([str(w.message) for w in caught], [])
