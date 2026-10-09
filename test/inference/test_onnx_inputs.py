#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the numpy feed builders behind the ONNX runtime.

Each builder turns the extracted features into the exact feed dict an exported
graph declares, so a wrong shape, dtype or edge convention here is a silently
wrong prediction. The builders are pure numpy, so these tests need neither
onnxruntime nor a downloaded bundle and always run.
"""

import unittest

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
from qaoa_training_pipeline.inference.model_registry import VERIFIED_ARCHITECTURES
from qaoa_training_pipeline.inference.onnx_inputs import (
    numpy_input_builders,
    prepare_diffusion_transformer,
    prepare_edge_transformer,
    prepare_gcn,
    prepare_gnn,
    prepare_graph_transformer,
    prepare_mlp,
)

from ..training_pipeline_test_case import TrainingPipelineTestCase

SCALAR_FEATURES = ["num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"]

# The builders never look at the scalar normalization, only at its output shape, so
# unit stats keep the packed vector equal to the raw features and the tests readable.
NORM_STATS = {name: {"mean": 0.0, "std": 1.0} for name in SCALAR_FEATURES}


def _zz(num_qubits, i, j, weight=1.0):
    """A single weighted ZZ term."""
    label = ["I"] * num_qubits
    label[i] = "Z"
    label[j] = "Z"
    return SparsePauliOp(["".join(reversed(label))], coeffs=[weight])


def _ring(n):
    """A uniformly weighted ring on ``n`` nodes."""
    return sum(_zz(n, i, (i + 1) % n) for i in range(n)).simplify()


def _star(n):
    """A star graph: node 0 joined to every other node (degrees differ)."""
    return sum(_zz(n, 0, i) for i in range(1, n)).simplify()


def _features(cost_op, pos_enc_dim=None):
    """Build the feature dict exactly as OnnxQAOAPredictor.predict does."""
    extractor = AIFeatureExtractor(in_features=SCALAR_FEATURES, norm_stats=NORM_STATS)
    x_vec, features = extractor.extract_and_pack_np(cost_op)
    features["x"] = x_vec
    if pos_enc_dim is not None:
        features["pos_enc_dim"] = pos_enc_dim
    return features


class TestBuilderRegistry(TrainingPipelineTestCase):
    """The registry is what dispatches a bundle to its feed builder."""

    def test_every_verified_architecture_has_a_builder(self):
        """A bundle that loads must have a builder for its declared model_type."""
        for architecture, (_, model_type) in VERIFIED_ARCHITECTURES.items():
            with self.subTest(architecture=architecture):
                self.assertIn(model_type, numpy_input_builders)

    def test_gin_and_gnn_share_a_builder(self):
        """Both message-passing models feed the same tensors."""
        self.assertIs(
            numpy_input_builders["graph_isomorphism_network"],
            numpy_input_builders["graph_neural_network"],
        )


class TestFeedShapesAndDtypes(TrainingPipelineTestCase):
    """Every builder emits finite arrays of the dtypes the graphs declare."""

    def setUp(self):
        super().setUp()
        self.features = _features(_ring(6), pos_enc_dim=4)

    def test_all_builders_emit_finite_arrays(self):
        """No builder produces NaN/inf, which would poison the prediction."""
        for model_type, builder in numpy_input_builders.items():
            with self.subTest(model_type=model_type):
                for name, array in builder(self.features).items():
                    self.assertTrue(
                        np.all(np.isfinite(array)), f"{model_type}/{name} is not finite"
                    )

    def test_index_inputs_are_int64_and_values_float32(self):
        """ONNX is strict about dtypes: indices int64, everything else float32."""
        int_inputs = {"edges", "edge_index", "node_count"}
        for model_type, builder in numpy_input_builders.items():
            for name, array in builder(self.features).items():
                with self.subTest(model_type=model_type, input=name):
                    expected = np.int64 if name in int_inputs else np.float32
                    self.assertEqual(array.dtype, expected)

    def test_mlp_feeds_only_the_scalar_vector(self):
        """The aggregate model sees no graph structure at all."""
        feed = prepare_mlp(self.features)
        self.assertEqual(list(feed), ["x"])
        self.assertEqual(feed["x"].shape, (1, len(SCALAR_FEATURES)))

    def test_diffusion_matches_edge_transformer(self):
        """t=0/no-noise is baked into the export, so the feeds are identical."""
        diffusion = prepare_diffusion_transformer(self.features)
        edge = prepare_edge_transformer(self.features)
        self.assertEqual(sorted(diffusion), sorted(edge))
        for name, array in diffusion.items():
            np.testing.assert_array_equal(array, edge[name])

    def test_diffusion_does_not_feed_the_timestep(self):
        """``t`` is in the features but must not reach the graph."""
        self.assertNotIn("t", prepare_diffusion_transformer(self.features))


class TestEdgeConventions(TrainingPipelineTestCase):
    """The edge-list conventions differ per builder and are easy to get wrong."""

    def setUp(self):
        super().setUp()
        self.num_nodes = 6
        self.features = _features(_ring(self.num_nodes), pos_enc_dim=4)
        self.num_edges = int(np.asarray(self.features["edges"]).shape[1])

    def test_edge_transformer_passes_edges_through_padded(self):
        """It consumes the (1, M, 2) padded list as-is, without symmetrizing."""
        feed = prepare_edge_transformer(self.features)
        self.assertEqual(feed["edges"].shape, (1, self.num_edges, 2))
        self.assertEqual(feed["edge_weights"].shape, (1, self.num_edges))

    def test_gnn_symmetrizes_into_a_flat_edge_index(self):
        """Message passing needs both directions: (2, 2M) with (2M, 1) attrs."""
        feed = prepare_gnn(self.features)
        self.assertEqual(feed["edge_index"].shape, (2, 2 * self.num_edges))
        self.assertEqual(feed["edge_attr"].shape, (2 * self.num_edges, 1))
        np.testing.assert_array_equal(feed["node_count"], [self.num_nodes])

    def test_gnn_edge_index_contains_every_reverse_edge(self):
        """Symmetrization must be exact, not approximate."""
        edge_index = prepare_gnn(self.features)["edge_index"]
        pairs = {tuple(pair) for pair in edge_index.T.tolist()}
        for source, target in pairs:
            self.assertIn((target, source), pairs)

    def test_zero_weight_edges_are_dropped(self):
        """A zero-weight edge is no edge; keeping it would skew the degrees."""
        features = dict(self.features)
        weights = np.asarray(features["edge_weights"]).copy()
        weights[0, 0] = 0.0
        features["edge_weights"] = weights
        feed = prepare_gnn(features)
        self.assertEqual(feed["edge_index"].shape, (2, 2 * (self.num_edges - 1)))

    def test_gcn_node_feature_is_the_z_scored_degree(self):
        """The export wrapper expects the per-graph z-score computed here."""
        feed = prepare_gcn(_features(_star(7)))
        node_x = feed["node_x"]
        self.assertEqual(node_x.shape, (7, 1))
        self.assertAlmostEqual(float(node_x.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(node_x.std(ddof=1)), 1.0, places=5)

    def test_gcn_regular_graph_degrees_collapse_to_zero(self):
        """A constant degree has zero spread; the guard must avoid a 0/0 NaN."""
        node_x = prepare_gcn(self.features)["node_x"]
        np.testing.assert_allclose(node_x, np.zeros_like(node_x))

    def test_gcn_edge_weight_is_flat(self):
        """GraphConv wants (2M,), not the (2M, 1) attr shape the GNN uses."""
        feed = prepare_gcn(self.features)
        self.assertEqual(feed["edge_weight"].shape, (2 * self.num_edges,))


class TestGraphTransformerPrecomputation(TrainingPipelineTestCase):
    """Its node features and Laplacian PE are computed here, not in the graph."""

    def setUp(self):
        super().setUp()
        self.num_nodes = 6
        self.features = _features(_ring(self.num_nodes), pos_enc_dim=4)

    def test_positional_encoding_width_follows_pos_enc_dim(self):
        """The width is a model hyperparameter, surfaced from model_init."""
        for pos_enc_dim in (2, 4, 8):
            with self.subTest(pos_enc_dim=pos_enc_dim):
                feed = prepare_graph_transformer(_features(_ring(6), pos_enc_dim))
                self.assertEqual(feed["pos_enc"].shape, (6, pos_enc_dim))

    def test_positional_encoding_is_padded_when_the_graph_is_small(self):
        """Fewer eigenvectors than requested must pad, not fail."""
        feed = prepare_graph_transformer(_features(_ring(3), pos_enc_dim=8))
        self.assertEqual(feed["pos_enc"].shape, (3, 8))

    def test_node_features_are_degree_and_weighted_degree(self):
        """Two columns: log1p(degree) and signed log1p(weighted degree)."""
        feed = prepare_graph_transformer(self.features)
        self.assertEqual(feed["node_features"].shape, (self.num_nodes, 2))
        # Every node of a ring has degree 2 after symmetrization.
        np.testing.assert_allclose(
            feed["node_features"][:, 0], np.full(self.num_nodes, np.log1p(2.0)), atol=1e-6
        )

    def test_default_positional_encoding_width(self):
        """Without pos_enc_dim in the features the builder falls back to 8."""
        feed = prepare_graph_transformer(_features(_ring(6)))
        self.assertEqual(feed["pos_enc"].shape, (6, 8))


if __name__ == "__main__":
    unittest.main()
