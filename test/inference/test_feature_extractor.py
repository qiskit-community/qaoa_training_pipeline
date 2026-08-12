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
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor

from ..training_pipeline_test_case import TrainingPipelineTestCase

SCALAR_FEATURES = ["num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"]

# Identity-ish norm stats (mean 0, std 1) so normalized == raw for easy checks.
IDENTITY_STATS = {name: {"mean": 0.0, "std": 1.0} for name in SCALAR_FEATURES}


def make_extractor(norm_stats=None):
    """Build an extractor over all scalar features."""
    return AIFeatureExtractor(in_features=SCALAR_FEATURES, norm_stats=norm_stats or IDENTITY_STATS)


class TestAIFeatureExtractor(TrainingPipelineTestCase):
    """Test the numpy feature-extraction path used by the ONNX backend."""

    def test_missing_norm_stats_raises(self):
        """A scalar feature without norm stats must fail fast at construction."""
        with self.assertRaises(KeyError):
            AIFeatureExtractor(
                in_features=SCALAR_FEATURES,
                norm_stats={"num_nodes": {"mean": 0.0, "std": 1.0}},
            )

    def test_extract_triangle_graph_topology(self):
        """A 3-qubit triangle operator maps to a 3-node, 3-edge graph."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        feats = make_extractor().extract_np(op)
        self.assertEqual(feats["num_nodes"], 3)
        self.assertEqual(feats["num_edges"], 3)
        self.assertAlmostEqual(feats["edges_per_node"], 1.0)
        # every node in a triangle has degree 2
        self.assertAlmostEqual(feats["mean_degree"], 2.0)
        self.assertAlmostEqual(feats["std_degree"], 0.0)

    def test_extract_line_graph_topology(self):
        """A 4-qubit line operator maps to a 4-node, 3-edge path graph."""
        op = SparsePauliOp.from_list([("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0)])
        feats = make_extractor().extract_np(op)
        self.assertEqual(feats["num_nodes"], 4)
        self.assertEqual(feats["num_edges"], 3)
        # path degrees are [1, 2, 2, 1] -> mean 1.5
        self.assertAlmostEqual(feats["mean_degree"], 1.5)
        self.assertGreater(feats["std_degree"], 0.0)

    def test_pack_features_shape_and_order(self):
        """pack_features_np returns a (1, n_features) array in sorted-name order."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
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
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        stats = {name: {"mean": 1.0, "std": 2.0} for name in SCALAR_FEATURES}
        ext = AIFeatureExtractor(in_features=SCALAR_FEATURES, norm_stats=stats)
        feats = ext.extract_np(op)
        x = ext.pack_features_np(feats).squeeze(0)
        expected = [(float(feats[name]) - 1.0) / 2.0 for name in sorted(SCALAR_FEATURES)]
        np.testing.assert_allclose(x, expected, rtol=1e-6)

    def test_pack_features_zero_std_is_safe(self):
        """A zero std must not produce inf/nan (guarded to std=1)."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        stats = {name: {"mean": 0.0, "std": 0.0} for name in SCALAR_FEATURES}
        ext = AIFeatureExtractor(in_features=SCALAR_FEATURES, norm_stats=stats)
        x = ext.pack_features_np(ext.extract_np(op))
        self.assertTrue(np.isfinite(x).all())

    def test_pack_features_missing_feature_raises(self):
        """Packing a feature dict missing a requested feature must raise."""
        ext = make_extractor()
        with self.assertRaises(KeyError):
            ext.pack_features_np({"num_nodes": 3})  # missing the rest

    def test_extract_and_pack_consistency(self):
        """extract_and_pack_np equals extract_np followed by pack_features_np."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
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
