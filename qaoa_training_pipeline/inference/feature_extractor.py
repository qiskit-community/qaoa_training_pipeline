"""
Feature extraction utilities for QAOA inference.

This module provides explicit feature extraction from cost operators,
making the feature engineering process transparent and configurable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.graph_utils import operator_to_graph
from qaoa_training_pipeline.inference.datamodule_utils import rescaling_factor


SCALAR_FEATURES = {"num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"}


class AIFeatureExtractor:
    """Extract features from QAOA cost operators.

    This class makes feature extraction explicit and configurable,
    separating it from the model inference logic.

    Example:
        extractor = AIFeatureExtractor(in_features=["num_nodes", "mean_degree"])
        x_vec, features = extractor.extract_and_pack_np(cost_op)
    """

    def __init__(
        self,
        in_features: list[str],
        norm_stats: dict[str, dict[str, float]],
    ) -> None:
        """
        Initialize feature extractor.

        Args:
            in_features: List of feature names to extract
            norm_stats: Dictionary of normalization statistics
        """
        self.in_features = list(in_features)
        self.norm_stats = norm_stats

        missing = [
            name for name in self.in_features if name in SCALAR_FEATURES and name not in norm_stats
        ]
        if missing:
            raise KeyError(
                f"FeatureExtractor: features {missing} listed in in_features have no "
                f"entry in norm_stats. "
                f"Available stats: {sorted(norm_stats)}"
            )

    def extract_np(self, cost_op: SparsePauliOp) -> dict[str, Any]:
        """Extract raw (unnormalized) features from a cost operator as numpy.

        Rescales the operator by ``rescaling_factor``, converts to a graph, and
        computes the same scalar/graph features used during training:
            edges         (1, M, 2) int64
            edge_weights  (1, M)    float32
            node_count    (1,)      int64
            t             (1,)      int64
            nodes, rescale_a
        """
        rescale_a = rescaling_factor(cost_op)
        graph = operator_to_graph(cost_op / rescale_a)

        edge_list = list(graph.edges())
        pre_factor = -0.5
        edge_weights = [graph.edges[u, v].get("weight", 1.0) / pre_factor for u, v in edge_list]
        degrees = np.asarray([deg for node, deg in graph.degree()])
        num_nodes = int(graph.number_of_nodes())
        num_edges = int(graph.number_of_edges())

        edges_arr = np.asarray(edge_list if edge_list else [[0, 0]], dtype=np.int64)
        weights_arr = np.asarray(edge_weights if edge_weights else [0.0], dtype=np.float32)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "edges_per_node": num_edges / num_nodes if num_nodes else 0.0,
            "mean_degree": float(np.mean(degrees)) if degrees.size else 0.0,
            "std_degree": float(np.std(degrees)) if degrees.size else 0.0,
            "rescale_a": float(rescale_a),
            "nodes": list(graph.nodes()),
            "edges": edges_arr[np.newaxis, ...],
            "edge_weights": weights_arr[np.newaxis, ...],
            "node_count": np.asarray([num_nodes], dtype=np.int64),
            "t": np.zeros(1, dtype=np.int64),
        }

    def pack_features_np(self, features: dict[str, Any]) -> np.ndarray:
        """Pack the requested ``in_features`` into a (1, F) float32 array.

        Values are placed in sorted-name order, applying normalization for
        entries listed in ``norm_stats``.
        """
        values: list[float] = []
        for name in sorted(self.in_features):
            if name not in features:
                raise KeyError(f"FeatureExtractor: feature {name!r} not produced by extract_np()")
            v = float(features[name])
            stats = self.norm_stats.get(name)
            if stats is not None:
                mean = float(stats["mean"])
                std = float(stats["std"])
                if std <= 1e-12:
                    std = 1.0
                v = (v - mean) / std
            values.append(v)
        return np.asarray(values, dtype=np.float32)[np.newaxis, ...]

    def extract_and_pack_np(self, cost_op: SparsePauliOp) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract and pack features in one call. Returns ``(x_vec, features)``."""
        features = self.extract_np(cost_op)
        x_vec = self.pack_features_np(features)
        return x_vec, features
