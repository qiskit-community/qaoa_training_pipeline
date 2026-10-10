"""
Feature extraction utilities for QAOA inference.

This module provides explicit feature extraction from cost operators,
making the feature engineering process transparent and configurable.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


SCALAR_FEATURES = {"num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"}

# The convention every shipped model was trained under: the max-cut cost
# operator is ``H = -0.5 * sum_{(u,v) in E} w_uv Z_u Z_v``, i.e. what
# ``graph_to_operator(graph, pre_factor=MAX_CUT_PRE_FACTOR)`` produces. The
# models consume edge *weights*, not Pauli coefficients, so ``extract_np``
# divides the coefficients by this factor to recover them. An operator built
# with a different pre-factor is not rejected -- it describes a perfectly valid
# Hamiltonian -- but its recovered weights are off by a constant, and the
# predicted gammas are quietly wrong.
MAX_CUT_PRE_FACTOR = -0.5


def warn_on_unexpected_sign_convention(cost_op: SparsePauliOp) -> None:
    """Warn when ``cost_op`` looks like it was built with the wrong pre-factor.

    Nothing in an operator records which pre-factor produced it, so the general
    case cannot be detected. The common mistake can be: under
    :data:`MAX_CUT_PRE_FACTOR` a graph with positive edge weights gives
    *negative* quadratic coefficients, so an operator whose quadratic
    coefficients are all positive was almost certainly built with the
    ``pre_factor=1.0`` default of
    :func:`~qaoa_training_pipeline.utils.graph_utils.graph_to_operator`.
    """
    coefficients = [
        float(np.real(term.coeffs[0])) for term in cost_op if int(sum(term.paulis[0].z)) > 1
    ]
    if coefficients and all(coefficient > 0.0 for coefficient in coefficients):
        warnings.warn(
            "All quadratic coefficients of cost_op are positive. The inference models "
            f"expect a max-cut operator in the convention H = {MAX_CUT_PRE_FACTOR} * "
            "sum_e w_e Z_i Z_j -- graph_to_operator(graph, "
            f"pre_factor={MAX_CUT_PRE_FACTOR}) -- under which positive edge weights "
            "give negative coefficients. An operator built with the pre_factor=1.0 "
            "default still yields finite angles, but the predicted gammas are wrong. "
            "Pass pre_factor=-0.5, or ignore this warning if your edge weights really "
            "are negative.",
            UserWarning,
            stacklevel=3,
        )


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

    def rescaling_factor(self, cost_op):
        """Return the QAOA cost-operator rescaling factor (RMS of per-order weights).

        The factor is used as a divisor both on input (``cost_op / rescale_a``) and
        on output (gammas ``/ rescale_a``). For a degenerate operator with no
        non-identity terms or all-zero coefficients the RMS is 0; we fall back to
        ``1.0`` so normalization/denormalization is a no-op instead of dividing by
        zero.
        """
        terms = defaultdict(list)

        for p in cost_op:
            order = sum(p.paulis[0].z)
            terms[order].append(np.real(p.coeffs[0]) ** 2)

        factor = 0
        for squared_weights in terms.values():
            factor += sum(squared_weights) / len(squared_weights)

        factor = np.sqrt(factor)
        return factor if factor > 0 else 1.0

    def extract_np(self, cost_op: SparsePauliOp) -> dict[str, Any]:
        """Extract raw (unnormalized) features from a cost operator as numpy.

        Rescales the operator by ``rescaling_factor``, converts to a graph, and
        computes the same scalar/graph features used during training:
            edges         (1, M, 2) int64
            edge_weights  (1, M)    float32
            node_count    (1,)      int64
            t             (1,)      int64
            nodes, rescale_a

        ``cost_op`` must be a max-cut operator in the training convention
        ``H = -0.5 * sum_e w_e Z_i Z_j`` (see :data:`MAX_CUT_PRE_FACTOR`): the
        edge weights the models were trained on are recovered by dividing the
        coefficients by that pre-factor. An operator built differently predicts
        finite but wrong angles, so the obvious case of it is warned about.
        """
        warn_on_unexpected_sign_convention(cost_op)

        rescale_a = self.rescaling_factor(cost_op)
        graph = operator_to_graph(cost_op / rescale_a)

        edge_list = list(graph.edges())
        edge_weights = [
            graph.edges[u, v].get("weight", 1.0) / MAX_CUT_PRE_FACTOR for u, v in edge_list
        ]
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
