#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Extraction of the graph features the inference models consume.

The models take a scale-normalized max-cut graph, not a cost operator, so this
module turns one into the other: it recovers the edge weights from the Pauli
coefficients and computes the scalar features the networks were trained on.
Which operators are acceptable in the first place is
:mod:`~qaoa_training_pipeline.inference.conventions`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.conventions import (
    validate_cost_op,
    warn_on_unexpected_sign_convention,
)
from qaoa_training_pipeline.pre_processing.feature_extraction import BaseFeatureExtractor
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


SCALAR_FEATURES = {"num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"}

# How far outside the training distribution a feature may land before the prediction is flagged.
OUT_OF_DISTRIBUTION_SIGMA = 3.0


class AIFeatureExtractor(BaseFeatureExtractor):
    """Extract and normalize the features the ONNX models consume.

    A :class:`~qaoa_training_pipeline.pre_processing.BaseFeatureExtractor`, so
    it carries the same ``__call__`` / :meth:`features` / config interface as
    :class:`~qaoa_training_pipeline.pre_processing.GraphFeatureExtractor`, and
    shares ``operator_to_graph`` with it. The bodies stay separate on purpose:
    that extractor returns a positional tuple under its own feature names
    (``avg_degree``, ``avg_weight``, ``density``, ``qaoa_depth``) with no
    normalization, while the vector here is fixed by a trained contract --
    ``sorted(in_features)`` ordering, each bundle's own mean/std from
    ``feature_normalization``, and the edge/weight tensors the graph networks
    read. Folding one into the other would make a published bundle's input
    vector depend on the other extractor's defaults.

    Example:
        extractor = AIFeatureExtractor(
            in_features=["num_nodes", "mean_degree"], norm_stats=stats, prefactor=0.5
        )
        normalized, scale = normalize_cost_op(cost_op)
        x_vec, features = extractor.extract_and_pack_np(normalized, scale)
    """

    def __init__(
        self,
        in_features: list[str],
        norm_stats: dict[str, dict[str, float]],
        prefactor: float,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            in_features: Names of the features to extract, in the order the
                model's input vector expects them.
            norm_stats: Per-feature ``{"mean": ..., "std": ...}`` normalization
                statistics, taken from the bundle's own training set.
            prefactor: The cost-operator convention this bundle was trained
                under, ``H = prefactor * sum_e w_e Z_i Z_j``, as its setup entry
                declares it. The models consume edge *weights*, so the Pauli
                coefficients are divided by it to recover them. Required: the
                convention belongs to the bundle, so this package keeps no
                default that could silently apply to the wrong model.

        Raises:
            ValueError: If ``prefactor`` is zero, which would make every
                recovered edge weight infinite.
        """
        self.in_features = list(in_features)
        self.norm_stats = norm_stats
        self.prefactor = float(prefactor)
        if self.prefactor == 0.0:
            raise ValueError(
                "prefactor must be non-zero: it is what the Pauli coefficients are divided "
                "by to recover the edge weights the model consumes."
            )
        # What the graph conversion is given. ``operator_to_graph`` *multiplies*
        # the coefficients by its pre_factor, while recovering weights means
        # dividing by the convention -- hence the reciprocal. (solve_max_cut
        # does the same thing with a hardcoded pre_factor=-2.)
        self._weight_recovery_factor = 1.0 / self.prefactor
        # One out-of-distribution warning per extractor, not per call: a size
        # sweep would otherwise emit a paragraph on every iteration.
        self._warned_out_of_distribution = False

        missing = [
            name for name in self.in_features if name in SCALAR_FEATURES and name not in norm_stats
        ]
        if missing:
            raise KeyError(
                f"FeatureExtractor: features {missing} listed in in_features have no "
                f"entry in norm_stats. "
                f"Available stats: {sorted(norm_stats)}"
            )

    def extract_np(self, cost_op: SparsePauliOp, cost_op_scale: float) -> dict[str, Any]:
        """Extract raw (unnormalized) features from a cost operator as numpy.

        Converts the operator to a graph and computes the same scalar/graph
        features used during training:
            edges         (1, M, 2) int64
            edge_weights  (1, M)    float32
            node_count    (1,)      int64
            nodes, cost_op_scale

        Args:
            cost_op: The cost operator, already divided by ``cost_op_scale``
                (:func:`~qaoa_training_pipeline.inference.scaling.normalize_cost_op`).
            cost_op_scale: The scale it was divided by, reported back as a
                feature.
        """
        cost_op = SparsePauliOp(cost_op).simplify()

        # A constant energy offset shifts every cut by the same amount, so it
        # cannot affect the angles -- but operator_to_graph has no edge to map
        # it onto and would reject the whole operator as "not Quadratic". Drop
        # it here, before it reaches either the graph or the normalization.
        identity = "I" * cost_op.num_qubits
        if any(str(term.paulis[0]) == identity for term in cost_op):
            kept = [
                (str(term.paulis[0]), complex(term.coeffs[0]))
                for term in cost_op
                if str(term.paulis[0]) != identity
            ]
            cost_op = (
                SparsePauliOp.from_list(kept, num_qubits=cost_op.num_qubits)
                if kept
                else SparsePauliOp.from_list([(identity, 0.0)], num_qubits=cost_op.num_qubits)
            )

        validate_cost_op(cost_op)
        warn_on_unexpected_sign_convention(cost_op, self.prefactor)

        # include_one_local is off because validate_cost_op has already refused
        # one-local terms: a self-loop here would be double-counted into the
        # degrees by the symmetrization in onnx_inputs.
        graph = operator_to_graph(
            cost_op, pre_factor=self._weight_recovery_factor, include_one_local=False
        )

        edge_list = list(graph.edges())
        edge_weights = [graph.edges[u, v].get("weight", 1.0) for u, v in edge_list]
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
            # Keyed "rescale_a", not cost_op_scale: this is a model *input
            # feature* name, matched against the bundle's own in_features, so
            # it is fixed by what the models were exported with.
            "rescale_a": float(cost_op_scale),
            "nodes": list(graph.nodes()),
            "edges": edges_arr[np.newaxis, ...],
            "edge_weights": weights_arr[np.newaxis, ...],
            "node_count": np.asarray([num_nodes], dtype=np.int64),
        }

    def pack_features_np(self, features: dict[str, Any]) -> np.ndarray:
        """Pack the requested ``in_features`` into a (1, F) float32 array.

        Values are placed in sorted-name order, applying normalization for
        entries listed in ``norm_stats``. A feature further than
        :data:`OUT_OF_DISTRIBUTION_SIGMA` from its training mean is warned
        about: the model will still predict, but it is extrapolating. The
        warning is emitted once per extractor, so sweeping over problem sizes
        does not emit one per iteration.
        """
        values: list[float] = []
        extrapolated: list[str] = []
        for name in sorted(self.in_features):
            if name not in features:
                raise KeyError(f"FeatureExtractor: feature {name!r} not produced by extract_np()")
            raw = float(features[name])
            v = raw
            stats = self.norm_stats.get(name)
            if stats is not None:
                mean = float(stats["mean"])
                std = float(stats["std"])
                if std <= 1e-12:
                    # The feature was constant across the training set, so a
                    # sigma distance is not defined. Normalizing by 1.0 to keep value finite
                    v = raw - mean
                    if abs(v) > 0.0:
                        extrapolated.append(
                            f"{name}={raw:g} differs from the training value {mean:g}, which "
                            "had zero variance in training"
                        )
                else:
                    v = (raw - mean) / std
                    if abs(v) > OUT_OF_DISTRIBUTION_SIGMA:
                        extrapolated.append(
                            f"{name}={raw:g} is {abs(v):.1f} sigma from the training mean "
                            f"{mean:g} (std {std:g})"
                        )
            values.append(v)

        if extrapolated and not self._warned_out_of_distribution:
            self._warned_out_of_distribution = True
            warnings.warn(
                "This problem is outside the distribution the model was trained on: "
                + "; ".join(extrapolated)
                + f". The threshold is {OUT_OF_DISTRIBUTION_SIGMA:g} sigma, taken from the "
                "bundle's own feature_normalization. The predicted angles are an "
                "extrapolation; treat them as a starting point for optimization rather "
                "than as trained-quality parameters.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(values, dtype=np.float32)[np.newaxis, ...]

    def extract_and_pack_np(
        self, cost_op: SparsePauliOp, cost_op_scale: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract and pack features in one call. Returns ``(x_vec, features)``.

        ``cost_op_scale`` is passed straight to :meth:`extract_np`.
        """
        features = self.extract_np(cost_op, cost_op_scale=cost_op_scale)
        x_vec = self.pack_features_np(features)
        return x_vec, features

    def __call__(
        self, cost_op: SparsePauliOp, cost_op_scale: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Alias of :meth:`extract_and_pack_np`, the base class's entry point."""
        return self.extract_and_pack_np(cost_op, cost_op_scale)

    def features(self) -> list[str]:
        """The feature names, in the order the packed vector carries them.

        Sorted, because that is the order :meth:`pack_features_np` packs in
        """
        return sorted(self.in_features)

    def to_config(self) -> dict:
        """Return a config based on the class instance."""
        config = super().to_config()
        config["in_features"] = list(self.in_features)
        config["norm_stats"] = self.norm_stats
        config["prefactor"] = self.prefactor
        return config

    @classmethod
    def from_config(cls, config: dict) -> "AIFeatureExtractor":
        """Return an instance of this class based on a config."""
        missing = [key for key in ("in_features", "norm_stats") if key not in config]
        if missing:
            raise KeyError(
                f"AIFeatureExtractor config is missing {missing}. All are required: "
                "without norm_stats the features would go to the network unnormalized, "
                "and the prefactor is the bundle's own cost-operator convention."
            )
        return cls(
            in_features=list(config["in_features"]),
            norm_stats=config["norm_stats"],
            prefactor=config["prefactor"],
        )
