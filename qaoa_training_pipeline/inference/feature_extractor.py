#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Extraction of the graph features the inference models consume.

The models take a scale-normalized max-cut graph, not a cost operator, so this
module turns one into the other: it normalizes the operator, recovers the edge
weights from the Pauli coefficients, computes the scalar features the networks
were trained on, and rejects the operators that cannot be expressed as such a
graph instead of silently feeding a malformed one.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.warn_utils import warn_at_caller
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

# How far outside the training distribution a feature may land before the
# prediction is flagged. The bundles carry the per-feature mean and standard
# deviation of their training set, so "far" is measurable: at 3 sigma the shipped
# models cover roughly 1-150 nodes, which is the range they were trained on.
# Beyond it the networks extrapolate, and extrapolate silently -- the angles come
# back finite and plausible with nothing to say they are guesses.
OUT_OF_DISTRIBUTION_SIGMA = 3.0


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
        warn_at_caller(
            "All quadratic coefficients of cost_op are positive. The inference models "
            f"expect a max-cut operator in the convention H = {MAX_CUT_PRE_FACTOR} * "
            "sum_e w_e Z_i Z_j -- graph_to_operator(graph, "
            f"pre_factor={MAX_CUT_PRE_FACTOR}) -- under which positive edge weights "
            "give negative coefficients. An operator built with the pre_factor=1.0 "
            "default still yields finite angles, but the predicted gammas are wrong. "
            "Pass pre_factor=-0.5, or ignore this warning if your edge weights really "
            "are negative."
        )


def validate_cost_op(cost_op: SparsePauliOp) -> None:
    """Reject operators the models cannot be given as a max-cut graph.

    Each rejection is something that would otherwise produce finite, plausible
    and wrong angles rather than an error:

    * **Non-real coefficients.** The graph conversion takes ``np.real`` of every
      coefficient, so a non-Hermitian operator would be read as a graph with
      the imaginary parts discarded -- a purely imaginary coefficient becomes a
      zero-weight edge.
    * **One-local ``Z`` terms.** The networks take edge weights only; there is
      no input for a local field. ``operator_to_graph`` would turn each such
      term into a self-loop, which the symmetrization then double-counts into
      the node degrees.
    * **Terms above second order.** No graph represents them.

    Raises:
        ValueError: On any of the above.
    """
    imaginary = [
        (str(term.paulis[0]), complex(term.coeffs[0]))
        for term in cost_op
        if abs(complex(term.coeffs[0]).imag) > 1e-12
    ]
    if imaginary:
        raise ValueError(
            f"cost_op has non-real coefficients: {imaginary}. The inference models take a "
            "real-weighted max-cut graph; a complex coefficient would be silently truncated "
            "to its real part."
        )

    one_local, higher_order = [], []
    for term in cost_op:
        order = int(sum(term.paulis[0].z))
        if order == 1:
            one_local.append(str(term.paulis[0]))
        elif order > 2:
            higher_order.append(str(term.paulis[0]))

    if one_local:
        raise ValueError(
            f"cost_op has one-local Z terms: {one_local}. The inference models predict angles "
            "for a max-cut graph and have no input for a local field, so such a term cannot be "
            "represented: it would become a self-loop and distort the node degrees the model "
            "reads. Drop the linear terms, or use a trainer that supports them."
        )
    if higher_order:
        raise ValueError(
            f"cost_op has terms above second order: {higher_order}. The inference models take a "
            "quadratic max-cut operator, H = -0.5 * sum_e w_e Z_i Z_j."
        )


class AIFeatureExtractor:
    """Extract and normalize the features the ONNX models consume.

    Example:
        extractor = AIFeatureExtractor(in_features=["num_nodes", "mean_degree"])
        x_vec, features = extractor.extract_and_pack_np(cost_op)
    """

    def __init__(
        self,
        in_features: list[str],
        norm_stats: dict[str, dict[str, float]],
    ) -> None:
        """Initialize the feature extractor.

        Args:
            in_features: Names of the features to extract, in the order the
                model's input vector expects them.
            norm_stats: Per-feature ``{"mean": ..., "std": ...}`` normalization
                statistics, taken from the bundle's own training set.
        """
        self.in_features = list(in_features)
        self.norm_stats = norm_stats
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

    def rescaling_factor(self, cost_op: SparsePauliOp) -> float:
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

        factor = float(np.sqrt(factor))
        return factor if factor > 0 else 1.0

    def extract_np(self, cost_op: SparsePauliOp) -> dict[str, Any]:
        """Extract raw (unnormalized) features from a cost operator as numpy.

        Rescales the operator by ``rescaling_factor``, converts to a graph, and
        computes the same scalar/graph features used during training:
            edges         (1, M, 2) int64
            edge_weights  (1, M)    float32
            node_count    (1,)      int64
            nodes, rescale_a

        ``cost_op`` must be a max-cut operator in the training convention
        ``H = -0.5 * sum_e w_e Z_i Z_j`` (see :data:`MAX_CUT_PRE_FACTOR`): the
        edge weights the models were trained on are recovered by dividing the
        coefficients by that pre-factor. An operator built differently predicts
        finite but wrong angles, so the obvious case of it is warned about.

        The operator is simplified first, so two spellings of the same
        Hamiltonian (an un-combined ``sum(per_edge_terms)`` and its simplified
        form) give the same features rather than different gammas, and any
        identity term is dropped. Operators that cannot be a max-cut graph at
        all are rejected by :func:`validate_cost_op`.
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
        warn_on_unexpected_sign_convention(cost_op)

        rescale_a = self.rescaling_factor(cost_op)
        # include_one_local is off because validate_cost_op has already refused
        # one-local terms: a self-loop here would be double-counted into the
        # degrees by the symmetrization in onnx_inputs.
        graph = operator_to_graph(cost_op / rescale_a, include_one_local=False)

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
                    # sigma distance is not defined. Normalizing by 1.0 keeps
                    # the value finite; report the difference as such rather
                    # than dressing it up as a number of standard deviations.
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
            warn_at_caller(
                "This problem is outside the distribution the model was trained on: "
                + "; ".join(extrapolated)
                + f". The threshold is {OUT_OF_DISTRIBUTION_SIGMA:g} sigma, taken from the "
                "bundle's own feature_normalization. The predicted angles are an "
                "extrapolation; treat them as a starting point for optimization rather "
                "than as trained-quality parameters."
            )
        return np.asarray(values, dtype=np.float32)[np.newaxis, ...]

    def extract_and_pack_np(self, cost_op: SparsePauliOp) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract and pack features in one call. Returns ``(x_vec, features)``."""
        features = self.extract_np(cost_op)
        x_vec = self.pack_features_np(features)
        return x_vec, features
