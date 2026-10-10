"""Numpy feed builders for the ONNX runtime.

Each exported model declares its own set of graph inputs, so each needs its own
translation from the extracted features into a feed dict. This module holds one
``prepare_<architecture>(features) -> dict[str, np.ndarray]`` per architecture
and the :data:`INPUT_BUILDERS` registry that
:class:`~qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor`
dispatches through, keyed by the config's ``model_init.model_type``.

``features`` is what ``AIFeatureExtractor.extract_np`` returns, with the packed
scalar vector added under ``"x"``. The feed's keys must match the input names of
the exported graph; the predictor drops any the graph does not declare, so a
builder may offer an input the export happens to omit.

Three conventions matter, and they differ per architecture because they follow
how each model was trained:

* The feature extractor emits a *batched* edge list, ``edges`` of shape
  ``(1, M, 2)`` with ``edge_weights`` of shape ``(1, M)``. Inference is always a
  single graph, so the message-passing builders unpack that leading dimension
  (see :func:`_graph_tensors`) while the transformer builders pass the batched
  arrays straight through.
* The transformer exports consume the edge list as given — padded and
  *un*symmetrized. The message-passing exports need both directions of every
  edge, and no zero-weight ones.
* Anything an ONNX graph cannot compute is precomputed here. That is why the
  graph transformer's Laplacian positional encoding and the GCN's z-scored
  degrees are numpy in this module rather than ops in the export.

Dtypes are not negotiable: onnxruntime rejects a feed whose arrays do not match
the declared types exactly, so indices are cast to ``int64`` and values to
``float32`` on the way out.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

# The extracted features, and the signature every builder in the registry has.
Features = dict[str, Any]
InputBuilder = Callable[[Features], dict[str, np.ndarray]]

# Below this, a per-graph spread of node degrees counts as no spread at all and
# the z-score falls back to a unit divisor instead of dividing by ~zero.
_MIN_DEGREE_STD = 1e-6

# Laplacian eigenvectors to use when the model config does not say.
_DEFAULT_POS_ENC_DIM = 8


class GraphTensors(NamedTuple):
    """One undirected graph, in the layout the message-passing exports expect.

    Attributes:
        edge_index: Both directions of every edge, shape ``(2, 2M)``, ``int64``.
        edge_weight: The matching weights, flat, shape ``(2M,)``, ``float32``.
        num_nodes: Node count, including any node left without an edge.
    """

    edge_index: np.ndarray
    edge_weight: np.ndarray
    num_nodes: int

    @property
    def edge_attr(self) -> np.ndarray:
        """``edge_weight`` as a column, shape ``(2M, 1)``.

        What the GNN and GIN exports declare, their ``edge_dim`` being 1. The
        GCN and graph transformer take the flat form instead.
        """
        return self.edge_weight.reshape(-1, 1)


def _graph_tensors(features: Features) -> GraphTensors:
    """Unpack the single graph in ``features`` into message-passing form.

    Zero-weight edges are dropped: a zero coupling is not an edge, and keeping
    it would inflate every degree the models derive from ``edge_index``. The
    survivors are then symmetrized, since the exports aggregate over incoming
    edges only and would otherwise see each edge from one side.
    """
    # Inference is one graph at a time, so drop the extractor's batch dimension.
    edges = np.asarray(features["edges"])[0]
    weights = np.asarray(features["edge_weights"])[0]

    is_real_edge = weights != 0.0
    edges = edges[is_real_edge]
    weights = weights[is_real_edge]

    # Each edge again as (target, source), so both endpoints receive messages.
    if edges.size:
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        weights = np.concatenate([weights, weights], axis=0)

    return GraphTensors(
        edge_index=edges.T.astype(np.int64),
        edge_weight=weights.astype(np.float32),
        num_nodes=int(np.asarray(features["node_count"]).reshape(-1)[0]),
    )


def _node_degrees(graph: GraphTensors) -> np.ndarray:
    """Per-node degree, counted from the symmetrized ``edge_index``."""
    return np.bincount(graph.edge_index[0], minlength=graph.num_nodes).astype(np.float32)


def _scalar_features(features: Features) -> np.ndarray:
    """The packed vector of graph-level scalars, which every export consumes."""
    return np.asarray(features["x"], dtype=np.float32)


def prepare_mlp(features: Features) -> dict[str, np.ndarray]:
    """Feed the aggregate model (``agg_transformer``).

    It sees no graph structure at all — only the scalar feature vector.
    """
    return {"x": _scalar_features(features)}


def prepare_edge_transformer(features: Features) -> dict[str, np.ndarray]:
    """Feed the edge transformer.

    It attends over the padded edge list as the extractor produced it, so the
    edges are passed through without symmetrizing and without dropping
    zero-weight entries — the padding is part of what it was trained on.
    ``edge_weights`` is offered only when present, since the unweighted export
    does not declare it.
    """
    feed = {
        "x": _scalar_features(features),
        "edges": np.asarray(features["edges"], dtype=np.int64),
    }

    edge_weights = features.get("edge_weights")
    if edge_weights is not None:
        feed["edge_weights"] = np.asarray(edge_weights, dtype=np.float32)
    return feed


def prepare_diffusion_transformer(features: Features) -> dict[str, np.ndarray]:
    """Feed the diffusion transformer, which takes the edge transformer's inputs.

    Its sampling timestep is not an input: the export bakes in evaluation at
    ``t = 0`` with no noise, so the ``t`` the extractor provides is unused.
    """
    return prepare_edge_transformer(features)


def prepare_gnn(features: Features) -> dict[str, np.ndarray]:
    """Feed the message-passing models (``graph_neural_network``, ``gin``).

    Both share this builder: they differ in how they aggregate, not in what
    they consume.
    """
    graph = _graph_tensors(features)
    return {
        "x": _scalar_features(features),
        "edge_index": graph.edge_index,
        "edge_attr": graph.edge_attr,
        "node_count": np.asarray(features["node_count"], dtype=np.int64),
    }


def prepare_gcn(features: Features) -> dict[str, np.ndarray]:
    """Feed the GCN.

    Its export reuses the trained ``GraphConv`` modules, so the per-node input
    has to arrive exactly as ``GCNModel.forward`` built it: the node degree,
    z-scored across the graph. That normalization is numpy here rather than ops
    in the graph, so it has to match training — including the unbiased
    (``ddof=1``) standard deviation.
    """
    graph = _graph_tensors(features)
    degrees = _node_degrees(graph)

    mean_degree = degrees.mean() if degrees.size else 0.0
    # A regular graph has no spread in its degrees, and a single node has no
    # unbiased estimate of one; either way, normalize by 1 and leave the
    # centred degrees as they are instead of producing inf or NaN.
    degree_std = degrees.std(ddof=1) if degrees.size > 1 else np.float32("nan")
    if not np.isfinite(degree_std) or degree_std < _MIN_DEGREE_STD:
        degree_std = 1.0

    node_x = ((degrees - mean_degree) / degree_std).reshape(-1, 1).astype(np.float32)

    return {
        "node_x": node_x,
        "edge_index": graph.edge_index,
        "edge_weight": graph.edge_weight,
        "x": _scalar_features(features),
    }


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)``: compress a signed magnitude, keeping its sign.

    Edge weights may be negative, so the plain ``log1p`` used for degrees would
    be undefined on a weighted degree.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def _degree_node_features(graph: GraphTensors) -> np.ndarray:
    """The graph transformer's node features: degree and weighted degree.

    Two columns, ``[log1p(degree), signed_log1p(weighted_degree)]``, both
    compressed because degrees grow with graph size while the attention layers
    expect inputs of a settled scale.

    Returns:
        Array of shape ``(num_nodes, 2)``, ``float32``.
    """
    degrees = _node_degrees(graph)

    weighted_degrees = np.zeros(graph.num_nodes, dtype=np.float32)
    if graph.edge_index.size:
        # Scatter-add, since a node appears once per incident edge.
        np.add.at(weighted_degrees, graph.edge_index[0], graph.edge_weight)

    return np.stack(
        [np.log1p(degrees), _signed_log1p(weighted_degrees)],
        axis=-1,
    ).astype(np.float32)


def _laplacian_positional_encoding(graph: GraphTensors, pos_enc_dim: int) -> np.ndarray:
    """Eigenvectors of the normalized Laplacian, as a positional encoding.

    Attention is permutation-invariant, so the graph transformer needs a notion
    of where a node sits. The low eigenvectors of the Laplacian supply it. There
    is no ONNX op for an eigendecomposition, so it is computed here and fed in
    as an ordinary input.

    The first eigenvector is dropped: it is constant on a connected graph and
    carries no positional information. The rest are sign-ambiguous, and not even
    unique within a degenerate eigenspace, so two eigensolvers can disagree on
    this array. The model was trained with random sign-flip augmentation and
    tolerates that; it is also why the predictions are only reproducible to a
    loose tolerance.

    Returns:
        Array of shape ``(num_nodes, pos_enc_dim)``, ``float32``, zero-padded
        when the graph has fewer eigenvectors than requested.
    """
    adjacency = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float32)
    if graph.edge_index.size:
        adjacency[graph.edge_index[0], graph.edge_index[1]] = 1.0

    # Symmetric normalization, L = I - D^-1/2 A D^-1/2. Degrees are clipped at
    # 1 so an isolated node contributes no division by zero.
    degrees = adjacency.sum(axis=1)
    inv_sqrt_degrees = np.clip(degrees, 1.0, None) ** -0.5
    normalized_adjacency = inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]
    laplacian = np.eye(graph.num_nodes, dtype=np.float32) - normalized_adjacency

    # float64 for the solve: eigenvectors of a near-degenerate spectrum are
    # sensitive, and this runs once per prediction.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.astype(np.float64))
    # eigh already returns ascending eigenvalues; sorting states the dependency.
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]

    pos_enc = eigenvectors[:, 1 : pos_enc_dim + 1]
    missing_columns = pos_enc_dim - pos_enc.shape[1]
    if missing_columns > 0:
        # A graph smaller than pos_enc_dim has too few eigenvectors; pad so the
        # export's fixed input width still holds.
        pos_enc = np.pad(pos_enc, ((0, 0), (0, missing_columns)))
    return pos_enc.astype(np.float32)


def prepare_graph_transformer(features: Features) -> dict[str, np.ndarray]:
    """Feed the graph transformer.

    Both of its per-node inputs are precomputed here — the degree features and
    the Laplacian positional encoding — because neither is expressible in the
    exported graph.
    """
    graph = _graph_tensors(features)
    pos_enc_dim = int(features.get("pos_enc_dim", _DEFAULT_POS_ENC_DIM))

    return {
        "x": _scalar_features(features),
        "node_features": _degree_node_features(graph),
        "pos_enc": _laplacian_positional_encoding(graph, pos_enc_dim),
        "edge_index": graph.edge_index,
        "edge_weight": graph.edge_weight,
    }


# Keyed by the ``model_init.model_type`` a bundle's config declares, which is
# what the predictor looks up. Note the two names that do not match their
# builder: ``agg_transformer`` is the ``mlp`` bundle, and GIN shares the GNN's
# builder. A model_type absent here cannot be loaded.
INPUT_BUILDERS: dict[str, InputBuilder] = {
    "agg_transformer": prepare_mlp,
    "edge_transformer": prepare_edge_transformer,
    "diffusion_transformer": prepare_diffusion_transformer,
    "graph_neural_network": prepare_gnn,
    "graph_isomorphism_network": prepare_gnn,
    "gcn": prepare_gcn,
    "graph_transformer": prepare_graph_transformer,
}
