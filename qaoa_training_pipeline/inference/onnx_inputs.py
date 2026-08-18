"""Numpy input preparation for the ONNX runtime.

Each ``prepare_<model>(features) -> dict[str, np.ndarray]`` builds the plain-numpy
feed dict for ``onnxruntime.InferenceSession.run``. The exported ONNX graph's
input names must match the keys produced here.

``features`` is the dict returned by ``AIFeatureExtractor.extract_np`` with the
packed ``x`` added under key ``"x"``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Features = dict[str, Any]
PrepareFn = Callable[[Features], dict[str, np.ndarray]]


# ---------- shared numpy helpers ------------------


def _symmetrize_edges_np(
    edges_mx2: np.ndarray,
    edge_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Add the reverse of every edge (symmetrize an undirected edge list)."""
    if edges_mx2.size == 0:
        return edges_mx2, edge_weight
    reversed_edges = edges_mx2[:, ::-1]
    sym_edges = np.concatenate([edges_mx2, reversed_edges], axis=0)
    sym_weight = (
        np.concatenate([edge_weight, edge_weight], axis=0) if edge_weight is not None else None
    )
    return sym_edges, sym_weight


# ---------- per-model input builders ---------------------------------------


def prepare_mlp(features: Features) -> dict[str, np.ndarray]:
    """agg_transformer: only the scalar feature vector."""
    return {"x": np.asarray(features["x"], dtype=np.float32)}


def prepare_edge_transformer(features: Features) -> dict[str, np.ndarray]:
    """edge_transformer: padded edges + weights passed through (no symmetrize),
    exactly like ``predict_edge_transformer``."""
    feed = {
        "x": np.asarray(features["x"], dtype=np.float32),
        "edges": np.asarray(features["edges"], dtype=np.int64),
    }
    ew = features.get("edge_weights")
    if ew is not None:
        feed["edge_weights"] = np.asarray(ew, dtype=np.float32)
    return feed


def prepare_diffusion_transformer(features: Features) -> dict[str, np.ndarray]:
    """diffusion_transformer: same as edge; eval t=0/no-noise is baked into the
    exported graph, so no ``t`` input."""
    return prepare_edge_transformer(features)


def _edges_with_attr_single(features: Features) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(edge_index (2, 2M), edge_attr (2M, 1))`` for the single-graph
    (B=1) case — symmetrized, with zero-weight edges dropped."""
    ed = np.asarray(features["edges"])[0]  # (M, 2)
    ew = np.asarray(features["edge_weights"])[0]  # (M,)
    mask = ew != 0.0
    ed_m = ed[mask]
    ew_m = ew[mask].reshape(-1, 1)  # (M', 1)
    sym_ed, sym_ew = _symmetrize_edges_np(ed_m, ew_m)
    edge_index = sym_ed.T.astype(np.int64)  # (2, 2M')
    edge_attr = sym_ew.astype(np.float32)  # (2M', 1)
    return edge_index, edge_attr


def prepare_gnn(features: Features) -> dict[str, np.ndarray]:
    """graph_neural_network / graph_isomorphism_network: fixed tensor inputs for
    the exported ONNX graph."""
    edge_index, edge_attr = _edges_with_attr_single(features)
    return {
        "x": np.asarray(features["x"], dtype=np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_count": np.asarray(features["node_count"], dtype=np.int64),
    }


def prepare_gcn(features: Features) -> dict[str, np.ndarray]:
    """gcn: the trained GraphConv modules are reused inside the export wrapper,
    so we pass symmetrized ``edge_index``/``edge_weight`` plus the per-node
    z-scored degree feature (computed here in numpy, matching GCNModel.forward's
    per-graph z-score) and the global scalar features ``x``.
    """
    edge_index, edge_attr = _edges_with_attr_single(features)  # (2,2M), (2M,1)
    edge_weight = edge_attr.reshape(-1).astype(np.float32)  # (2M,)
    num_nodes = int(np.asarray(features["node_count"]).reshape(-1)[0])

    # Node degree from the symmetrized edge_index (count of edge_index[0]).
    deg = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)
    mean = deg.mean() if deg.size else 0.0
    # Use the unbiased (Bessel, ddof=1) std estimator to match training.
    std = deg.std(ddof=1) if deg.size > 1 else np.float32("nan")
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    node_x = ((deg - mean) / std).reshape(-1, 1).astype(np.float32)  # (N,1)

    return {
        "node_x": node_x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "x": np.asarray(features["x"], dtype=np.float32),
    }


def _signed_log1p_np(x: np.ndarray) -> np.ndarray:
    """Signed log1p: ``sign(x) * log1p(|x|)``."""
    return np.sign(x) * np.log1p(np.abs(x))


def _laplacian_pe_np(edge_index: np.ndarray, num_nodes: int, pos_enc_dim: int) -> np.ndarray:
    """Laplacian positional encoding via eigendecomposition.

    The eigendecomposition has no ONNX op, so we bake it into the numpy feed and
    pass it as a graph input. Laplacian eigenvectors are sign-ambiguous (and
    non-unique inside degenerate eigenspaces); the model was trained with random
    sign-flip augmentation, so it tolerates eigensolver differences.
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edge_index.size:
        adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.clip(deg, 1.0, None) ** -0.5
    norm_adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    laplacian = np.eye(num_nodes, dtype=np.float32) - norm_adj

    # eigh returns ascending eigenvalues; argsort keeps the ordering explicit.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.astype(np.float64))
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    pos_enc = eigenvectors[:, 1 : pos_enc_dim + 1]
    if pos_enc.shape[1] < pos_enc_dim:
        pos_enc = np.pad(pos_enc, ((0, 0), (0, pos_enc_dim - pos_enc.shape[1])))
    return pos_enc.astype(np.float32)


def _gt_default_node_features_np(
    edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Default node features: ``[log1p(deg), signed_log1p(weighted_deg)]``."""
    deg = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)
    weighted_deg = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.size:
        np.add.at(weighted_deg, edge_index[0], edge_weight.reshape(-1))
    return np.stack([np.log1p(deg), _signed_log1p_np(weighted_deg)], axis=-1).astype(np.float32)


def prepare_graph_transformer(features: Features) -> dict[str, np.ndarray]:
    """graph_transformer: precompute the degree node features and Laplacian
    positional encoding in numpy (eigendecomposition is not ONNX-friendly), then
    feed them plus the symmetrized edge_index/edge_weight and the global features
    to the exported ONNX graph."""
    edge_index, edge_attr = _edges_with_attr_single(features)  # (2,2M), (2M,1)
    edge_weight = edge_attr.reshape(-1).astype(np.float32)  # (2M,)
    num_nodes = int(np.asarray(features["node_count"]).reshape(-1)[0])

    pos_enc_dim = int(features.get("pos_enc_dim", 8))
    node_features = _gt_default_node_features_np(edge_index, edge_weight, num_nodes)
    pos_enc = _laplacian_pe_np(edge_index, num_nodes, pos_enc_dim)

    return {
        "x": np.asarray(features["x"], dtype=np.float32),
        "node_features": node_features,
        "pos_enc": pos_enc,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }


# Registry — keyed by model_type, mirrors models.predictors dispatch.
numpy_input_builders: dict[str, PrepareFn] = {
    "agg_transformer": prepare_mlp,
    "edge_transformer": prepare_edge_transformer,
    "diffusion_transformer": prepare_diffusion_transformer,
    "graph_neural_network": prepare_gnn,
    "graph_isomorphism_network": prepare_gnn,
    "gcn": prepare_gcn,
    "graph_transformer": prepare_graph_transformer,
}
