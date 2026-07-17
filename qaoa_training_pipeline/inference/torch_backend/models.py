"""Per-architecture build and predict functions.

Each supported model type has two small functions here:

- ``build_<name>(cfg)``: construct the ``nn.Module`` from a ``model_init``
  dict (the ``model_init`` block of a model config).
- ``predict_<name>(model, features)``: call the model with the correct
  positional/keyword shape.

The two flat dicts at the bottom — ``builders`` and ``predictors`` — are the
only extension points. To support a new model, add a build/predict pair and
register it in both dicts.

``features`` is a dict produced by the feature extractor and shaped by the
predictor. Keys used here:

    x                (B, F_graph)   Global scalar features.
    edges            (B, M, 2)      Padded edge lists.
    edge_weights     (B, M)         Padded edge weights (may be None).
    node_count       (B,)           Number of nodes per graph.
    t                (B,)           Diffusion timestep (diffusion models only).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


Features = dict[str, Any]
BuildFn = Callable[[dict], nn.Module]
PredictFn = Callable[[nn.Module, Features], torch.Tensor]


# ---------- shared feature-shaping helpers ---------------------------------


def _symmetrize_edges(
    edges_mx2: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add the reverse of every edge so message passing sees an undirected graph.

    Args:
        edges_mx2:   Edge list of shape ``(M, 2)`` = [src, dst] per row.
        edge_weight: Optional per-edge weights, shape ``(M,)`` or ``(M, 1)``.

    Returns:
        ``(edges_2mx2, weight_2m)`` — edges of shape ``(2M, 2)`` with each edge
        followed by its reverse, and weights duplicated to match (or ``None``).
    """
    if edges_mx2.numel() == 0:
        return edges_mx2, edge_weight
    reversed_edges = edges_mx2.flip(1)  # (M, 2): swap [src, dst] -> [dst, src]
    sym_edges = torch.cat([edges_mx2, reversed_edges], dim=0)  # (2M, 2)
    sym_weight = torch.cat([edge_weight, edge_weight], dim=0) if edge_weight is not None else None
    return sym_edges, sym_weight


def edges_with_attr(features: Features) -> list:
    """Same as ``_edges_list`` but each entry is ``(edge_index, edge_attr)``
    when edge weights are available. Edges are symmetrized to match training."""

    edges = features["edges"]
    weights = features.get("edge_weights")
    out = []
    for i in range(edges.shape[0]):
        ed = edges[i]
        if weights is not None:
            ew = weights[i]
            mask = ew != 0.0
            sym_ed, sym_ew = _symmetrize_edges(ed[mask], ew[mask].unsqueeze(-1))
            out.append((sym_ed.t().contiguous(), sym_ew))
        else:
            mask = ~((ed[:, 0] == 0) & (ed[:, 1] == 0))
            sym_ed, _ = _symmetrize_edges(ed[mask])
            out.append(sym_ed.t().contiguous())
    return out


def pyg_batch(features: Features):
    """Assemble a PyG ``Batch`` from per-graph edges, edge weights, and node counts.

    GCN inspects ``data.edge_weight`` — attach it so the message passing is
    weighted when the caller supplied weights.
    """
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import degree

    x, edges, node_count = features["x"], features["edges"], features["node_count"]
    weights = features.get("edge_weights")
    data_list = []
    for i in range(x.shape[0]):
        ed = edges[i]
        if weights is not None:
            ew = weights[i]
            keep = ew != 0.0
            ed_sym, ew_sym = _symmetrize_edges(ed[keep], ew[keep].view(-1))
            edge_weight = ew_sym.view(-1)
        else:
            keep = ~((ed[:, 0] == 0) & (ed[:, 1] == 0))
            ed_sym, _ = _symmetrize_edges(ed[keep])
            edge_weight = None
        edge_index = ed_sym.T
        num_nodes = int(node_count[i])
        deg = degree(index=edge_index[0].long(), num_nodes=num_nodes).unsqueeze(1)
        data_list.append(
            Data(
                x=deg,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=num_nodes,
                features=x[i].unsqueeze(0),
            )
        )
    return Batch.from_data_list(data_list).to(x.device)


# ---------- per-model build/predict pairs ----------------------------------


def build_mlp(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.mlp import MLP

    return MLP(
        input_dim=cfg["input_dim"],
        output_dim=cfg["output_dim"],
        embed_dim=cfg.get("embed_dim", 256),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )


def predict_mlp(model: nn.Module, features: Features) -> torch.Tensor:
    return model(features["x"])


def build_edge_transformer(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.edge_transformer import (
        EdgeTransformer,
    )

    # NOTE: `dim_feedforward` in the config is legacy — main's EdgeTransformer
    # doesn't accept it. Silently ignore.
    return EdgeTransformer(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg.get("embed_dim", 256)),
        n_heads=int(cfg.get("n_heads", 4)),
        num_layers=int(cfg.get("num_layers", 4)),
        edge_embed_dim=int(cfg.get("edge_embed_dim", 32)),
        use_positional_encoding=bool(cfg.get("use_positional_encoding", True)),
    )


def predict_edge_transformer(model: nn.Module, features: Features) -> torch.Tensor:
    return model(features["x"], features["edges"], features.get("edge_weights"))


def build_diffusion_transformer(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.ddpm_transformer import (
        DDPMTransformer,
    )

    # NOTE: `dim_feedforward` is legacy — DDPMTransformer doesn't accept it.
    return DDPMTransformer(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg.get("embed_dim", 256)),
        n_heads=int(cfg.get("n_heads", 4)),
        num_layers=int(cfg.get("num_layers", 4)),
        edge_embed_dim=int(cfg.get("edge_embed_dim", 32)),
        timesteps=int(cfg.get("timesteps", 100)),
    )


def predict_diffusion_transformer(model: nn.Module, features: Features) -> torch.Tensor:
    return model(
        features["x"],
        features["edges"],
        features["t"],
        edge_weights=features.get("edge_weights"),
    )


def build_gcn(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_convolutional_network import (
        GCNModel,
    )

    return GCNModel(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
    )


def predict_gcn(model: nn.Module, features: Features) -> torch.Tensor:
    return model(pyg_batch(features))


def build_gnn(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_neural_network import (
        GNNRegression,
    )

    # LEGACY: configs carry `node_input_dim` — the *fused* node_fuse input width
    # (node_feature_dim + pos_dim + embed_dim), not a per-node feature dim.
    # Main's GNNRegression rebuilds it from `node_feature_dim`; default 0 means
    # the module computes degree-based features.
    return GNNRegression(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg.get("embed_dim", 64)),
        num_layers=int(cfg.get("num_layers", 4)),
        edge_dim=cfg.get("edge_dim"),
        node_feature_dim=int(cfg.get("node_feature_dim", 0)),
    )


def predict_gnn(model: nn.Module, features: Features) -> torch.Tensor:
    return model(features["x"], edges_with_attr(features), node_count=features["node_count"])


def build_gin(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_isomorphism_network import (
        GINRegression,
    )

    # LEGACY: `node_input_dim` in the config is the *fused* node input width,
    # not a per-node feature dim. Ignore it.
    return GINRegression(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg.get("embed_dim", 512)),
        num_layers=int(cfg.get("num_layers", 6)),
        edge_dim=cfg.get("edge_dim"),
        node_feature_dim=int(cfg.get("node_feature_dim", 0)),
    )


def predict_gin(model: nn.Module, features: Features) -> torch.Tensor:
    return model(features["x"], edges_with_attr(features), node_count=features["node_count"])


def build_graph_transformer(cfg: dict) -> nn.Module:
    from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_transformer import (
        GraphTransformer,
    )

    return GraphTransformer(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg.get("embed_dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=int(cfg.get("num_heads", 8)),
        ff_dim=int(cfg.get("ff_dim", 512)),
        pos_enc_dim=int(cfg.get("pos_enc_dim", 8)),
        node_feature_dim=int(cfg.get("node_feature_dim", 2)),
    )


def edges_unpadded_with_weights(features: Features):
    """Return per-graph edges of shape ``(M_i, 2)`` plus matching edge-weight
    tensors of shape ``(M_i,)``. GT transposes edges internally.
    """
    edges = features["edges"]
    weights = features.get("edge_weights")
    edges_out = []
    weights_out = None if weights is None else []
    for i in range(edges.shape[0]):
        ed = edges[i]
        if weights is not None:
            ew = weights[i]
            mask = ew != 0.0
            edges_out.append(ed[mask].contiguous())
            weights_out.append(ew[mask])
        else:
            mask = ~((ed[:, 0] == 0) & (ed[:, 1] == 0))
            edges_out.append(ed[mask].contiguous())
    return edges_out, weights_out


def predict_graph_transformer(model: nn.Module, features: Features) -> torch.Tensor:

    edge_list, edge_weights = edges_unpadded_with_weights(features)
    # Match training forward(): symmetrize each graph's edges (and weights) so the
    # degree-based node features and Laplacian PE see an undirected graph.
    if edge_weights is None:
        sym = [_symmetrize_edges(e)[0] for e in edge_list]
        edge_list, sym_weights = sym, None
    else:
        pairs = [_symmetrize_edges(e, w) for e, w in zip(edge_list, edge_weights)]
        edge_list = [e for e, _ in pairs]
        sym_weights = [w for _, w in pairs]
    return model(
        features["x"],
        edge_list,
        features["node_count"],
        edge_weights=sym_weights,
    )


builders: dict[str, BuildFn] = {
    "agg_transformer": build_mlp,
    "edge_transformer": build_edge_transformer,
    "diffusion_transformer": build_diffusion_transformer,
    "gcn": build_gcn,
    "graph_neural_network": build_gnn,
    "graph_isomorphism_network": build_gin,
    "graph_transformer": build_graph_transformer,
}

predictors: dict[str, PredictFn] = {
    "agg_transformer": predict_mlp,
    "edge_transformer": predict_edge_transformer,
    "diffusion_transformer": predict_diffusion_transformer,
    "gcn": predict_gcn,
    "graph_neural_network": predict_gnn,
    "graph_isomorphism_network": predict_gin,
    "graph_transformer": predict_graph_transformer,
}
