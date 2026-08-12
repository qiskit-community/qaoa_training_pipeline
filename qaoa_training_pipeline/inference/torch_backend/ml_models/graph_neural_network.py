import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """Simple edge-aware message passing block using an edge list."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run one message-passing step on a single graph.

        Args:
            h: Node embeddings of shape (N, embed_dim).
            edge_index: Edge indices of shape (2, M) with [src, dst].
            edge_emb: Optional edge embeddings of shape (M, embed_dim).

        Returns:
            Updated node embeddings of shape (N, embed_dim).
        """
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
            return self.update_mlp(torch.cat([h, agg], dim=-1))

        src, dst = edge_index[0], edge_index[1]
        x_src = h[src]
        x_dst = h[dst]

        if edge_weight is not None:
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)

            if edge_weight.shape[1] != 1:
                raise ValueError(
                    "Expected exactly one scalar weight per edge, "
                    f"got shape {tuple(edge_weight.shape)}"
                )

            x_src = x_src * edge_weight

        msg_in = torch.cat([x_dst, x_src], dim=-1)
        msg = self.msg_mlp(msg_in)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        return self.update_mlp(torch.cat([h, agg], dim=-1))


class GNNRegression(nn.Module):
    """
    Graph-level regressor for arbitrary scalar, node, and edge features.

    Runtime inputs:
        - graph_features: Tensor of shape (B, F_graph)
        - edges: list of per-graph edge specifications
        - node_features: optional list of per-graph node feature tensors
        - edge_features: optional list of per-graph edge feature tensors

    The model dynamically adapts its input projection layers on first use,
    based on the actual number of scalar, node, and edge features requested
    from the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_dim: Optional[int] = None,
        embed_dim: int = 64,
        num_layers: int = 4,
        node_feature_dim: int = 0,
        droput: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.node_feature_dim = node_feature_dim

        if input_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.global_embedding_const = None
        else:
            self.global_proj = None
            self.global_embedding_const = nn.Parameter(torch.zeros(embed_dim))

        node_fuse_in = node_feature_dim + embed_dim
        self.node_fuse = nn.Sequential(
            nn.Linear(node_fuse_in, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.layers = nn.ModuleList([MessagePassingLayer(embed_dim) for _ in range(num_layers)])
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim + input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(droput),
            nn.Linear(embed_dim, output_dim),
        )

    @staticmethod
    def _normalize_edge_item(
        item: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """
        Normalize one edge specification into edge indices and optional attributes.

        Args:
            item: Edge specification for one graph.
            device: Target device.
            dtype: Target dtype for edge attributes.

        Returns:
            Tuple of:
                - edge_index with shape (2, M)
                - edge_attr with shape (M, D_edge), or None
        """
        edge_index_raw: Any = None
        edge_attr_raw: Any = None

        if isinstance(item, dict):
            edge_index_raw = item.get("edge_index", item.get("index", None))
            edge_attr_raw = item.get("edge_attr", item.get("attr", None))
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and (torch.is_tensor(item[0]) or isinstance(item[0], list))
        ):
            edge_index_raw, edge_attr_raw = item[0], item[1]
        else:
            edge_index_raw = item

        if isinstance(edge_index_raw, list):
            edge_index_tensor = torch.tensor(
                edge_index_raw,
                dtype=torch.long,
                device=device,
            )
        elif edge_index_raw is None:
            edge_index_tensor = torch.empty(2, 0, dtype=torch.long, device=device)
        elif torch.is_tensor(edge_index_raw):
            edge_index_tensor = edge_index_raw.to(device=device, dtype=torch.long)
        else:
            raise ValueError(
                f"Unsupported edge specification type: {type(edge_index_raw).__name__}"
            )

        if edge_index_tensor.numel() == 0:
            edge_index_tensor = torch.empty(2, 0, dtype=torch.long, device=device)
        elif edge_index_tensor.dim() == 2 and edge_index_tensor.shape[0] == 2:
            pass
        elif edge_index_tensor.dim() == 2 and edge_index_tensor.shape[1] == 2:
            edge_index_tensor = edge_index_tensor.t().contiguous()
        else:
            raise ValueError(
                "edge_index must be (2, M) or (M, 2), " f"got {tuple(edge_index_tensor.shape)}"
            )

        edge_attr_tensor: Optional[torch.Tensor]
        if edge_attr_raw is None:
            edge_attr_tensor = None
        else:
            if torch.is_tensor(edge_attr_raw):
                edge_attr_tensor = edge_attr_raw.to(device=device, dtype=dtype)
            else:
                edge_attr_tensor = torch.tensor(edge_attr_raw, dtype=dtype, device=device)

            if edge_attr_tensor.dim() == 1:
                edge_attr_tensor = edge_attr_tensor.unsqueeze(-1)

            if edge_attr_tensor.shape[0] != edge_index_tensor.shape[1]:
                raise ValueError(
                    "edge_attr length "
                    f"{edge_attr_tensor.shape[0]} != #edges {edge_index_tensor.shape[1]}"
                )

        return cast(torch.LongTensor, edge_index_tensor), edge_attr_tensor

    @staticmethod
    def _ensure_2d_feature_tensor(
        features: Optional[torch.Tensor],
        expected_rows: int,
        device: torch.device,
        dtype: torch.dtype,
        feature_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Convert feature tensors into shape (N, F) or (M, F).

        Args:
            features: Input feature tensor or None.
            expected_rows: Required first dimension.
            device: Target device.
            dtype: Target dtype.
            feature_name: Feature group name for error messages.

        Returns:
            Feature tensor with shape (expected_rows, F), or None.
        """
        if features is None:
            return None

        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=dtype, device=device)
        else:
            features = features.to(device=device, dtype=dtype)

        if features.dim() == 1:
            features = features.unsqueeze(-1)
        elif features.dim() != 2:
            raise ValueError(
                f"{feature_name} must be 1D or 2D per graph, got shape {tuple(features.shape)}"
            )

        if features.shape[0] != expected_rows:
            raise ValueError(
                f"{feature_name} row count {features.shape[0]} != expected {expected_rows}"
            )

        return features

    def _infer_num_nodes(
        self,
        graph_features_row: torch.Tensor,
        edge_index: torch.LongTensor,
        node_features: Optional[torch.Tensor],
    ) -> int:
        """
        Infer the number of nodes for a single graph.

        Args:
            graph_features_row: Graph-level feature row of shape (F_graph,).
            edge_index: Edge indices of shape (2, M).
            node_features: Optional node features of shape (N, F_node).

        Returns:
            Number of nodes in the graph.
        """
        if node_features is not None:
            return int(node_features.shape[0])

        if edge_index.numel() > 0:
            return int(edge_index.max().item()) + 1

        if graph_features_row.numel() > 0:
            inferred = int(round(float(graph_features_row[0].item())))
            if inferred > 0:
                return inferred

        raise ValueError(
            "Unable to infer number of nodes. Provide node features, non-empty edges, "
            "or ensure the first scalar feature is num_nodes."
        )

    def _build_node_features(
        self,
        graph_features_row: torch.Tensor,
        edge_index: torch.LongTensor,
        node_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build initial node embeddings for one graph.

        Args:
            graph_features_row: Graph-level features of shape (F_graph,).
            edge_index: Edge indices of shape (2, M).
            node_features: Optional node features of shape (N, F_node).
            num_nodes: Optional number of nodes. If provided, used as authoritative source.

        Returns:
            Node embedding tensor of shape (N, embed_dim).
        """
        device = graph_features_row.device
        dtype = graph_features_row.dtype

        if num_nodes is None:
            num_nodes = self._infer_num_nodes(graph_features_row, edge_index, node_features)

        if self.global_proj is not None:
            global_embedding = self.global_proj(graph_features_row.unsqueeze(0))
        else:
            # Zero global features: use the learnable constant embedding.
            global_embedding = self.global_embedding_const.to(device=device, dtype=dtype).unsqueeze(
                0
            )
        global_embedding = global_embedding.expand(num_nodes, -1)

        parts = [global_embedding]
        if node_features is not None:
            parts.insert(0, node_features)

        node_raw = torch.cat(parts, dim=-1)
        return self.node_fuse(node_raw)

    def forward(
        self,
        x: torch.Tensor,
        edges: List[
            Union[
                Dict[str, torch.Tensor],
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                torch.Tensor,
                List[Tuple[int, int]],
            ]
        ],
        node_count: torch.Tensor,
        node_features: Optional[List[torch.Tensor]] = None,
        edge_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Run graph regression on a batch of graphs.

        Args:
            x: Graph-level scalar features of shape (B, F_graph).
            edges: List of per-graph edge specifications of length B.
            node_count: Tensor of node counts per graph, shape (B,).
                       Used as authoritative source for number of nodes.
            node_features: Optional list of per-graph node feature tensors.
            edge_features: Optional list of per-graph edge feature tensors.

        Returns:
            Tensor of shape (B, output_dim).
        """
        if not torch.is_tensor(x):
            raise ValueError("`x` must be a tensor of shape (B, F_graph).")
        if x.dim() != 2:
            raise ValueError(f"`x` must be 2D, got shape {tuple(x.shape)}")

        batch_size = x.shape[0]
        if len(edges) != batch_size:
            raise ValueError(f"edges length {len(edges)} != batch size {batch_size}")
        if node_count.shape[0] != batch_size:
            raise ValueError(f"node_count length {node_count.shape[0]} != batch size {batch_size}")
        if node_features is not None and len(node_features) != batch_size:
            raise ValueError(
                f"node_features length {len(node_features)} != batch size {batch_size}"
            )
        if edge_features is not None and len(edge_features) != batch_size:
            raise ValueError(
                f"edge_features length {len(edge_features)} != batch size {batch_size}"
            )

        outputs = []
        for batch_index in range(batch_size):
            graph_features_row = x[batch_index]
            device = graph_features_row.device
            dtype = graph_features_row.dtype

            edge_index, edge_attr_from_edges = self._normalize_edge_item(
                edges[batch_index],
                device=device,
                dtype=dtype,
            )
            edge_index = cast(torch.LongTensor, edge_index)

            # Use node_count as authoritative source for number of nodes
            num_nodes = int(node_count[batch_index].item())

            node_features_batch: Optional[torch.Tensor] = None
            if node_features is not None:
                node_features_batch = self._ensure_2d_feature_tensor(
                    node_features[batch_index],
                    expected_rows=num_nodes,
                    device=device,
                    dtype=dtype,
                    feature_name="node_features",
                )

            num_edges = edge_index.shape[1]
            edge_features_batch: Optional[torch.Tensor] = None
            if edge_features is not None:
                edge_features_batch = self._ensure_2d_feature_tensor(
                    edge_features[batch_index],
                    expected_rows=num_edges,
                    device=device,
                    dtype=dtype,
                    feature_name="edge_features",
                )

            if edge_attr_from_edges is not None and edge_features_batch is not None:
                raise ValueError(
                    "Edge weights were provided through both `edges` and "
                    "`edge_features`. Provide exactly one source."
                )

            edge_weight = (
                edge_attr_from_edges if edge_attr_from_edges is not None else edge_features_batch
            )

            if edge_weight is None:
                raise ValueError("Missing edge weights. Unweighted graphs must explicitly use 1.0.")

            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)

            if edge_weight.shape != (num_edges, 1):
                raise ValueError(
                    f"Expected edge weights with shape ({num_edges}, 1), "
                    f"got {tuple(edge_weight.shape)}."
                )

            h = self._build_node_features(
                graph_features_row,
                edge_index,
                node_features_batch,
                num_nodes=num_nodes,
            )

            for layer in self.layers:
                h = layer(h, edge_index, edge_weight)

            graph_embedding = h.mean(dim=0)
            prediction_input = torch.cat(
                [graph_embedding, graph_features_row],
                dim=-1,
            )

            outputs.append(self.output_mlp(prediction_input))

        return torch.stack(outputs, dim=0)
