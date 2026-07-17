import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, MessagePassing
from typing import List, Tuple, Dict, Optional, Union


class EdgeConditionedGINConv(MessagePassing):
    """
    Edge-conditioned GIN layer that uses edge weights in message passing.
    Messages are modulated by edge features before aggregation.
    """

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super().__init__(aggr="add")
        self.nn = mlp
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, embed_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply MLP to aggregated messages + self-connection
        out = self.nn((1 + self.eps) * x + out)

        return out

    def message(self, x_j, edge_attr):
        """
        Compute messages from neighbors, modulated by edge features.

        Args:
            x_j: Neighbor node features [num_edges, embed_dim]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.shape[1] != 1:
                raise ValueError(
                    "GINRegression currently expects exactly one scalar edge weight, "
                    f"got {tuple(edge_attr.shape)}."
                )
            return x_j * edge_attr
        else:
            return x_j


class GINRegression(nn.Module):
    """
    Edge-conditioned GIN graph regressor
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        edge_dim: Optional[int] = None,
        embed_dim: int = 64,
        num_layers: int = 4,
        node_feature_dim: int = 0,
        use_gdv_features: bool = False,
        use_topology_encoding: bool = False,
        pooling: str = "mean",
        dropout: float = 0.05,
        jk_mode: Optional[str] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.node_feature_dim = node_feature_dim
        self.use_gdv_features = use_gdv_features
        self.use_topology_encoding = use_topology_encoding
        self.pooling = pooling
        self.dropout = dropout
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode

        self.global_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        node_fuse_in = embed_dim + node_feature_dim
        self.node_fuse = nn.Sequential(
            nn.Linear(embed_dim + node_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # GDV feature projection (if using GDV)
        self.gdv_proj = None
        if use_gdv_features:
            # GDV features are 16-dimensional, project to embed_dim
            self.gdv_proj = nn.Sequential(
                nn.Linear(16, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.ReLU(),
            )

        # Edge-conditioned GIN layers with GraphNorm and 2x expansion
        self.gin_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            mlp = Sequential(
                Linear(embed_dim, embed_dim * 2),
                ReLU(),
                nn.Dropout(dropout),
                Linear(embed_dim * 2, embed_dim),
            )
            # Use edge-conditioned GIN if edge features are provided
            if edge_dim is not None:
                self.gin_layers.append(EdgeConditionedGINConv(mlp, eps=0.0, train_eps=True))
            else:
                self.gin_layers.append(GINConv(mlp, eps=0.0, train_eps=True))
            self.layer_norms.append(nn.LayerNorm(embed_dim))

        # Pre-pooling projection with GraphNorm
        self.pre_pool_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim + input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    @staticmethod
    def _normalize_edge_item(item, device, dtype=torch.float32):
        """Normalize edge specification into standard format."""
        edge_index = None
        edge_attr = None

        if isinstance(item, dict):
            edge_index = item.get("edge_index", item.get("index", None))
            edge_attr = item.get("edge_attr", item.get("attr", None))
        elif (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and (torch.is_tensor(item[0]) or isinstance(item[0], list))
        ):
            edge_index, edge_attr = item[0], item[1]
        else:
            edge_index = item

        if isinstance(edge_index, list):
            if len(edge_index) == 0:
                edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device)
            if edge_index.dim() == 2:
                if edge_index.shape[0] == 2:
                    pass
                elif edge_index.shape[1] == 2:
                    edge_index = edge_index.t().contiguous()
                else:
                    raise ValueError(f"edge_index must be (2, M) or (M, 2)")

        if edge_attr is not None:
            if not torch.is_tensor(edge_attr):
                edge_attr = torch.tensor(edge_attr, dtype=dtype, device=device)
            else:
                edge_attr = edge_attr.to(device=device, dtype=dtype)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"edge_attr length {edge_attr.shape[0]} does not match "
                    f"number of edges {edge_index.shape[1]}."
                )

        return edge_index.long(), edge_attr

    def _build_node_features(
        self,
        agg_row: torch.Tensor,
        N: int,
        node_features: Optional[torch.Tensor] = None,
        gdv_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build node features from graph statistics and structure.

        Args:
            agg_row: Aggregated graph-level features of shape (F_graph,)
            N: Number of nodes
            node_features: Pre-computed node features (num_nodes, num_features) or None
                          Should contain degree_zscore and/or triangle features if enabled
            gdv_features: Pre-computed GDV features (num_nodes, num_features) or None
        Returns:
            Node embedding tensor of shape (N, embed_dim)
        """
        device = agg_row.device
        dtype = agg_row.dtype

        # Global embedding from ALL graph-level features
        g = self.global_proj(agg_row.unsqueeze(0))
        g = g.expand(N, -1)

        # Build features: [global + optional node_features + optional gdv]
        parts = [g]

        # Add pre-computed node features if provided
        if node_features is not None:
            node_feats = node_features.to(device=device, dtype=dtype)
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(-1)
            if node_feats.shape[0] != N:
                raise ValueError(
                    f"Node features have {node_feats.shape[0]} nodes but graph has {N} nodes"
                )
            parts.append(node_feats)

        if self.use_gdv_features:
            # Use pre-computed GDV features (required - no on-the-fly computation)
            if gdv_features is None:
                raise ValueError(
                    "GDV features are enabled but not provided. "
                    "Pre-computed GDV features must be passed to the model."
                )

            gdv_feats = gdv_features.to(device=device, dtype=dtype)
            # Ensure correct number of nodes
            if gdv_feats.shape[0] != N:
                raise ValueError(
                    f"GDV features have {gdv_feats.shape[0]} nodes but graph has {N} nodes"
                )

            # Apply projection if configured
            if self.gdv_proj is not None:
                gdv_feats = self.gdv_proj(gdv_feats)
            parts.append(gdv_feats)

        node_raw = torch.cat(parts, dim=-1)
        h0 = self.node_fuse(node_raw)

        return h0

    def _pool_graph(self, h):
        """Apply graph-level pooling."""
        if self.pooling == "add":
            return h.sum(dim=0)
        elif self.pooling == "mean":
            return h.mean(dim=0)
        elif self.pooling == "max":
            return h.max(dim=0)[0]
        return h.mean(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edges: List,
        node_count: torch.Tensor,
        node_features: Optional[List[torch.Tensor]] = None,
        gdv_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through deep GIN network.

        Args:
            x: (B, input_dim) aggregated graph features
            edges: List of length B with per-graph edge specifications
            node_count: (B,) tensor of node counts per graph
            node_features: Optional list of length B with pre-computed node features per graph
            gdv_features: Optional list of length B with pre-computed GDV features per graph

        Returns:
            (B, output_dim) predictions
        """
        if not torch.is_tensor(x):
            raise ValueError("x must be a Tensor")

        B = x.shape[0]
        if len(edges) != B:
            raise ValueError(f"edges length {len(edges)} != batch size {B}")
        if node_count.shape[0] != B:
            raise ValueError(f"node_count length {node_count.shape[0]} != batch size {B}")

        if node_features is not None and len(node_features) != B:
            raise ValueError(f"node_features length {len(node_features)} != batch size {B}")
        if gdv_features is not None and len(gdv_features) != B:
            raise ValueError(f"gdv_features length {len(gdv_features)} != batch size {B}")

        outputs = []

        for b in range(B):
            agg_row = x[b]
            device = agg_row.device
            dtype = agg_row.dtype

            edge_index_b, edge_attr_b = self._normalize_edge_item(
                edges[b], device=device, dtype=dtype
            )

            # Use node_count as authoritative source for number of nodes
            num_nodes = int(node_count[b].item())

            # Get pre-computed features for this graph if available
            node_feats_b = node_features[b] if node_features is not None else None
            gdv_b = gdv_features[b] if gdv_features is not None else None

            # Build initial node features
            h = self._build_node_features(
                agg_row,
                N=num_nodes,
                node_features=node_feats_b,
                gdv_features=gdv_b,
            )

            # Edge-conditioned GIN message passing with strong residual connections
            # Edge attributes are now passed directly to each layer for proper edge-weighted aggregation
            for i, (gin_layer, layer_norm) in enumerate(zip(self.gin_layers, self.layer_norms)):
                # Pass edge attributes to edge-conditioned layers
                if isinstance(gin_layer, EdgeConditionedGINConv):
                    h_new = gin_layer(h, edge_index_b, edge_attr_b)
                else:
                    h_new = gin_layer(h, edge_index_b)

                # Dropout for regularization
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                h = layer_norm(h + h_new)

                # Residual connection for all layers
                h = F.relu(h)

            # Graph pooling
            g = self._pool_graph(self.pre_pool_proj(h))

            # Concatenate with graph-level features
            g_concat = torch.cat([g, agg_row], dim=-1)

            # Deep prediction network
            y_b = self.output_mlp(g_concat)
            outputs.append(y_b)

        return torch.stack(outputs, dim=0)
