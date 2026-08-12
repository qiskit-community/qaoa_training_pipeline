"""
Graph Transformer for QAOA Parameter Prediction

This module implements a Graph Transformer architecture for predicting QAOA parameters.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


class LaplacianPositionalEncoding(nn.Module):
    """
    Compute Laplacian positional encodings for graph nodes.

    """

    def __init__(self, pos_enc_dim: int = 8, normalization: str = "sym"):
        """
        Args:
            pos_enc_dim: Dimension of positional encoding
            normalization: Type of Laplacian normalization ('sym' or 'rw')
        """
        super().__init__()
        self.pos_enc_dim = pos_enc_dim

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute Laplacian positional encodings for a single graph.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph

        Returns:
            Positional encodings [num_nodes, pos_enc_dim]
        """
        device = edge_index.device

        # Convert edge_index to adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Compute degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)

        norm_adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        laplacian = torch.eye(num_nodes, device=device) - norm_adj

        # Compute eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

            # Take the smallest k eigenvectors (excluding the constant eigenvector)
            # Sort by eigenvalue
            idx = eigenvalues.argsort()
            eigenvectors = eigenvectors[:, idx]

            # Take first pos_enc_dim eigenvectors (after the trivial one at idx=0)
            pos_enc = eigenvectors[:, 1 : self.pos_enc_dim + 1]

            # Pad if necessary
            if pos_enc.shape[1] < self.pos_enc_dim:
                pos_enc = F.pad(pos_enc, (0, self.pos_enc_dim - pos_enc.shape[1]))

        except RuntimeError as error:
            # Deterministic zero fallback
            logger.warning("Laplacian eigendecomposition failed: %s", error)
            pos_enc = torch.zeros(
                num_nodes,
                self.pos_enc_dim,
                device=device,
            )

        return pos_enc


class GraphMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer for graphs.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(x.abs())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, num_nodes, embed_dim]
            edge_index: Edge connectivity [2, num_edges]
            attention_mask: Optional mask for attention scores
            edge_weight: Optional per-edge weights [num_edges]; biases attention
                scores toward heavier edges (log-weight additive bias).

        Returns:
            Updated node features [batch_size, num_nodes, embed_dim]
        """
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch_size, num_nodes, embed_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, num_nodes, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # [batch_size, num_heads, num_nodes, num_nodes]

        # Create edge mask: only attend to connected nodes
        edge_mask = torch.zeros(num_nodes, num_nodes, device=x.device)
        edge_mask[edge_index[0], edge_index[1]] = 1.0
        # Add self-loops
        edge_mask.fill_diagonal_(1.0)

        # Apply edge mask
        edge_mask = edge_mask.view(1, 1, num_nodes, num_nodes)  # [1, 1, num_nodes, num_nodes]
        attn_scores = attn_scores.masked_fill(edge_mask == 0, float("-inf"))

        if edge_weight is not None and edge_index.numel() > 0:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
            if edge_weight.numel() != edge_index.shape[1]:
                raise ValueError(
                    f"Expected {edge_index.shape[1]} edge weights, " f"got {edge_weight.numel()}."
                )

            encoded_weight = self._signed_log1p(edge_weight).unsqueeze(-1)
            per_head_bias = self.edge_bias_mlp(encoded_weight).transpose(0, 1)
            dense_bias = torch.zeros(
                self.num_heads,
                num_nodes,
                num_nodes,
                device=x.device,
                dtype=x.dtype,
            )
            dense_bias[:, edge_index[0], edge_index[1]] = per_head_bias
            attn_scores = attn_scores + dense_bias.unsqueeze(0)

        # Apply optional attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch_size, num_heads, num_nodes, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer with attention and feedforward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feedforward network hidden dimension
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        # Multi-head attention
        self.self_attn = GraphMultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # Feedforward network
        self.ff_net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, num_nodes, embed_dim]
            edge_index: Edge connectivity [2, num_edges]
            attention_mask: Optional mask for padded nodes [batch_size, num_nodes]
            edge_weight: Optional per-edge weights [num_edges]

        Returns:
            Updated node features [batch_size, num_nodes, embed_dim]
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(x, edge_index, attention_mask, edge_weight)
        x = x + attn_output
        x = self.norm1(x)

        # Feedforward with residual connection
        ff_output = self.ff_net(x)
        x = x + ff_output
        x = self.norm2(x)

        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer model for QAOA parameter prediction.

    Architecture:
    1. Node feature embedding + Laplacian positional encoding
    2. Stack of Graph Transformer layers
    3. Graph-level pooling (mean)
    4. Concatenate with global features
    5. MLP regression head
    """

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 2,
        node_feature_dim: int = 2,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        pos_enc_dim: int = 8,
        dropout: float = 0.1,
        pooling: str = "mean",
        random_sign_flip_pe: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of global input features
            output_dim: Dimension of output (2*p for gammas and betas)
            node_feature_dim: Default dimension of node features (used if not dynamically determined)
            embed_dim: Embedding dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feedforward network hidden dimension
            pos_enc_dim: Dimension of Laplacian positional encoding
            dropout: Dropout probability
            pooling: Graph pooling method ('mean', 'sum', or 'max')
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.random_sign_flip_pe = random_sign_flip_pe

        self.node_feature_dim = node_feature_dim if node_feature_dim > 0 else 2
        self.node_embedding = nn.Linear(self.node_feature_dim, embed_dim)

        # Laplacian positional encoding
        self.pos_encoder = LaplacianPositionalEncoding(pos_enc_dim=pos_enc_dim)
        self.pos_embedding = nn.Linear(pos_enc_dim, embed_dim)

        # Stack of Graph Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Global feature projection
        self.global_proj = nn.Linear(input_dim, embed_dim)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # *2 for graph + global features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(x.abs())

    def pool_node_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            x: Node features [batch_size, num_nodes, embed_dim]

        Returns:
            Graph-level embedding [batch_size, embed_dim]
        """
        # No padding, use standard pooling
        if self.pooling == "mean":
            graph_embedding = x.mean(dim=1)  # [batch_size, embed_dim]
        elif self.pooling == "sum":
            graph_embedding = x.sum(dim=1)
        elif self.pooling == "max":
            graph_embedding = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return graph_embedding

    def compute_default_node_features(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        deg = degree(edge_index[0], num_nodes=num_nodes).to(torch.float32)

        if edge_weight is None:
            weighted_deg = deg
        else:
            edge_weight = edge_weight.to(device=edge_index.device, dtype=torch.float32)
            weighted_deg = torch.zeros(num_nodes, device=edge_index.device)
            weighted_deg.index_add_(0, edge_index[0], edge_weight.view(-1))

        # Two permutation-respecting structural features.
        return torch.stack(
            [
                torch.log1p(deg),
                self._signed_log1p(weighted_deg),
            ],
            dim=-1,
        )

    def forward(
        self,
        global_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_count: torch.Tensor,
        node_features: Optional[List[torch.Tensor]] = None,
        edge_weights: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with per-graph edge processing.

        Args:
            global_features: Global graph features [batch_size, input_dim]
            edge_index: Edge connectivity, either:
                - [2, num_edges] for single graph (backward compat)
                - list of [M_i, 2] tensors, one per graph in the batch
            node_count: Tensor of node counts per graph [batch_size]
            node_features: Optional list of per-graph node features
                          [num_nodes_i, node_feature_dim]
            edge_weights: Optional list of per-graph edge weights [M_i]

        Returns:
            Predicted QAOA parameters [batch_size, output_dim]
        """
        batch_size = global_features.shape[0]
        device = global_features.device

        # Per-graph edges: list of [M_i, 2] tensors (no padding)
        edge_indices = [e.t().contiguous() for e in edge_index]  # each [2, M_i]

        # Process each graph separately
        graph_embeddings = []

        for i in range(batch_size):
            # Get graph-specific data
            edges_i = edge_indices[i]  # [2, num_edges_i]
            num_nodes_i = int(node_count[i].item())
            ew_i = edge_weights[i].to(device).view(-1) if edge_weights is not None else None

            # Get node features for this graph
            if node_features is not None:
                nf_i = node_features[i].to(device)
                if nf_i.dim() == 1:
                    nf_i = nf_i.unsqueeze(-1)
                if nf_i.shape[1] != self.node_feature_dim:
                    raise ValueError(
                        f"Expected node feature width {self.node_feature_dim}, "
                        f"got {nf_i.shape[1]}."
                    )

            else:
                nf_i = self.compute_default_node_features(edges_i, num_nodes_i, ew_i)

            x_i = self.node_embedding(nf_i)

            # Add positional encoding
            pos_enc_i = self.pos_encoder(edges_i, num_nodes_i)

            # Laplacian eigenvectors are sign-ambiguous. Random sign augmentation
            # prevents the model from depending on one arbitrary eigensolver sign.
            if self.training and self.random_sign_flip_pe:
                signs = torch.where(
                    torch.rand(pos_enc_i.shape[1], device=device) < 0.5,
                    -torch.ones(pos_enc_i.shape[1], device=device),
                    torch.ones(pos_enc_i.shape[1], device=device),
                )
                pos_enc_i = pos_enc_i * signs

            pos_emb_i = self.pos_embedding(pos_enc_i)
            x_i = x_i + pos_emb_i
            x_i = self.dropout(x_i)

            # Add batch dimension for transformer layers
            x_i = x_i.unsqueeze(0)  # [1, num_nodes_i, embed_dim]

            # Apply transformer layers
            for layer in self.transformer_layers:
                x_i = layer(x_i, edges_i, attention_mask=None, edge_weight=ew_i)

            # Pool to graph-level embedding
            graph_emb_i = self.pool_node_features(x_i)
            graph_embeddings.append(graph_emb_i.squeeze(0))

        # Stack graph embeddings
        graph_embedding = torch.stack(graph_embeddings, dim=0)

        # Process global features
        global_emb = self.global_proj(global_features)

        # Concatenate and predict
        combined = torch.cat([graph_embedding, global_emb], dim=1)
        output = self.regression_head(combined)

        return output
