import torch
import torch.nn as nn
import math


class EdgeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 2,
        embed_dim: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        edge_embed_dim: int = 32,
        max_nodes: int = None,  # Deprecated: kept for backward compatibility
        use_positional_encoding: bool = True,
        permute_nodes_during_training: bool = True,
    ):
        super(EdgeTransformer, self).__init__()

        self.edge_embed_dim = edge_embed_dim
        self.use_positional_encoding = use_positional_encoding
        self.permute_nodes_during_training = permute_nodes_during_training

        # Project graph-level features to the transformer embedding dimension.
        self.feature_project = nn.Linear(input_dim, embed_dim)

        # NEW: Use sinusoidal positional encoding instead of fixed embedding table
        # This allows the model to handle graphs of any size without retraining
        if not use_positional_encoding:
            # Legacy mode: use fixed embedding table (not recommended)
            if max_nodes is None:
                raise ValueError("max_nodes must be specified when use_positional_encoding=False")
            self.edge_embedding = nn.Embedding(
                num_embeddings=max_nodes, embedding_dim=edge_embed_dim
            )
        else:
            # Modern mode: no embedding table needed, computed on-the-fly
            self.edge_embedding = None

        # Project the averaged edge embedding + weight to the transformer embedding dimension.
        # edge_embed_dim for node embeddings + 1 for edge weight
        self.edge_project = nn.Linear(edge_embed_dim * 2 + 1, embed_dim)

        # Transformer encoder that will process the fused sequence.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final regression layer.
        self.fc = nn.Linear(embed_dim, output_dim)

    def sinusoidal_positional_encoding(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings for node indices.
        This allows the model to handle graphs of any size without retraining.

        Based on "Attention is All You Need" (Vaswani et al., 2017)

        Args:
            node_indices: Tensor of shape (B, num_edges, 2) containing node indices

        Returns:
            Positional encodings of shape (B, num_edges, 2, edge_embed_dim)
        """
        B, num_edges, _ = node_indices.shape
        device = node_indices.device

        # Convert node indices to float for encoding
        position = node_indices.to(torch.float32)

        # Create dimension indices for sinusoidal encoding
        # Use even indices for sin, odd for cos
        dim_indices = torch.arange(0, self.edge_embed_dim, 2, device=device, dtype=torch.float32)
        div_term = torch.exp(dim_indices * -(math.log(10000.0) / self.edge_embed_dim))

        # Reshape for broadcasting
        position = position.unsqueeze(-1)  # (B, num_edges, 2, 1)
        div_term = div_term.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, edge_embed_dim // 2)

        # Compute sinusoidal encodings
        pe = torch.zeros(B, num_edges, 2, self.edge_embed_dim, device=device, dtype=position.dtype)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        return pe  # (B, num_edges, 2, edge_embed_dim)

    @staticmethod
    def _randomly_relabel_edges(edges: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return edges
        num_nodes = int(edges.max().item()) + 1
        if num_nodes < 2:
            return edges
        permutation = torch.randperm(num_nodes, device=edges.device)
        return permutation[edges]

    def forward(
        self,
        features: torch.Tensor,
        edges,
        edge_weights=None,
    ) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, input_dim), e.g., (B, 5)
            edges: list of (num_edges_i, 2) tensors, one per graph (variable length)
            edge_weights: optional list of (num_edges_i,) tensors, one per graph
        Returns:
            output: Tensor of shape (B, output_dim)
        """
        B = features.size(0)

        # Process each graph as its own length-1 batch (graphs have different edge counts).
        outputs = []
        for i in range(B):
            feat_i = features[i : i + 1]  # (1, input_dim)
            edges_i = edges[i].to(device=features.device, dtype=torch.long)  # (num_edges, 2)

            if self.training and self.permute_nodes_during_training:
                edges_i = self._randomly_relabel_edges(edges_i)

            edges_i = edges_i.unsqueeze(0)  # (1, num_edges, 2)

            weights_i = (
                edge_weights[i].unsqueeze(0) if edge_weights is not None else None
            )  # (1, num_edges)
            num_edges = edges_i.size(1)

            # Process graph-level features as a global token.
            # Shape: (1, embed_dim) then unsqueeze to (1, 1, embed_dim)
            feature_token = self.feature_project(feat_i).unsqueeze(1)

            # Embed each node in the edge using positional encoding or legacy embedding
            if self.use_positional_encoding:
                # NEW: Use sinusoidal positional encoding (supports any graph size)
                edge_embeds = self.sinusoidal_positional_encoding(
                    edges_i
                )  # (1, num_edges, 2, edge_embed_dim)
            else:
                # LEGACY: Use fixed embedding table (limited to max_nodes)
                edge_embeds = self.edge_embedding(edges_i)  # (1, num_edges, 2, edge_embed_dim)

            # Combine the two node embeddings per edge.
            edge_mean = edge_embeds.mean(dim=2)  # Shape: (1, num_edges, edge_embed_dim)
            edge_diff = (edge_embeds[:, :, 0] - edge_embeds[:, :, 1]).abs()

            edge_repr = torch.cat([edge_mean, edge_diff], dim=-1)

            # If edge weights are provided, concatenate them to edge representations
            if weights_i is not None:
                # Ensure edge_weights has shape (1, num_edges, 1)
                edge_repr = torch.cat(
                    [
                        edge_repr,
                        weights_i.to(
                            device=edge_repr.device,
                            dtype=edge_repr.dtype,
                        ).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                # Default to weight 1.0 for all edges
                ones = torch.ones(1, num_edges, 1, device=edge_repr.device, dtype=edge_repr.dtype)
                edge_repr = torch.cat(
                    [edge_repr, ones], dim=-1
                )  # (1, num_edges, edge_embed_dim + 1)

            # Project edge representations to the transformer embedding dimension.
            edge_tokens = self.edge_project(edge_repr)  # (1, num_edges, embed_dim)

            # Fuse the tokens by concatenating the global feature token with edge tokens.
            fused_tokens = torch.cat(
                [feature_token, edge_tokens], dim=1
            )  # (1, 1 + num_edges, embed_dim)

            # With batch_first=True, input shape is (batch, sequence_length, embed_dim).
            out = self.transformer(fused_tokens)  # (1, seq_len, embed_dim)

            # Use the first token (global feature token) as the graph representation.
            graph_repr = out[:, 0, :]  # (1, embed_dim)
            outputs.append(self.fc(graph_repr))  # (1, output_dim)

        return torch.cat(outputs, dim=0)  # (B, output_dim)
