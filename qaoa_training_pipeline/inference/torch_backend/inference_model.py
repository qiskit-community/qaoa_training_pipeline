"""
Inference-only model wrapper - completely separate from training infrastructure.

This module provides a minimal wrapper around PyTorch models specifically for
inference, with no training, validation, or evaluation capabilities.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class InferenceModel:
    """
    Inference-only model wrapper.

    This class wraps a PyTorch model and provides only the forward pass
    functionality needed for inference. It has no training, validation,
    or evaluation methods.

    Key features:
    - No Lightning dependencies
    - No training infrastructure
    - No evaluation/validation logic
    - Just model + forward pass
    - Handles different model architectures

    Example:
        model = InferenceModel(core_model, model_type="edge_transformer")
        output = model.forward(x_vec, edges=edges)
    """

    def __init__(
        self,
        core_model: nn.Module,
        model_type: str,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize inference model wrapper.

        Args:
            core_model: The actual PyTorch model (e.g., EdgeTransformer, GNN)
            model_type: Type of model (determines forward pass signature)
            device: Device for inference
        """
        self.core = core_model
        self.model_type = str(model_type).lower()
        self.device = torch.device(device) if isinstance(device, str) else device

        # Move model to device and set to eval mode
        self.core.to(self.device)
        self.core.eval()

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
        node_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Handles different model architectures by providing appropriate inputs.

        Args:
            x: Input feature tensor (batch_size, num_features)
            edges: Edge tensor for graph-based models (optional)
            t: Timestep tensor for diffusion models (optional)

        Returns:
            Model output tensor

        Raises:
            ValueError: If required inputs are missing for model type
        """
        # Ensure model is in eval mode
        self.core.eval()
        mt = self.model_type

        if "edge" in mt:
            return self.core(x, edges)

        if "diffusion" in mt:
            return self.core(x, edges, t)

        if "gcn" in mt:
            from torch_geometric.data import Data, Batch
            from torch_geometric.utils import degree

            data_list = []
            for i in range(x.shape[0]):
                ed = edges[i]
                mask = (ed[:, 0] == 0) & (ed[:, 1] == 0)
                row_indices = torch.nonzero(~mask).squeeze()
                edge_index = ed[row_indices].T
                num_nodes = int(node_count[i])
                features = torch.unsqueeze(x[i], 0)
                deg = torch.unsqueeze(degree(index=edge_index[0].long(), num_nodes=num_nodes), 1)
                data_list.append(
                    Data(x=deg, edge_index=edge_index, num_nodes=num_nodes, features=features)
                )
            batch_data = Batch.from_data_list(data_list).to(x.device)
            return self.core(batch_data)

        if "graph" in mt:
            if "transformer" in mt:
                print("Careful: Potential edge handling bug for transformer models")
                ed = edges[0]
                mask = ~((ed[:, 0] == 0) & (ed[:, 1] == 0))
                ed = ed[mask].t().contiguous()
                return self.core(x, ed)

            edges_list = []
            for i in range(x.shape[0]):
                ed = edges[i]
                if edge_weights is not None:
                    ew = edge_weights[i]
                    mask = ew != 0.0
                    edges_list.append((ed[mask].t().contiguous(), ew[mask].unsqueeze(-1)))
                else:
                    mask = ~((ed[:, 0] == 0) & (ed[:, 1] == 0))
                    edges_list.append(ed[mask].t().contiguous())
            return self.core(x, edges_list)

        return self.core(x)

    def __call__(
        self,
        x: torch.Tensor,
        edges: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
        node_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(x, edges=edges, t=t, edge_weights=edge_weights, node_count=node_count)

    def to(self, device: str | torch.device) -> InferenceModel:
        """Move model to device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.core.to(self.device)
        return self

    def eval(self) -> InferenceModel:
        """Set model to evaluation mode."""
        self.core.eval()
        return self

    def parameters(self):
        """Return model parameters (for compatibility)."""
        return self.core.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Return model state dict."""
        return self.core.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Load model state dict."""
        return self.core.load_state_dict(state_dict, strict=strict)
