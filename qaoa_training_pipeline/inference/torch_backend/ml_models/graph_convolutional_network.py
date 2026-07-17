import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool, GraphNorm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.utils import degree
import torch.nn as nn
import torch
import torch_geometric.utils as pyg_utils
from torch.nn import Linear, ReLU, Sequential

"""
In PyTorch Geometric (PyG), handling multiple graphs of varying sizes in a single batch
 is seamless due to its specialized DataLoader. The DataLoader automatically combines individual Data objects into a single large, 
 block-diagonal graph with an accompanying batch vector that maps each node to its original graph
"""


class GCNModel(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2, input_dim=5, dropout: float = 0.1):
        # super().__init__()
        super(GCNModel, self).__init__()

        # the input channels parameter in the first layer should match the dimension of your chosen synthetic features or embeddings
        # (e.g., 1 for the tensor of ones metho
        self.conv1 = GraphConv(1, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.ln1 = torch.nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.ln2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _compute_node_degree(edge_index, num_nodes):
        return degree(index=edge_index[0], num_nodes=num_nodes)

    def forward(self, data):

        # 1. Access batched elements
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        extra_features = data.features.to(
            device=edge_index.device,
            dtype=torch.float32,
        )
        num_nodes = data.num_nodes
        batch = data.batch
        # x = data.x

        # 2. Decompose batch to list of individual graphs
        individual_graphs_list = data.to_data_list()

        # 3. Compute per-node degrees and z-score them PER GRAPH
        normalized_degrees = []

        for graph in individual_graphs_list:
            node_degree = self._compute_node_degree(
                graph.edge_index,
                graph.num_nodes,
            ).to(
                device=edge_index.device,
                dtype=extra_features.dtype,
            )

            mean_degree = node_degree.mean()
            std_degree = node_degree.std()

            if not torch.isfinite(std_degree) or std_degree < 1e-6:
                std_degree = torch.ones_like(std_degree)

            normalized_degrees.append((node_degree - mean_degree) / std_degree)

        x = torch.cat(normalized_degrees, dim=0).unsqueeze(-1).to(extra_features.dtype)

        # 6. First conv. layer
        x = self.conv1(x, edge_index, edge_weight)

        # 7. relu & dropout.
        x = F.relu(x)
        x = self.dropout(x)

        # 8. Second conv. layer
        x = self.conv2(x, edge_index, edge_weight)

        # 9. relu
        x = F.relu(x)
        x = x.to(torch.float32)

        # 10. Graph Pooling
        # Specifically, use mean pooling to get a single representation per graph for graph-level regression
        x = global_mean_pool(x, batch)

        # 11. Concatenate graph embedding (graph-level vector) with global features
        x = torch.cat([x, extra_features], dim=1)
        x = x.to(torch.float32)

        # 12. Pass onto the final linear years
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.ln2(x)

        return x
