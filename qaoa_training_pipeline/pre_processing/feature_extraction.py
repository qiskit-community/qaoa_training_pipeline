#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to extract features from cost operators."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import networkx as nx
import numpy as np

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.graph_utils import operator_to_graph

class BaseFeatureExtractor(ABC):
    """A base class that extracts properties from cost operators."""

    @abstractmethod
    def __call__(self, cost_op: SparsePauliOp) -> Tuple:
        """Extract features of the given cost operator."""

    def to_config(self) -> Dict:
        """Return the config of the feature extractor."""

    def to_config(self) -> dict:
        """Return a config based on the class instance."""
        return {"feature_extractor_name": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict):
        """Return an instance of this class based on a config."""


class GraphFeatureExtractor(BaseFeatureExtractor):
    """
    An initial implementation for a graph feature extractor to be potentially
    used in a "qaoa training pipeline service".

    The features currently handled by the extractor are:

        * size (int): number of edges in the graph
        * order (int): number of nodes in the graph
        * degree (tuple(float, float)): the degree of a node is the number of edges adjacent to it.
            We interpret the degree of a graph as the average of the degree of its nodes, this feature
            is thus formatted as a tuple of (avg_degree, std), where `std` is the standard deviation.
            If ``std==0.0``, the graph is regular.
        * weights (tuple(float, float)): average weight of all edges in the graph, formatted as a
            tuple (avg, std), i.e. using the same format as for the `degree` variable.
        * density (float): graph density defined as :math: d = \frac{2m}{n(n-1)} . Where ``n`` is
            the number of nodes and ``m`` is the number of edges.

    """

    def __init__(
        self,
        num_nodes: bool=True,
        num_edges: bool=True,
        avg_node_degree: bool=True,
        avg_edge_weights: bool=True,
        standard_devs: bool=True,
        density: bool=True,
    ):
        """Setup the class.
        
        Args:
            num_nodes: If True, add the number of nodes to the features.
            num_edges: If True, add the number of edges to the features.
            avg_node_degree: If True, add the average node degree to the features.
            avg_edge_weights: If True, add the average edge weight to the features.
            standard_devs: If True, add the standard deviations with the averages.
            density: If True, add the graph edge density to the features.
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.avg_node_degree = avg_node_degree
        self.avg_edge_weights = avg_edge_weights
        self.standard_devs = standard_devs
        self.density = density

    def __call__(self, cost_op: SparsePauliOp, qaoa_depth: int) -> Tuple:
        """
        Extract features from a graph

        Args:
            cost_op: Input cost operator which must be convertible to a graph.

        Returns:
            A tuple of extracted features.
        """
        graph = operator_to_graph(cost_op)

        features = [qaoa_depth]

        if self.num_nodes:
            features.append(graph.order())

        if self.num_edges:
            features.append(len(graph.edges()))

        # Compute average node degeree for the graph + std
        if self.avg_node_degree:
            degree_list = list(deg[1] for deg in graph.degree())
            features.append(float(np.average(degree_list)))

            if self.standard_devs:
                features.append(float(np.std(degree_list)))

        if self.avg_edge_weights:

            # If a weight is None it will be overwritten to 1
            weight_list = [
                data[2] if data[2] is not None else 1.0 for data in graph.edges.data("weight")
            ]

            features.append(float(np.average(weight_list)))

            if self.standard_devs:
                features.append(float(np.std(weight_list)))

        # Compute density
        if self.density:
            features.append(nx.density(graph))

        return tuple(features)

    def to_config(self) -> dict:
        """Return an instance of this class based on a config."""
        config = super().to_config()

        config["num_nodes"] = self.num_nodes
        config["num_edges"] = self.num_edges
        config["avg_node_degree"] = self.avg_node_degree
        config["avg_edge_weights"] = self.avg_edge_weights
        config["standard_devs"] = self.standard_devs
        config["density"] = self.density

        return config

    @classmethod
    def from_config(cls, config) -> "GraphFeatureExtractor":
        """Setup the feature extractor from a config."""
        return cls(
            config.get("num_nodes", True),
            config.get("num_edges", True),
            config.get("avg_node_degree", True),
            config.get("avg_edge_weights", True),
            config.get("standard_devs", True),
            config.get("density", True),
        )
