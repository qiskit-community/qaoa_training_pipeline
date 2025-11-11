#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to extract features from cost operators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class BaseFeatureExtractor(ABC):
    """A base class that extracts properties from cost operators."""

    @abstractmethod
    def __call__(self, cost_op: SparsePauliOp) -> Tuple:
        """Extract features of the given cost operator."""

    def to_config(self) -> dict:
        """Return a config based on the class instance."""
        return {"feature_extractor_name": self.__class__.__name__}

    @abstractmethod
    def features(self) -> List[str]:
        """Return a list of feature names."""

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

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        extract_num_nodes: bool = True,
        extract_num_edges: bool = True,
        extract_avg_node_degree: bool = True,
        extract_avg_edge_weights: bool = True,
        extract_standard_devs: bool = True,
        extract_density: bool = True,
        extra_features: Optional[Dict] = None,
    ):
        """Setup the class.

        Args:
            exctract_num_nodes: If True, add the number of nodes to the features.
            exctract_num_edges: If True, add the number of edges to the features.
            exctract_avg_node_degree: If True, add the average node degree to the features.
            exctract_avg_edge_weights: If True, add the average edge weight to the features.
            exctract_standard_devs: If True, add the standard deviations with the averages.
            exctract_density: If True, add the graph edge density to the features.
            extra_features: Features to be added to the list of features that are not
                dependent on the graphs. The features are added when calling `__call__`.
        """
        self.exctract_num_nodes = extract_num_nodes
        self.exctract_num_edges = extract_num_edges
        self.exctract_avg_node_degree = extract_avg_node_degree
        self.exctract_avg_edge_weights = extract_avg_edge_weights
        self.exctract_standard_devs = extract_standard_devs
        self.exctract_density = extract_density
        self._extra_feature = extra_features or dict()

    def features(self):
        """Return the names of the features."""
        names = ["qaoa_depth"]

        if self.extract_num_nodes:
            names.append("num_nodes")

        if self.extract_num_edges:
            names.append("num_edges")

        if self.extract_avg_node_degree:
            names.append("avg_degree")

            if self.extract_standard_devs:
                names.append("std_degree")

        if self.extract_avg_edge_weights:
            names.append("avg_weight")

            if self.extract_standard_devs:
                names.append("std_weight")

        if self.extract_density:
            names.append("density")

        for feature in self._extra_feature:
            names.append(feature)

        return names

    def __call__(self, cost_op: SparsePauliOp, qaoa_depth: int) -> Tuple:
        """
        Extract features from a graph

        Args:
            cost_op: Input cost operator which must be convertible to a graph.
            qaoa_depth: Depth of the corresponding QAOA circuit

        Returns:
            A tuple of extracted features.
        """
        graph = operator_to_graph(cost_op)

        features = [qaoa_depth]

        if self.extract_num_nodes:
            features.append(graph.order())

        if self.extract_num_edges:
            features.append(len(graph.edges()))

        # Compute average node degeree for the graph + std
        if self.extract_avg_node_degree:
            degree_list = list(deg[1] for deg in graph.degree())
            features.append(float(np.average(degree_list)))

            if self.extract_standard_devs:
                features.append(float(np.std(degree_list)))

        if self.extract_avg_edge_weights:

            # If a weight is None it will be overwritten to 1
            weight_list = [
                data[2] if data[2] is not None else 1.0 for data in graph.edges.data("weight")
            ]

            features.append(float(np.average(weight_list)))

            if self.extract_standard_devs:
                features.append(float(np.std(weight_list)))

        # Compute density
        if self.extract_density:
            features.append(nx.density(graph))

        for feature in self._extra_feature.values():
            features.append(feature)

        return tuple(features)

    def to_config(self) -> dict:
        """Return an instance of this class based on a config."""
        config = super().to_config()

        config["extract_num_nodes"] = self.extract_num_nodes
        config["extract_num_edges"] = self.extract_num_edges
        config["extract_avg_node_degree"] = self.extract_avg_node_degree
        config["extract_avg_edge_weights"] = self.extract_avg_edge_weights
        config["extract_standard_devs"] = self.extract_standard_devs
        config["extract_density"] = self.extract_density

        return config

    @classmethod
    def from_config(cls, config) -> "GraphFeatureExtractor":
        """Setup the feature extractor from a config."""
        return cls(
            config.get("extract_num_nodes", True),
            config.get("extract_num_edges", True),
            config.get("extract_avg_node_degree", True),
            config.get("extract_avg_edge_weights", True),
            config.get("extract_standard_devs", True),
            config.get("extract_density", True),
        )


FEATURE_EXTRACTORS = {
    "GraphFeatureExtractor": GraphFeatureExtractor,
}
