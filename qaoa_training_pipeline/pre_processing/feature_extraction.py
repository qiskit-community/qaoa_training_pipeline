#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An initial implementation for a graph feature extractor."""

from typing import Union
import networkx as nx
import numpy as np

from qaoa_training_pipeline.utils.graph_utils import dict_to_graph


class FeatureExtractor:
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

    def __init__(self) -> None:
        pass

    def extract(self, graph: Union[nx.Graph, dict]) -> dict:
        """
        Extract features from a graph

        Args:
            graph: Input graph, defined as a :class:`networkx.Graph` instance or as a
                serializable dict built from a networkx graph via
                :func:`.qaoa_training_pipeline.utils.graph_utils.graph_to_dict`.

        Returns:
            A dictionary of extracted features.
        """

        if isinstance(graph, dict):
            graph = dict_to_graph(graph)

        # Compute average node degeree for the graph + std
        degree_list = list(deg[1] for deg in graph.degree())
        degree = (float(np.average(degree_list)), float(np.std(degree_list)))

        # If a weight is None it will be overwritten to 1
        weight_list = [
            data[2] if data[2] is not None else 1.0 for data in graph.edges.data("weight")
        ]

        # Compute order (n_vertices)
        order = graph.order()

        # Compute size (n_edges). We could also use nx.Graph.size
        size = len(weight_list)

        weights = (float(np.average(weight_list)), float(np.std(weight_list)))

        # Compute density
        density = nx.density(graph)

        out_features = {
            "size": size,
            "order": order,
            "degree": degree,
            "weights": weights,
            "density": density,
            "qaoa_depth": None,
        }

        return out_features
