#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for feature extraction."""

from test import TrainingPipelineTestCase

import networkx as nx

from qaoa_training_pipeline.pre_processing.feature_extraction import GraphFeatureExtractor
from qaoa_training_pipeline.utils.graph_utils import graph_to_dict
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


class TestFeatureExtractor(TrainingPipelineTestCase):
    """Test the feature extractor."""

    def setUp(self) -> None:
        self.graph_20_3 = nx.random_regular_graph(n=20, d=3, seed=0)
        self.cost_op = graph_to_operator(self.graph_20_3)
        self.dict_graph_20_3 = graph_to_dict(self.graph_20_3)

        self.ref_graph_20_3 = (None, 20, 30, 3.0, 0.0, 1.0, 0.0, 0.15789473684210525)
        self.ref_graph_20_3_nostd = (None, 20, 30, 3.0, 1.0, 0.15789473684210525)

    def test_extract_graph(self):
        """Test graph extraction."""
        extractor = GraphFeatureExtractor()
        features = extractor(self.cost_op, None)
        self.assertTupleEqual(features, self.ref_graph_20_3)

    def test_turn_off_stds(self):
        """Test dict extraction."""
        extractor = GraphFeatureExtractor(standard_devs=False)
        features = extractor(self.cost_op, None)
        self.assertTupleEqual(features, self.ref_graph_20_3_nostd)

    def test_keys(self):
        """Test density extraction."""
        extra_features = {"class": "mis"}

        extractor = GraphFeatureExtractor(extra_features=extra_features)
        feature_keys = extractor.features()

        expected_keys = [
            "qaoa_depth",
            "num_nodes",
            "num_edges",
            "avg_degree",
            "std_degree",
            "avg_weight",
            "std_weight",
            "density",
            "class",
        ]

        self.assertListEqual(feature_keys, expected_keys)
