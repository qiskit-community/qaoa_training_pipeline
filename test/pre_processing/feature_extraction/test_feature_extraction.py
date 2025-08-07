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

from qaoa_training_pipeline.pre_processing.feature_extraction import FeatureExtractor
from qaoa_training_pipeline.utils.graph_utils import graph_to_dict


class TestFeatureExtractor(TrainingPipelineTestCase):
    """Test the feature extractor."""

    def setUp(self) -> None:
        self.graph_20_3 = nx.random_regular_graph(n=20, d=3, seed=0)
        self.dict_graph_20_3 = graph_to_dict(self.graph_20_3)
        self.ref_graph_20_3 = {
            "size": 30,
            "order": 20,
            "degree": (3.0, 0.0),
            "weights": (1.0, 0.0),
            "density": 0.15789473684210525,
            "qaoa_depth": None,
        }

    def test_extract_graph(self):
        """Test graph extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract(self.graph_20_3)
        self.assertDictEqual(features, self.ref_graph_20_3)

    def test_extract_dict(self):
        """Test dict extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract(self.dict_graph_20_3)
        self.assertDictEqual(features, self.ref_graph_20_3)

    def test_density_def(self):
        """Test density extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract(self.graph_20_3)
        density = features["density"]
        expected = 2 * features["size"] / (features["order"] * (features["order"] - 1))
        self.assertEqual(density, expected)
