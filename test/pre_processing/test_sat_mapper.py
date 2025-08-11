#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the SATMapper."""

from test import TrainingPipelineTestCase

from qaoa_training_pipeline.pre_processing.sat_mapping import SATMapper
from qaoa_training_pipeline.utils.data_utils import load_input


class TestSATMapper(TrainingPipelineTestCase):
    """Tests for the SATMApper."""

    def test_simple(self):
        """Simple test on a three node graph."""

        graph = load_input("test/data/test_graph.json")

        expected = [{"nodes": [1, 2], "weight": 1}, {"nodes": [0, 1], "weight": 1}]

        sat_mapper = SATMapper(timeout=10)
        sat_graph = sat_mapper(graph)

        self.assertEqual(sat_mapper.min_k, 0)
        self.assertListEqual(sat_graph["edge list"], expected)

    def test_from_str(self):
        """Test string initialization."""

        sat_mapper = SATMapper.from_str("10")
        self.assertEqual(sat_mapper.timeout, 10)
