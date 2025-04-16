# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the class that fits to a model."""

import networkx as nx
from ddt import ddt, data
from unittest import TestCase

from qaoa_training_pipeline.training.models.random_regular_fit import RandomRegularDepthOneFit
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


@ddt
class TestRandomRegularModel(TestCase):
    @data(3, 4, 5, 7, 8, 9)
    def test_model(self, degree: int):
        model = RandomRegularDepthOneFit()

        graph = nx.random_regular_graph(d=degree, n=26)
        cost_op = graph_to_operator(graph)

        self.assertEqual(len(model.train(cost_op)["optimized_params"]), 2)

    def test_from_config(self):
        """Test that we can initialize from a config."""
        config = {}

        trainer = RandomRegularDepthOneFit.from_config(config)

        self.assertTrue(isinstance(trainer, RandomRegularDepthOneFit))

        config = {
            "evaluator": "EfficientDepthOneEvaluator",
            "evaluator_init": {},
        }

        trainer = RandomRegularDepthOneFit.from_config(config)

        self.assertTrue(isinstance(trainer, RandomRegularDepthOneFit))
