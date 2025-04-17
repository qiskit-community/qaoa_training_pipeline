#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the reweighted trainer."""

import networkx as nx
from unittest import TestCase

from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.training.reweighting import ReweightingTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


class TestReweightedTrainer(TestCase):
    """Methods to test the reweighted trainer."""

    def test_train(self):
        """Simple trainig test."""

        edges = [
            (0, 1, 3),
            (0, 2, 1),
            (0, 3, -4),
            (0, 4, -1),
            (0, 5, 1),
            (1, 2, 1),
            (1, 3, 3),
            (2, 3, 1),
            (2, 4, -1),
            (2, 5, 4),
            (3, 4, 3),
            (3, 5, -3),
        ]

        graph = nx.Graph()
        graph.add_weighted_edges_from(edges)
        cost_op = graph_to_operator(graph)

        # Setup the trainers.
        sub_trainer = ScipyTrainer(evaluator=EfficientDepthOneEvaluator())
        trainer = ReweightingTrainer(trainer1=sub_trainer)
        result = trainer.train(cost_op=cost_op, trainer1_kwargs={"params0": [1, 1]})

        self.assertTrue(result["success"])
