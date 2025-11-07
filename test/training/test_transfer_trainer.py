#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for transfer training."""

from test import TrainingPipelineTestCase

import networkx as nx

from qaoa_training_pipeline.evaluation import StatevectorEvaluator
from qaoa_training_pipeline.pre_processing.feature_extraction import GraphFeatureExtractor
from qaoa_training_pipeline.pre_processing.feature_matching import TrivialFeatureMatcher
from qaoa_training_pipeline.pre_processing.angle_aggregation import AverageAngleAggregator
from qaoa_training_pipeline.training.data_loading import TrivialDataLoader
from qaoa_training_pipeline.training.transfer_trainer import TransferTrainer
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


class TestTransferTraininer(TrainingPipelineTestCase):
    """Tests for the transfer trainer."""

    def setUp(self):
        """Setup for the tests."""

        # The data contains 4 instances of QAOA angles with depth 2.
        self._data = {
            (2, 6, 9, 3.0, -0.5, 0.6): [
                [2.0493, 0.2858, 0.5022, 0.9166],
                [1.9483, 0.2896, 0.5436, 1.0704],
                [0.6028, 0.4205, 0.4908, 0.7687],
                [1.9433, 0.2150, 0.4336, 1.0426],
            ]
        }

        self._qaoa_depth = 2

        self._trainer = TransferTrainer(
            data_loader=TrivialDataLoader(self._data),
            feature_extractor=GraphFeatureExtractor(standard_devs=False),
            feature_matcher=TrivialFeatureMatcher(),
            angle_aggregator=AverageAngleAggregator(),
            evaluator=StatevectorEvaluator(),
        )

    def test_simple(self):
        """Simple test."""
        graph = nx.random_regular_graph(n=6, d=3, seed=123)
        cost_op = graph_to_operator(graph, pre_factor=-0.5)

        result = self._trainer.train(cost_op, qaoa_depth=self._qaoa_depth)

        self.assertEqual(len(result["optimized_qaoa_angles"]), 2 * self._qaoa_depth)

    def test_to_config(self):
        """Test that we can get a config and that it contains the correct entries."""
        config = self._trainer.to_config()

        config_keys = [
            "data_loader",
            "data_loader_init",
            "feature_extractor",
            "feature_extractor_init",
            "feature_matcher",
            "feature_matcher_init",
            "angle_aggregator",
            "angle_aggregator_init",
        ]

        for key in config_keys:
            self.assertTrue(key in config)

    def test_from_config(self):
        """Test that we can create a class from a config."""

        config = {
            "trainer_name": "TransferTrainer",
            "data_loader": "TrivialDataLoader",
            "data_loader_init": {
                "function_name": "TrivialDataLoader",
                "data": {
                    (2, 6, 9, 3.0, -0.5, 0.6): [
                        [
                            2.0493085568376133,
                            0.2858951231432829,
                            0.5022300575131282,
                            0.9166402122021843,
                        ],
                        [
                            1.9483262509651624,
                            0.28960301948585854,
                            0.5436859274700686,
                            1.0704401466577644,
                        ],
                        [
                            0.6028677683974585,
                            0.420576559738587,
                            0.49083954545850017,
                            0.768706450964894,
                        ],
                        [
                            1.9433497539547053,
                            0.2150960231722957,
                            0.43364162849341265,
                            1.0426597177407184,
                        ],
                    ]
                },
            },
            "feature_extractor": "GraphFeatureExtractor",
            "feature_extractor_init": {
                "feature_extractor_name": "GraphFeatureExtractor",
                "num_nodes": True,
                "num_edges": True,
                "avg_node_degree": True,
                "avg_edge_weights": True,
                "standard_devs": False,
                "density": True,
            },
            "feature_matcher": "TrivialFeatureMatcher",
            "feature_matcher_init": {},
            "angle_aggregator": "AverageAngleAggregator",
            "angle_aggregator_init": {"angle_aggregator_name": "AverageAngleAggregator", "axis": 0},
            "evaluator": "StatevectorEvaluator",
            "evaluator_init": {"statevector_init_args": {}},
        }

        trainer = TransferTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, TransferTrainer))

    def test_config_round_trip(self):
        """Test that we can get a config and that it contains the correct entries."""
        config = self._trainer.to_config()
        trainer = TransferTrainer.from_config(config)
        self.assertTrue(isinstance(trainer, TransferTrainer))
