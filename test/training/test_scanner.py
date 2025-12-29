#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the class DepthOneScanTrainer."""

from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.training.parameter_scanner import (
    DepthOneScanTrainer,
    DepthOneGammaScanTrainer,
)
from qaoa_training_pipeline.training.functions import IdentityFunction
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class TestDepthOneScanTrainer(TrainingPipelineTestCase):
    """Tests of the class DepthOneScanTrainer."""

    def setUp(self):
        """Setup variables."""
        self.trainer = DepthOneScanTrainer(EfficientDepthOneEvaluator())
        self.cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

    def test_simple(self):
        """Basic test of the class DepthOneScanTrainer."""
        result = self.trainer.train(self.cost_op, num_points=3)
        self.assertTrue(len(result["energy_history"]) == 9)

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        kwargs = self.trainer.parse_train_kwargs("num_points:8")
        self.assertDictEqual(kwargs, {"num_points": 8})

        kwargs = self.trainer.parse_train_kwargs("parameter_ranges:1.2/2.2/3.3/4.5")
        self.assertDictEqual(kwargs, {"parameter_ranges": [(1.2, 2.2), (3.3, 4.5)]})

    def test_scan_range(self):
        """Test that when we specify a range the angles stay in that range."""
        kwargs = self.trainer.parse_train_kwargs("num_points:3:parameter_ranges:1.2/2.2/3.3/4.5")
        result = self.trainer.train(self.cost_op, **kwargs)

        opt_params = result["optimized_params"]
        self.assertTrue(opt_params[0] >= 1.2)
        self.assertTrue(opt_params[0] <= 2.2)
        self.assertTrue(opt_params[1] >= 3.3)
        self.assertTrue(opt_params[1] <= 4.5)

    def test_from_config(self):
        """Test the serialization."""
        config = {
            "evaluator": "EfficientDepthOneEvaluator",
            "evaluator_init": {},
            "energy_minimization": True,
            "qaoa_angles_function": "IdentityFunction",
            "qaoa_angles_function_init": {},
        }

        trainer = DepthOneScanTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, DepthOneScanTrainer))
        self.assertTrue(trainer._energy_minimization)
        self.assertTrue(isinstance(trainer.qaoa_angles_function, IdentityFunction))


class TestDepthOneGammaScanTrainer(TrainingPipelineTestCase):
    """Tests of the class DepthOneGammaScanTrainer."""

    def setUp(self):
        """Setup variables."""
        self.trainer = DepthOneGammaScanTrainer(EfficientDepthOneEvaluator())
        self.cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1), ("ZZII", -1)])
        self.graph = operator_to_graph(self.cost_op)

    def test_simple(self):
        """Basic test of the class DepthOneScanTrainer."""
        result = self.trainer.train(self.cost_op, num_points=3)
        self.assertTrue(len(result["energy_history"]) == 3)

    def test_scan_range(self):
        """Test that when we specify a range the angles stay in that range."""
        kwargs = self.trainer.parse_train_kwargs("num_points:3:parameter_ranges:3.3/4.5")
        result = self.trainer.train(self.cost_op, **kwargs)

        opt_params = result["optimized_params"]
        self.assertTrue(opt_params[1] >= 3.3)
        self.assertTrue(opt_params[1] <= 4.5)

    def test_from_config(self):
        """Test the serialization."""
        config = {
            "evaluator": "EfficientDepthOneEvaluator",
            "evaluator_init": {},
            "energy_minimization": True,
            "qaoa_angles_function": "IdentityFunction",
            "qaoa_angles_function_init": {},
        }

        trainer = DepthOneGammaScanTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, DepthOneGammaScanTrainer))
        self.assertTrue(trainer._energy_minimization)
        self.assertTrue(isinstance(trainer.qaoa_angles_function, IdentityFunction))

    def test_prod_cos_edges_from_node(self):
        """
        Test the product term of edges connected to a node.
        """
        prod = self.trainer._prod_cos_edges_from_node(
            self.graph, node=0, nbrs=[1, 2, 3], gamma=1.5, weight_attr="weight"
        )
        expected_prod = -0.9702769379215033
        assert prod == expected_prod

    def test_prod_cos_triangle_terms(self):
        """
        Test the product term of edges connected to a node and it's neighbor, creating a triangle.

        """
        prod = self.trainer._prod_cos_triangle_terms(
            self.graph, u=0, v=3, mutual_nbrs=[2], gamma=1.5, weight_attr="weight", plus=True
        )
        expected_prod = 0.960170286650366
        self.assertAlmostEqual(prod, expected_prod)
        prod = self.trainer._prod_cos_triangle_terms(
            self.graph, u=0, v=3, mutual_nbrs=[2], gamma=1.5, weight_attr="weight", plus=False
        )
        expected_prod = 1
        self.assertAlmostEqual(prod, expected_prod)

    def test_compute_a_b_matrices_for_gamma(self):
        """
        Test the computation of A and B matrices as shown in the paper.
        """
        a_matrix, b_matrix = self.trainer._compute_a_b_matrices_for_gamma(self.graph, gamma=1.5)
        expected_a_matrix = -0.0013910591808833639
        self.assertAlmostEqual(a_matrix, expected_a_matrix)
        expected_b_matrix = -0.01951626068306727
        self.assertAlmostEqual(b_matrix, expected_b_matrix)

    def test_beta_star_for_gamma(self):
        """
        Test that the optimal beta value is received for a set gamma.
        """
        beta = self.trainer._beta_star_for_gamma(self.graph, gamma=1.5)
        expected_beta = -0.7499982063376642
        self.assertAlmostEqual(beta, expected_beta)
