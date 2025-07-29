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
from qaoa_training_pipeline.training.parameter_scanner import DepthOneScanTrainer
from qaoa_training_pipeline.training.functions import IdentityFunction


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
