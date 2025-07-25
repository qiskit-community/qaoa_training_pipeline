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

    def test_simple(self):
        """Basic test of the class DepthOneScanTrainer."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer = DepthOneScanTrainer(EfficientDepthOneEvaluator())
        result = trainer.train(cost_op, num_points=3)
        self.assertTrue(len(result["energy_history"]) == 9)

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
