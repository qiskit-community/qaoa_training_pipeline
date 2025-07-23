#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the class DepthOneScanTrainer."""

from qiskit.quantum_info import SparsePauliOp

from test import TrainingPipelineTestCase
from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.training.parameter_scanner import DepthOneScanTrainer


class TestDepthOneScanTrainer(TrainingPipelineTestCase):
    """Tests of the class DepthOneScanTrainer."""

    def test_simple(self):
        """Basic test of the class DepthOneScanTrainer."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer = DepthOneScanTrainer(EfficientDepthOneEvaluator())
        result = trainer.train(cost_op, num_points=3)
        self.assertTrue(len(result["energy_history"]) == 9)
