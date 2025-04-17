#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the TQA trainer."""

from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.training.tqa_trainer import TQATrainer


class TestTQA(TrainingPipelineTestCase):
    """Class to test the TQA trainer."""

    def test_no_optim(self):
        """ "Test that we can run without doing any optimization."""
        result = TQATrainer().train(None, 3)

        self.assertListEqual(result["optimized_params"], [0.875, 0.625, 0.375, 0.125, 0.375, 0.625])

    def test_optim(self):
        """Test that we can optimize the dt of the TQA schedule."""
        evaluator = MPSEvaluator()

        trainer = TQATrainer(evaluator)

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        result = trainer.train(cost_op, reps=4)

        self.assertEqual(result["success"], "True")
        self.assertEqual(len(result["optimized_params"]), 8)

    def test_from_config(self):
        """Test that we can create TQA trainers from configs."""
        config = {}

        trainer = TQATrainer.from_config(config)
        self.assertIsNone(trainer.evaluator)

        config = {"evaluator": "MPSEvaluator", "evaluator_init": {"bond_dim_circuit": 2}}

        trainer = TQATrainer.from_config(config)
        self.assertTrue(isinstance(trainer.evaluator, MPSEvaluator))

        self.assertEqual(trainer.evaluator.to_config()["bond_dim_circuit"], 2)
