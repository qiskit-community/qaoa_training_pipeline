#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for recursive training."""

from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.training.recursion import RecursionTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.parameter_extenders import interpolate


class TestRecursion(TrainingPipelineTestCase):
    """Tests for the recursive trainer."""

    def test_simple(self):
        """Test a simple recursion training."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        scipy_trainer = ScipyTrainer(MPSEvaluator())
        trainer = RecursionTrainer(scipy_trainer, interpolate)

        result_pre = scipy_trainer.train(cost_op, params0=[0, 0])

        result = trainer.train(cost_op, params0=result_pre["optimized_params"], reps=3)

        self.assertTrue(result[2]["energy"] < result[3]["energy"])
        self.assertEqual(len(result["optimized_params"]), 6)

    def test_from_config(self):
        """Test that we can setup from a config."""
        config = {
            "trainer": "ScipyTrainer",
            "trainer_init": {
                "evaluator": "MPSEvaluator",
                "evaluator_init": {
                    "bond_dim_circuit": 24,
                    "use_vidal_form": True,
                    "threshold_circuit": 0.001,
                },
                "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
            },
            "parameter_extender": "interpolate",
        }

        trainer = RecursionTrainer.from_config(config)
        self.assertTrue(isinstance(trainer, RecursionTrainer))
