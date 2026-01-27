#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the SciPy trainer."""

import numpy as np
from ddt import data, ddt
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector

from qaoa_training_pipeline.evaluation import (
    EfficientDepthOneEvaluator,
    LightConeEvaluator,
    MPSEvaluator,
)
from qaoa_training_pipeline.training import ScipyTrainer
from qaoa_training_pipeline.training.functions import FourierFunction
from test import TrainingPipelineTestCase


@ddt
class TestSciPyTrainer(TrainingPipelineTestCase):
    """Test that we can use the scipy trainer."""

    def test_basic_with_light_cone(self):
        """Test that a simple training works."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer1 = ScipyTrainer(LightConeEvaluator())
        result = trainer1.train(cost_op, params0=[0.5, 0.2])

        self.assertTrue(result["success"])

    def test_basic_with_efficient_depth_one(self):
        """Test that a simple training works."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer1 = ScipyTrainer(EfficientDepthOneEvaluator())
        result = trainer1.train(cost_op, params0=[0.5, 0.2])

        self.assertTrue(result["success"])

        # Check that history is present.
        self.assertTrue(len(result["energy_history"]) > 0)
        self.assertTrue(len(result["parameter_history"]) > 0)
        self.assertTrue(len(result["energy_evaluation_time"]) > 0)

    def test_min_vs_max(self):
        """Test that a simple training works."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer1 = ScipyTrainer(EfficientDepthOneEvaluator())
        trainer2 = ScipyTrainer(EfficientDepthOneEvaluator(), energy_minimization=True)
        result1 = trainer1.train(cost_op, params0=[0.5, 0.2])
        result2 = trainer2.train(cost_op, params0=[0.5, 0.2])

        self.assertTrue(result1["energy"] > result2["energy"])
        self.assertTrue(result1["energy"] > 0)
        self.assertTrue(0 > result2["energy"])
        self.assertFalse(np.allclose(result1["optimized_params"], result2["optimized_params"]))

    @data(True, False)
    def test_basic_with_mps_simulator_depth_one(self, is_vidal_form):
        """Test that MPS and EfficientDepthOne give the same results."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])
        initial_guess = [0.2, 0.4]
        optimize_args = {"tol": 1.0e-8}

        trainer_efficient_depth_one = ScipyTrainer(EfficientDepthOneEvaluator(), optimize_args)
        trainer_mps = ScipyTrainer(MPSEvaluator(use_vidal_form=is_vidal_form), optimize_args)

        result_efficient_depth_one = trainer_efficient_depth_one.train(
            cost_op, params0=initial_guess
        )
        self.assertTrue(result_efficient_depth_one["success"])

        result_mps = trainer_mps.train(cost_op, params0=initial_guess)
        self.assertTrue(result_mps["success"])

        self.assertAlmostEqual(
            trainer_mps.energy_history[-1], trainer_efficient_depth_one.energy_history[-1], 8
        )

        # Check that history is present.
        hist_len = len(result_mps["energy_history"])
        self.assertTrue(hist_len > 0)
        self.assertEqual(len(result_mps["parameter_history"]), hist_len)
        self.assertEqual(len(result_mps["energy_evaluation_time"]), hist_len)

    @data(True, False)
    def test_mps_vidal_simulator_vs_light_cone_depth_two(self, is_vidal_form):
        """Test that Vidal MPS and Light Cone optimizers give the same results."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IZZI", -1)])
        initial_guess = [0.2, 0.3, 0.4, 0.5]
        n_shots = 100000

        trainer_light_cone = ScipyTrainer(LightConeEvaluator(n_shots))
        trainer_mps = ScipyTrainer(MPSEvaluator(use_vidal_form=is_vidal_form))

        result_light_cone = trainer_light_cone.train(cost_op, params0=initial_guess)
        self.assertTrue(result_light_cone["success"])

        result_mps = trainer_mps.train(cost_op, params0=initial_guess)
        self.assertTrue(result_mps["success"])

        # We do not expect perfect equality due to shot noise
        self.assertLess(
            abs(min(trainer_mps.energy_history) - min(trainer_light_cone.energy_history)), 0.05
        )

    def test_basic_with_mps_simulator_depth_two_statevector(self):
        """Test that Matrix Product State and EfficientDepthOne give the same results."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])
        initial_guess = [0.2, 0.3, 0.4, 0.5]

        ansatz = qaoa_ansatz(cost_operator=cost_op, reps=len(initial_guess) // 2)
        ansatz_state_vector = Statevector(ansatz.assign_parameters(initial_guess))
        initial_energy_state_vector = ansatz_state_vector.expectation_value(cost_op)

        mps_evaluator = MPSEvaluator()
        initial_energy_mps = mps_evaluator.evaluate(cost_op, initial_guess)
        self.assertAlmostEqual(initial_energy_mps, initial_energy_state_vector)

        trainer_mps = ScipyTrainer(MPSEvaluator())
        result_mps = trainer_mps.train(cost_op, params0=initial_guess)
        self.assertTrue(result_mps["success"])

        final_energy_state_vector = Statevector(
            ansatz.assign_parameters(result_mps["optimized_params"])
        ).expectation_value(cost_op)

        # We do not expect perfect equality due to shot noise
        self.assertAlmostEqual(trainer_mps.energy_history[-1], final_energy_state_vector, 5)

    def test_from_config(self):
        """Test the serialization."""
        config = {
            "evaluator": "LightConeEvaluator",
            "evaluator_init": {},
            "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
            "energy_minimization": False,
            "qaoa_angles_function": "FourierFunction",
            "qaoa_angles_function_init": {"depth": 2},
        }

        trainer = ScipyTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, ScipyTrainer))
        self.assertFalse(trainer._energy_minimization)
        self.assertTrue(isinstance(trainer.qaoa_angles_function, FourierFunction))

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        kwargs_str = "params0:1.2/3.4"
        kwargs = ScipyTrainer(EfficientDepthOneEvaluator()).parse_train_kwargs(kwargs_str)

        self.assertDictEqual(kwargs, {"params0": [1.2, 3.4]})
