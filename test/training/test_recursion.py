#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for recursive training."""

from ddt import ddt, data
from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.training.recursion import RecursionTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.parameter_extenders import interpolate


@ddt
class TestRecursion(TrainingPipelineTestCase):
    """Tests for the recursive trainer."""

    def setUp(self):
        """Setup values for tests."""
        self.cost_op = SparsePauliOp.from_list([("ZIZ", -1), ("IZZ", -1)])

    def test_simple(self):
        """Test a simple recursion training."""
        

        scipy_trainer = ScipyTrainer(MPSEvaluator())
        trainer = RecursionTrainer(scipy_trainer, interpolate)

        result_pre = scipy_trainer.train(self.cost_op, params0=[0, 0])

        result = trainer.train(self.cost_op, params0=result_pre["optimized_params"], reps=3)

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

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        scipy_trainer = ScipyTrainer(MPSEvaluator())
        trainer = RecursionTrainer(scipy_trainer, interpolate)

        kwargs = trainer.parse_train_kwargs("reps:8:params0:1/2")
        self.assertDictEqual(kwargs, {"reps": 8, "params0": [1.0, 2.0]})

    @data(2, None)
    def test_fourier_from_config(self, qaoa_reps):
        """Test that we can setup a recursive Fourier training from a config.
        
        Note: if qaoa_reps is 2 then we always have QAOA depth 2 but we are doing a recursion
        on the Fourier coefficients going from 2 per beta/gamma to 4 Fourier coefficients. If 
        instead, qaoa_reps is None then we are going from QAOA depth 2 to QAOA depth 4 while we 
        optimize the Fourier coefficients.
        """
        config = {
            "trainer": "ScipyTrainer",
            "trainer_init": {
                "evaluator": "StatevectorEvaluator",
                "evaluator_init": {},
                "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
                "qaoa_angles_function": "FourierFunction",
                "qaoa_angles_function_init": {"depth": qaoa_reps},  # p is defined by this number
            },
            "parameter_extender": "interpolate",
        }

        trainer = RecursionTrainer.from_config(config)

        result = trainer.train(self.cost_op, params0=[1, 1, 0, 0], reps=4)

        expected_len = 4 if qaoa_reps is None else qaoa_reps
        
        self.assertEqual(len(result["optimized_params"]), 2 * 4)
        self.assertEqual(len(result["optimized_qaoa_angles"]), 2 * expected_len)
