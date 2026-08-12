"""Tests for recursive transition states training."""

from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.training.recursive_transition_states import RecursiveTransitionStates
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer


class TestRecursion(TrainingPipelineTestCase):
    """Tests for the recursive transition states trainer."""

    def test_simple(self):
        """Test a simple recursive training."""
        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        scipy_trainer = ScipyTrainer(MPSEvaluator())
        trainer = RecursiveTransitionStates(
            scipy_trainer,
            reps=3,
        )

        result_pre = scipy_trainer.provide_params(cost_op, params0=[0, 0])

        result = trainer.provide_params(
            cost_op,
            params0=result_pre["optimized_params"],
            mixer=None,
            initial_state=None,
            ansatz_circuit=None,
        )

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
            "reps": 3,
        }

        trainer = RecursiveTransitionStates.from_config(config)
        self.assertTrue(isinstance(trainer, RecursiveTransitionStates))

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        scipy_trainer = ScipyTrainer(MPSEvaluator())
        trainer = RecursiveTransitionStates(scipy_trainer)

        kwargs = trainer.parse_runtime_kwargs("reps:8:params0:1/2")
        self.assertDictEqual(kwargs, {"reps": 8, "params0": [1.0, 2.0]})
