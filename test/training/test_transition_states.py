#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the Transition States trainer."""

from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline import EfficientDepthOneEvaluator
from qaoa_training_pipeline.evaluation import LightConeEvaluator, MPSEvaluator
from qaoa_training_pipeline.training import TransitionStatesTrainer, ScipyTrainer


class TestTransitionStates(TrainingPipelineTestCase):
    """Class to test the transition states."""

    def setUp(self):
        """Setup some useful variables."""
        self.dummy_trainer = ScipyTrainer(EfficientDepthOneEvaluator())

    def test_transition_state_creation_depth_one(self):
        """Test that we create the correct list of TS."""

        ts_trainer = TransitionStatesTrainer(self.dummy_trainer)
        ts_states = ts_trainer.make_ts([1, 2])

        # Cases for j =i
        self.assertTrue([1, 0, 2, 0] in ts_states)
        self.assertTrue([0, 1, 0, 2] in ts_states)

        # cases for j = i + 1
        self.assertTrue([1, 0, 0, 2] in ts_states)

        # Ensure we have 2p+1 TS with p = 1
        self.assertEqual(len(ts_states), 3)

    def test_transition_state_creation_depth_two(self):
        """Test that we create the correct list of TS."""

        ts_trainer = TransitionStatesTrainer(self.dummy_trainer)
        ts_states = ts_trainer.make_ts([1, 2, 3, 4])

        # Cases for j =i
        self.assertTrue([1, 2, 0, 3, 4, 0] in ts_states)
        self.assertTrue([1, 0, 2, 3, 0, 4] in ts_states)
        self.assertTrue([0, 1, 2, 0, 3, 4] in ts_states)

        # cases for j = i + 1
        self.assertTrue([1, 0, 2, 0, 3, 4] in ts_states)
        self.assertTrue([1, 2, 0, 3, 0, 4] in ts_states)

        # Ensure we have 2p+1 TS with p = 2
        self.assertEqual(len(ts_states), 5)

    def test_transition_state_creation_depth_three(self):
        """Test that we create the correct list of TS."""

        ts_trainer = TransitionStatesTrainer(self.dummy_trainer)
        ts_states = ts_trainer.make_ts([1, 2, 3, 4, 5, 6])

        # Cases for j =i
        self.assertTrue([1, 2, 3, 0, 4, 5, 6, 0] in ts_states)
        self.assertTrue([1, 0, 2, 3, 4, 0, 5, 6] in ts_states)

        # cases for j = i + 1
        self.assertTrue([1, 0, 2, 3, 0, 4, 5, 6] in ts_states)
        self.assertTrue([1, 2, 3, 0, 4, 5, 0, 6] in ts_states)

        # Ensure we have 2p+1 TS with p = 3
        self.assertEqual(len(ts_states), 7)

    def test_training(self):
        """Test starting from a QAOA depth one."""

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        trainer1 = ScipyTrainer(EfficientDepthOneEvaluator())
        result1 = trainer1.train(cost_op, [0.5, 0.2])

        trainer2 = TransitionStatesTrainer(ScipyTrainer(LightConeEvaluator()))
        p0 = result1["optimized_params"]
        result2 = trainer2.train(cost_op, p0)

        self.assertTrue(result1["energy"] < result2["energy"])

    def test_from_config(self):
        """Test that we can load from a config file."""
        config = {
            "trainer": "ScipyTrainer",
            "trainer_init": {
                "evaluator": "LightConeEvaluator",
                "evaluator_init": {},
                "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
            },
        }

        trainer = TransitionStatesTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, TransitionStatesTrainer))
        self.assertTrue(isinstance(trainer.trainer, ScipyTrainer))

    def test_evaluator_config(self):
        """Test that we can load from a config file."""
        config = {
            "trainer": "ScipyTrainer",
            "trainer_init": {
                "evaluator": "MPSEvaluator",
                "evaluator_init": {"bond_dim_circuit": 20, "use_vidal_form": True},
                "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
            },
        }

        trainer = TransitionStatesTrainer.from_config(config)

        self.assertTrue(isinstance(trainer, TransitionStatesTrainer))
        self.assertTrue(isinstance(trainer.trainer, ScipyTrainer))
        self.assertTrue(isinstance(trainer.trainer.evaluator, MPSEvaluator))
