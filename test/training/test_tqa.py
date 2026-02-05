#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the TQA trainer."""

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.training.tqa_trainer import TQATrainer

# Disable import order for this line. Python has a stdlib test module, but this
# is our own one. Therefore, it is imported with third-party libraries.
from test import TrainingPipelineTestCase  # pylint: disable=wrong-import-order


class TestTQA(TrainingPipelineTestCase):
    """Class to test the TQA trainer."""

    def test_no_optim(self):
        """ "Test that we can run without doing any optimization."""
        reps = 3
        trainer = TQATrainer()

        with self.assertRaises(
            ValueError,
            msg="Calling qaoa_angles_function without reps=... "
            + "on untrained TQATrainer should raise an error.",
        ):
            _ = trainer.qaoa_angles_function([0.2])

        self.assertTrue(
            len(trainer.qaoa_angles_function([0.2], reps=reps)) == 2 * reps,
            msg="Calling qaoa_angles_function with reps=... on untrained "
            + "TQATrainer should return list of angles.",
        )

        result = trainer.train(None, reps=reps)

        self.assertListEqual(
            result["optimized_qaoa_angles"],
            [0.875, 0.625, 0.375, 0.125, 0.375, 0.625],
            msg="Number of QAOA angles is not as expected.",
        )
        self.assertListEqual(
            result["optimized_params"],
            [0.75],
            msg="Optimized params with default argument should be [0.75]",
        )

        # Check that history is not present.
        self.assertTrue(len(result["energy_history"]) == 0)
        self.assertTrue(len(result["parameter_history"]) == 0)
        self.assertTrue(len(result["energy_evaluation_time"]) == 0)
        # Double check that the default number of reps for qaoa_angles_function
        # is the same as the most recent run.
        self.assertTrue(
            len(trainer.qaoa_angles_function(result["optimized_params"])) == 2 * reps,
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return list of angles.",
        )

        result = trainer.train(None, reps=reps + 1)
        self.assertTrue(
            len(trainer.qaoa_angles_function(result["optimized_params"])) == 2 * (reps + 1),
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return list of angles.",
        )

    def test_optim(self):
        """Test that we can optimize the dt of the TQA schedule."""
        evaluator = MPSEvaluator()

        reps = 4
        trainer = TQATrainer(evaluator)

        with self.assertRaises(
            ValueError,
            msg="Calling qaoa_angles_function without reps=... "
            + "on untrained TQATrainer should raise an error.",
        ):
            _ = trainer.qaoa_angles_function([0.2])

        self.assertTrue(
            len(trainer.qaoa_angles_function([0.2], reps=reps)) == 2 * reps,
            msg="Calling qaoa_angles_function with reps=... "
            + "on untrained TQATrainer should return list of angles.",
        )

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        result: ParamResult = trainer.train(cost_op, reps=reps)

        self.assertEqual(result["success"], "True")
        self.assertEqual(
            len(result["optimized_params"]),
            1,
            msg="There is only one parameter, dt, for TQATrainer.",
        )
        self.assertEqual(
            len(result["optimized_qaoa_angles"]),
            2 * reps,
            msg="Number of QAOA angles is not as expected.",
        )
        self.assertTrue(
            len(trainer.qaoa_angles_function(result["optimized_params"])) == 2 * reps,
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return list of angles.",
        )
        self.assertListEqual(
            result["optimized_qaoa_angles"],
            trainer.qaoa_angles_function(result["optimized_params"]),
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return the same angles.",
        )

        # Check that history is present.
        self.assertTrue(len(result["energy_history"]) > 0)
        self.assertTrue(len(result["parameter_history"]) > 0)
        self.assertTrue(len(result["energy_evaluation_time"]) > 0)

    def test_from_config(self):
        """Test that we can create TQA trainers from configs."""
        config = {}

        trainer = TQATrainer.from_config(config)
        self.assertIsNone(trainer._evaluator)

        config = {
            "evaluator": "MPSEvaluator",
            "evaluator_init": {"bond_dim_circuit": 2},
        }

        trainer = TQATrainer.from_config(config)
        self.assertTrue(isinstance(trainer.evaluator, MPSEvaluator))

        self.assertEqual(trainer.evaluator.to_config()["bond_dim_circuit"], 2)

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        kwargs_str = "reps:3"
        kwargs = TQATrainer().parse_train_kwargs(kwargs_str)

        self.assertDictEqual(kwargs, {"reps": 3})

    def test_lr_schedule(self):
        """Test that we can create LR schedules from configs."""
        evaluator = MPSEvaluator()

        reps = 4
        trainer = TQATrainer(evaluator, initial_dt=(0.5, 0.5))

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        result: ParamResult = trainer.train(cost_op, reps=reps)

        self.assertEqual(
            len(result["optimized_params"]),
            2,
            msg="There is only one parameter, dt, for TQATrainer.",
        )
        self.assertEqual(
            len(result["optimized_qaoa_angles"]),
            2 * reps,
            msg="Number of QAOA angles is not as expected.",
        )
        self.assertTrue(
            len(trainer.qaoa_angles_function(result["optimized_params"])) == 2 * reps,
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return list of angles.",
        )

        # Check that history is present.
        self.assertTrue(len(result["energy_history"]) > 0)
        self.assertTrue(len(result["parameter_history"]) > 0)
        self.assertTrue(len(result["energy_evaluation_time"]) > 0)
        self.assertEqual(trainer.qaoa_angles_function._tqa_schedule.__name__, "lr_schedule")
