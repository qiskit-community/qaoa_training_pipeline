#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the TQA trainer."""

# Disable import order for this line. Python has a stdlib test module, but this
# is our own one. Therefore, it is imported with third-party libraries.

from test import TrainingPipelineTestCase  # pylint: disable=wrong-import-order

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.framework.from_config_provider import (
    FromConfigParamsProvider,
)
from qaoa_training_pipeline.framework.param_result import ParamResult
from qaoa_training_pipeline.training.lrqaoa_trainer import LRQAOATrainer
from qaoa_training_pipeline.training.tqa_trainer import TQATrainer


class TestTQA(TrainingPipelineTestCase):
    """Class to test the TQA trainer."""

    def test_no_optim(self):
        """ "Test that we can run without doing any optimization."""
        reps = 3

        with self.assertRaises(
            ValueError,
            msg="Creating TQATrainer without reps=... "
            + "on untrained TQATrainer should raise an error.",
        ):
            trainer = TQATrainer(reps=None, evaluator=None)

        trainer = TQATrainer(evaluator=None, reps=reps)
        self.assertTrue(
            len(trainer.qaoa_angles_function([0.2])) == 2 * reps,
            msg="Calling qaoa_angles_function with reps=... on untrained "
            + "TQATrainer should return list of angles.",
        )

        params_provider = FromConfigParamsProvider.from_config(
            {"params0": [0.75], "qaoa_angles_function": "IdentityFunction"}
        )
        params = params_provider.provide_params()["optimized_params"]
        result = trainer.provide_params(None, params0=params)

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
        trainer = TQATrainer(evaluator=None, reps=reps + 1)
        result = trainer.provide_params(None, params0=params)
        self.assertTrue(
            len(trainer.qaoa_angles_function(result["optimized_params"])) == 2 * (reps + 1),
            msg="Calling qaoa_angles_function without reps=... "
            + "on trained TQATrainer should return list of angles.",
        )

    def test_optim(self):
        """Test that we can optimize the dt of the TQA schedule."""
        evaluator = MPSEvaluator()

        reps = 4
        trainer = TQATrainer(evaluator, minimize_args={"options": {"maxiter": 50}})

        with self.assertRaises(
            ValueError,
            msg="Calling qaoa_angles_function/Using TQA without reps=... "
            + "on untrained TQATrainer should raise an error.",
        ):
            trainer = TQATrainer(evaluator=evaluator, reps=None)
            _ = trainer.qaoa_angles_function([0.2])
        trainer = TQATrainer(evaluator=evaluator, reps=reps)
        self.assertTrue(
            len(trainer.qaoa_angles_function([0.2])) == 2 * reps,
            msg="Calling qaoa_angles_function with reps=... "
            + "on untrained TQATrainer should return list of angles.",
        )

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        result: ParamResult = trainer.provide_params(cost_op, params0=[0.2])

        self.assertEqual(
            result["success"],
            "True",
            msg=f"Full optimizer result: {result!r}",
        )
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
        config = {"evaluator": "StatevectorEvaluator", "evaluator_init": {}, "reps": 1}

        trainer = TQATrainer.from_config(config)

        self.assertIsInstance(trainer, TQATrainer)

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        kwargs_str = "reps:3"
        kwargs = TQATrainer.parse_runtime_kwargs(kwargs_str)
        self.assertDictEqual(kwargs, {"reps": 3})

    def test_lr_schedule(self):
        """Test that we can create LR schedules from configs."""
        evaluator = MPSEvaluator()

        reps = 4
        trainer = LRQAOATrainer(reps=reps, evaluator=evaluator)

        cost_op = SparsePauliOp.from_list([("ZIIZ", -1), ("IZIZ", -1), ("IIZZ", -1)])

        result: ParamResult = trainer.provide_params(cost_op, params0=[0.5, 0.5])

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
