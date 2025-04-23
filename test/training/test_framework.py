#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the framework."""

from test import TrainingPipelineTestCase

from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator

from qaoa_training_pipeline.training.optimized_parameter_loader import OptimizedParametersLoader
from qaoa_training_pipeline.training.parameter_scanner import DepthOneScanTrainer
from qaoa_training_pipeline.training.random_point import RandomPoint
from qaoa_training_pipeline.training.recursion import RecursionTrainer
from qaoa_training_pipeline.training.reweighting import ReweightingTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.tqa_trainer import TQATrainer
from qaoa_training_pipeline.training.transition_states import TransitionStatesTrainer


class TestTrainingFramework(TrainingPipelineTestCase):
    """Generic tests of the framework."""

    def test_minimization_property(self):
        """Test the minimization property of the different trainers."""

        # Tests for trainers where minimization is not defined
        for cls in [RandomPoint, OptimizedParametersLoader]:
            trainer = cls()

            # pylint: disable=pointless-statement
            with self.assertRaises(ValueError):
                trainer.minimization

        for val in [True, False]:
            # Test for direct trainers.

            for cls in [DepthOneScanTrainer, ScipyTrainer, TQATrainer]:
                trainer = cls(MPSEvaluator(), energy_minimization=val)
                self.assertEqual(trainer.minimization, val)

            # Tests for trainers with sub-trainers
            sub_trainer = ScipyTrainer(EfficientDepthOneEvaluator(), energy_minimization=val)

            for cls in [ReweightingTrainer, RecursionTrainer, TransitionStatesTrainer]:
                trainer = cls(sub_trainer)
                self.assertEqual(trainer.minimization, val)
