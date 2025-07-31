#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for parameter loading."""

from test import TrainingPipelineTestCase

from qaoa_training_pipeline.training.optimized_parameter_loader import OptimizedParametersLoader


class TestOptimizedParameterLoader(TrainingPipelineTestCase):
    """Methods to test loading optimized parameters."""

    def test_train(self):
        """Test that we can load from a file."""

        loader = OptimizedParametersLoader()

        kwargs = loader.parse_train_kwargs("folder:test/data/:file_pattern:20nodes_random7regular")

        result = loader.train(None, **kwargs)

        self.assertEqual(
            result["optimized_params"],
            [0.44901865190957657, 0.19974528971646474],
        )

    def test_parse_train_kwargs(self):
        """Test parsing of training args."""
        trainer = OptimizedParametersLoader()

        kwargs = trainer.parse_train_kwargs("folder:my_folder:file_pattern:*.json")
        self.assertDictEqual(kwargs, {"folder": "my_folder", "file_pattern": "*.json"})
