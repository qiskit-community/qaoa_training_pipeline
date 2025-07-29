#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the QAOA angles functions."""

from test import TrainingPipelineTestCase

import numpy as np

from qaoa_training_pipeline.training.functions import IdentityFunction, FourierFunction


class TestFunctions(TrainingPipelineTestCase):
    """Tests for the functions."""

    def test_identity(self):
        """Test the identity function."""

        self.assertListEqual(IdentityFunction()([1, 2, 3, 4]), [1, 2, 3, 4])

    def test_fourier(self):
        """Test the Fourier function."""

        function = FourierFunction(2)

        angles = function([1, 1])

        beta1 = np.cos(0.5 * 0.5 * np.pi / 2)
        beta2 = np.cos(0.5 * 1.5 * np.pi / 2)
        gamma1 = np.sin(0.5 * 0.5 * np.pi / 2)
        gamma2 = np.sin(0.5 * 1.5 * np.pi / 2)

        self.assertEqual(len(angles), 4)
        self.assertTrue(np.allclose(angles, [beta1, beta2, gamma1, gamma2]))

    def test_fourier_from_config(self):
        """Test the Fourier function from the config."""
        config = {"depth": 2}

        function = FourierFunction.from_config(config)

        self.assertTrue(function._depth, 2)
