#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for angle aggregation."""

from test import TrainingPipelineTestCase

import numpy as np

from qaoa_training_pipeline.pre_processing.angle_aggregation import AverageAngleAggregator


class TestAverageAngleAggregator(TrainingPipelineTestCase):
    """Test the feature extractor."""

    def setUp(self):
        self.qaoa_angles = np.random.uniform(0, 2 * np.pi, (2, 10))

    def test_aggregate_angles(self):
        """Test angle aggregation."""
        aggregator = AverageAngleAggregator()
        features = aggregator(self.qaoa_angles)
        self.assertTupleEqual(features.shape, (1, 10))

    def test_aggregate_angles_with_wrapping(self):
        """Test angle aggregation with wrapping."""
        aggregator = AverageAngleAggregator(wrap_angles=True, beta_wrap=np.pi, gamma_wrap=np.pi / 2)
        features = aggregator(self.qaoa_angles)
        self.assertTupleEqual(features.shape, (1, 10))
