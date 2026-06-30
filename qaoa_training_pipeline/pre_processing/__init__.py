#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of methods to pre-process data before QAOA parameter training."""

from .base_processing import BasePreprocessor
from .angle_aggregation import BaseAngleAggregator, TrivialAngleAggregator, AverageAngleAggregator
from .feature_extraction import BaseFeatureExtractor, GraphFeatureExtractor
from .feature_matching import BaseFeatureMatcher, TrivialFeatureMatcher, MinimumNormFeatureMatcher
from .sat_mapping import SATMapper, SATResult

__all__ = [
    "BasePreprocessor",
    "BaseAngleAggregator",
    "TrivialAngleAggregator",
    "AverageAngleAggregator",
    "BaseFeatureExtractor",
    "GraphFeatureExtractor",
    "BaseFeatureMatcher",
    "TrivialFeatureMatcher",
    "MinimumNormFeatureMatcher",
    "SATMapper",
    "SATResult",
    "PREPROCESSORS",
]

PREPROCESSORS = {
    "sat": SATMapper,
}
