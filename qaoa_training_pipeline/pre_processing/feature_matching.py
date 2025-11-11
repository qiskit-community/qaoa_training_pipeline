#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to match features from cost operators."""

from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple
import numpy as np


class BaseFeatureMatcher(ABC):
    """A base class that matches features from problem instances to an existing set."""

    @abstractmethod
    def __call__(self, features: Tuple, set_of_features: Set[Tuple]) -> Tuple:
        """Match the given features to features in the `set_of_features`.

        This call method is designed to find the best match between the given 
        `features` and those in the `set_of_features`.

        Args:
            features: A tuple of features.
            set_of_features: Each element of the set is a tuple of features.
        """

    def to_config(self) -> Dict:
        """Return a config based on the class instance."""
        return dict()

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict):
        """Return the config of the feature matcher."""


class TrivialFeatureMatcher(BaseFeatureMatcher):
    """Perform a trivial mapping."""

    def __call__(self, features: Tuple, set_of_features: Set[Tuple]):
        """Perform a trivial match."""

        if features not in set_of_features:
            raise KeyError(
                f"{self.__class__.__name__} could not find feature {features} in {set_of_features}."
            )

        return features

    @classmethod
    def from_config(cls, config: Dict) -> "TrivialFeatureMatcher":
        """Create the trivial feature matcher."""
        return cls()


class MinimumNormFeatureMatcher(BaseFeatureMatcher):
    """Match the features based on the smallest inner product."""

    def __call__(self, features: Tuple, set_of_features: Set[Tuple]):
        """Find the key in the data that minimizes the dot product.
        
        A dot product is computed between the given `features` and each of the
        existing features stored in the data (as keys in the dict). The features
        in the data with the minimum dot product are returned. For example, if
        the sets of features in the data are `f1` and `f2` we return
        min(np.dot(x, f1), np.dot(x, f2)).
        """
        return min(set_of_features, key=lambda x: np.dot(x, features))

    @classmethod
    def from_config(cls, config: Dict) -> "MinimumNormFeatureMatcher":
        """Create the trivial feature matcher."""
        return cls()


FEATURE_MATCHERS = {
    "TrivialFeatureMatcher": TrivialFeatureMatcher,
    "MinimumNormFeatureMatcher": MinimumNormFeatureMatcher,
}
