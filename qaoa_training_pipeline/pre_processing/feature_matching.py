#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to match features from cost operators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseFeatureMatcher(ABC):
    """A base class that matches features from problem instances to data."""

    @abstractmethod
    def __call__(self, features: Tuple, data: Any) -> Tuple:
        """Match the given features to features describing the data.
        
        For example, we have data that containes a map between some features of
        problem instances and their corresponding optimized QAOA angles. This call
        method is designed to find the best match between the given `fieatures` and
        those in the data.
        """

    def to_config(self) -> Dict:
        """Return a config based on the class instance."""
        return {"feature_matcher_name": self.__class__.__name__}
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict):
        """Return the config of the feature matcher."""


class TrivialFeatureMatcher(BaseFeatureMatcher):
    """Perform a trivial mapping."""

    def __call__(self, features: Tuple, data: Dict):
        """Perform a trivial match."""

        if features not in data:
            raise KeyError(
                f"{self.__class__.__name__} could not find feature {features} in data."
            )

        return features

    @classmethod
    def from_config(cls, config: Dict) -> "TrivialFeatureMatcher":
        """Create the trivial feature matcher."""
        return cls()
