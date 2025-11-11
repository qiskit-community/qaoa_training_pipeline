#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to aggregate angles."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np


class BaseAngleAggregator(ABC):
    """A base class to aggregate QAOA angles.
    
    Children of this base class should take one or more lists of QAOA angles
    and return a single list of floats that correspond to QAOA angles. This
    aggregation can, for instance, return QAOA angles as the mean of multiple
    lists of QAOA angles.
    """

    @abstractmethod
    def __call__(self, qaoa_angles: Any) -> List:
        """Aggregate a set of angles into a list of angles that can be bound into a circuit.

        For example, for example, multiple sets of angles might have been identified as
        good QAOA angles for a given instance. We could then average over these angles or
        try and extrapolate these data to new problem instances.
        """

    def to_config(self) -> dict:
        """Return a config based on the class instance."""
        return {"angle_aggregator_name": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """Return the config of the angle aggregator."""


class TrivialAngleAggregator(BaseAngleAggregator):
    """Perform a trivial aggregation.
    
    This class assumes that the QAOA angles passed to `__call__` are already
    QAOA angles that are ready to be bound to a quantum circuit.
    """

    def __call__(self, qaoa_angles: List):
        """Do not do any aggregation."""
        return qaoa_angles

    @classmethod
    def from_config(cls, config: Dict) -> "TrivialAngleAggregator":
        """Create the trivial feature matcher."""
        return cls()


class AverageAngleAggregator(BaseAngleAggregator):
    """Average a set of angles together."""

    def __init__(self, axis: Union[int, List[int]] = 0):
        """Setup the angle aggregator.

        Args:
            axis: The axis along which to perform the averaging.
        """
        self._axis = axis

    def __call__(self, qaoa_angles: np.array):
        """Average over the qaoa_angles."""
        if any(i >= len(qaoa_angles.shape) for i in self._axis):
            raise ValueError("Input data not coherent with chosen axes")
        return np.average(qaoa_angles, axis=self._axis)

    def to_config(self) -> Dict:
        """Create config from the angle aggregator."""
        config = super().to_config()
        config["axis"] = self._axis
        return config

    @classmethod
    def from_config(cls, config: Dict) -> "AverageAngleAggregator":
        """Create the trivial feature matcher."""
        return cls(config.get("axis", 0))


ANGLE_AGGREGATORS = {
    "TrivialAngleAggregator": TrivialAngleAggregator,
    "AverageAngleAggregator": AverageAngleAggregator,
}
