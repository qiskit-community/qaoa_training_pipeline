# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to identify extrema in arrays of data."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseExtremaLocator(ABC):
    """A base class to define an extrema locator."""

    @abstractmethod
    def __call__(self, array: np.array) -> Tuple[int, float]:
        """The method to return the index and the energy."""


class Argmax(BaseExtremaLocator):
    """Wrapper of the numpy argmax."""

    def __call__(self, array: np.array) -> Tuple[int, float]:
        """Get the maximum energy and where it is."""
        idx = int(np.argmax(array))

        return idx, np.ravel(array)[idx]


class Argmin(BaseExtremaLocator):
    """Wrapper of the numpy argmin."""

    def __call__(self, array: np.array) -> Tuple[int, float]:
        """Get the minimum energy and where it is."""
        idx = int(np.argmin(array))

        return idx, np.ravel(array)[idx]
