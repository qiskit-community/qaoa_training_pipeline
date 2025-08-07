#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pre-processing functionality for train."""

from abc import ABC, abstractmethod
from typing import Dict


class BasePreprocessor(ABC):
    """Base class for pre-processing input data to train.py."""

    @abstractmethod
    def __call__(self, input_data: Dict) -> Dict:
        """Pre-processes the input data and return data in the same format."""

    @classmethod
    @abstractmethod
    def from_str(cls, input_str: str) -> "BasePreprocessor":
        """Initialize the pre-processor from a string."""
