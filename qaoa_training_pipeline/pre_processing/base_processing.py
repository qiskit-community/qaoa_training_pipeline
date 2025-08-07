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

    def __init__(self):
        """Initialize the pre-processor."""
        self.duration = None

    @abstractmethod
    def __call__(self, input_data: Dict) -> Dict:
        """Pre-processes the input data and return data in the same format."""

    @classmethod
    @abstractmethod
    def from_str(cls, input_str: str) -> "BasePreprocessor":
        """Initialize the pre-processor from a string."""

    def to_config(self) -> dict:
        """Serialize the pre-processor to a dictionary."""
        return {
            "pre_processor_name": self.__class__.__name__,
            "duration": self.duration,
        }
