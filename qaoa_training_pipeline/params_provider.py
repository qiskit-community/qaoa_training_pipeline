#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Abstract base class for QAOA parameter providers.

This module defines the ParamsProvider abstract base class, which defines the structure for 
QAOA angle retrieval in subclasses. Parameter providers are responsible for supplying QAOA 
angles.
"""


from abc import ABC, abstractmethod
import warnings
from typing import TypeVar

from qaoa_training_pipeline.training.param_result import ParamResult

T = TypeVar("T")


class ParamsProvider(ABC):
    """A parameter provider is a class that provides QAOA angles."""

    @abstractmethod
    def provide_params(self, **kwargs) -> ParamResult:
        """Return a ParamResult object containing the parameters."""
        raise NotImplementedError("Sub-classes must implement `provide_params`.")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict):
        """Return an instance of the class based on a config."""
        raise NotImplementedError("Sub-classes must implement `from_config`.")

    @abstractmethod
    def to_config(self) -> dict:
        """Creates a serializable dictionary to keep track of how results are created.

        Note: This data structure is not intended for us to recreate the class instance.
        """
        raise NotImplementedError("Sub-classes must implement `to_config`.")

    @staticmethod
    def extract_list(list_str: str, dtype: type = float) -> list:
        """Extract a list of elements from a string in format v0/v1/v2"""
        return [dtype(val) for val in list_str.split("/")]

    def _warn_ignored_inputs(self, **kwargs):
        for name, variable in kwargs.items():
            if variable is not None:
                warnings.warn(f"{self.__class__.__name__} ignores {name} input")

    def _require(self, arg: T | None, name: str) -> T:
        """Raise a ValueError if the argument is None."""
        if arg is None:
            raise ValueError(f"{self.__class__.__name__} requires {name} to be defined")
        return arg
