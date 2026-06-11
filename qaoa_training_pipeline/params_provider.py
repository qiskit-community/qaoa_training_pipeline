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
from qaoa_training_pipeline.functions import (
    BaseAnglesFunction,
    IdentityFunction,
)

T = TypeVar("T")


class ParamsProvider(ABC):
    """A parameter provider is a class that provides QAOA angles."""

    def __init__(
        self, 
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ):
        """Initialize the parameter provider."""
        self._qaoa_angles_function = qaoa_angles_function or IdentityFunction()

    @property
    def qaoa_angles_function(self) -> BaseAnglesFunction:
        """Return the QAOA angles function of the params provider."""
        return self._qaoa_angles_function

    @abstractmethod
    def provide_params(self, *args, **kwargs) -> ParamResult:
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
    
    def parse_runtime_kwargs(self, kwargs_str: str | None = None) -> dict:
        """Method to parse keyword arguments passed when using the pipeline
        from the command line.

        The kwarg string is given, e.g., in form `k1:v1:k2:v2`. If the value is
        a list then the values in the list must be spaced by a `/`, for example,
        `params0:1.234/4.56`.
        """
        if kwargs_str is None:
            return dict()

        items = kwargs_str.split(":")

        if len(items) % 2 != 0:
            raise ValueError(
                f"Malformed keyword arguments {kwargs_str}: should be k1:v1:k2:v2_...."
            )

        return {items[idx]: items[idx + 1] for idx in range(0, len(items), 2)}

