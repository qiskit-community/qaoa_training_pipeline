#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from abc import ABC, abstractmethod
import warnings
from typing import TypeVar

T = TypeVar("T")

class PipelineComponent(ABC):
    """An abstract class that all pipeline components should inherit from.
    """

    def __init__(self):
        super().__init__()

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


    @abstractmethod
    def parse_train_kwargs(self, args_str: str | None = None) -> dict:
        """Extract training key word arguments from a string."""
        raise NotImplementedError("Sub-classes must implement `parse_train_kwargs`.")

    @staticmethod
    def extract_train_kwargs(kwargs_str: str | None = None) -> dict:
        """A standardized manner to parse keyword arguments.

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

    @staticmethod
    def extract_list(list_str: str, dtype: type = float) -> list:
        """Extract a list of elements from a string in format v0/v1/v2"""
        return [dtype(val) for val in list_str.split("/")]

    def _warn_ignored_inputs(self, **kwargs):
        for name, variable in kwargs.items():
            if variable is not None:
                warnings.warn(f"{self.__class__.__name__} ignores {name} input")

    def _require(self, arg: T | None, name: str) -> T:
        if arg is None:
            raise ValueError(f"{self.__class__.__name__} requires {name} to be defined")
        return arg