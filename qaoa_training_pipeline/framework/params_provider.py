#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
This module defines the ParamsProvider abstract base class, which serves as the foundation
for all QAOA parameter providers in the training pipeline. Parameter providers are responsible
for supplying QAOA angles that define the QAOA circuit. 

The ParamsProvider class defines an interface for performing the following tasks:
    - Providing QAOA angles through provide_params
    - Applying angle basis transformation functions via qaoa_angles_function
    - Serialization/deserialization through to_config/from_config
    - Parsing runtime arguments for command-line usage

Subclasses implement the abstract class to provide QAOA angles in different ways,
such as by database look-up, or diverse QAOA angles training methods.
"""


from abc import ABC, abstractmethod
import warnings
from typing import TypeVar

from qaoa_training_pipeline.framework.param_result import ParamResult
from qaoa_training_pipeline.training.functions import (
    BaseAnglesFunction,
    IdentityFunction,
)

T = TypeVar("T")


class ParamsProvider(ABC):
    """Abstract base class for providing QAOA angles in the training pipeline.

    ParamsProvider defines the interface for all parameter providers in the QAOA training
    pipeline. It manages the initial QAOA angle provisioning and provides methods for
    configuration management and command line input validation and parsing.

    Attributes:
        _qaoa_angles_function: Function that transforms QAOA angles to a different basis
        before they are used, e.g. the Fourier basis. Defaults to IdentityFunction
        (no transformation).

    Abstract methods that sub-classes implement:
        - provide_params: Provides QAOA angles to the next element of the pipeline.
        - from_config: Initializes the ParamsProvider from a configuration file.
        - to_config: Serializes the ParamsProvider to a config file
    """

    def __init__(
        self,
        *,
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ):
        """Initialize the parameter provider.

        Args:
            qaoa_angles_function: Optional function to transform QAOA angles to a different
            basis, e.g. Fourier. If None, uses IdentityFunction (no transformation).
        """
        self._qaoa_angles_function = qaoa_angles_function or IdentityFunction()

    @property
    def qaoa_angles_function(self) -> BaseAnglesFunction:
        """Get the QAOA angles transformation function.

        Returns:
            The BaseAnglesFunction instance used to transform QAOA angles.
        """
        return self._qaoa_angles_function

    @abstractmethod
    def provide_params(self) -> ParamResult:
        """Provide QAOA angles to the next element in the pipeline.

        This abstract method must be implemented by subclasses to define how QAOA
        angles are generated or retrieved.

        Returns:
            ParamResult object containing the QAOA angles and associated metadata.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `provide_params`.")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "ParamsProvider":
        """Create an instance of the parameter provider from a configuration dictionary.

        This class method enables deserialization of parameter providers from saved
        configurations, allowing pipeline reconstruction from stored settings.

        Args:
            config: Dictionary containing the configuration parameters needed to
            instantiate the parameter provider.

        Returns:
            An instance of ParamsProvider.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `from_config`.")

    @abstractmethod
    def to_config(self) -> dict:
        """Serialize the parameter provider to a configuration dictionary.

        Creates a serializable dictionary representation of the parameter provider's
        configuration. This is primarily used for tracking how results are generated
        and for logging purposes.

        Returns:
            Dictionary containing the serializable configuration of the used ParamsProvider.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `to_config`.")

    @staticmethod
    def extract_list(list_str: str, dtype: type = float) -> list:
        """Extract a list of typed elements from a slash-separated string.

        Parses a string containing values separated by forward slashes and converts
        each value to the specified data type. Necessary to parse command-line provided
        arguments

        Args:
            list_str: String containing values separated by '/' (e.g., "1.0/2.5/3.7").
            dtype: Type to convert each element to. Defaults to float.

        Returns:
            List of elements converted to the specified dtype.

        Example:
            >>> ParamsProvider.extract_list("1.0/2.5/3.7")
            [1.0, 2.5, 3.7]
            >>> ParamsProvider.extract_list("1/2/3", dtype=int)
            [1, 2, 3]
        """
        return [dtype(val) for val in list_str.split("/")]

    def _warn_ignored_inputs(self, **kwargs) -> None:
        """Warn about ignored input parameters.

        Raises a warning for any provided non-None keyword arguments that are not used by
        the given ParamsProvider implementation. Helps users identify when
        they're passing unnecessary or unsupported parameters.

        Args:
            **kwargs: Keyword arguments to check.
        """
        for name, variable in kwargs.items():
            if variable is not None:
                warnings.warn(f"{self.__class__.__name__} ignores {name} input")

    def _require(self, arg: T | None, name: str) -> T:
        """Helper method for input validation that raises a ValueError if a
        required argument is None.

        Args:
            arg: The argument to validate.
            name: Name of the argument for error messaging.

        Returns:
            The validated argument.

        Raises:
            ValueError: If the argument is None.
        """
        if arg is None:
            raise ValueError(f"{self.__class__.__name__} requires {name} to be defined")
        return arg

    def parse_runtime_kwargs(self, kwargs_str: str | None = None) -> dict:
        """Parse keyword arguments, usually provided via command-line, from a colon-separated
        string format, "k1:v1:k2:v2:...", and returns the arguments in a dictionary format.
        This enables dynamic parameter passing when running the pipeline from the command line.

        Args:
            kwargs_str: String containing keyword arguments in format "k1:v1:k2:v2:...".
            If None, returns an empty dictionary. List values should use
            forward slashes as separators (e.g., "params0:1.234/4.56").

        Returns:
            Dictionary mapping argument names to their string values.

        Raises:
            ValueError: If the kwargs_str has an odd number of colon-separated items
            (indicating malformed key-value pairs).

        Example:
            >>> provider.parse_runtime_kwargs("depth:3:optimizer:COBYLA")
            {'depth': '3', 'optimizer': 'COBYLA'}
            >>> provider.parse_runtime_kwargs("params:1.0/2.0/3.0")
            {'params': '1.0/2.0/3.0'}
        """
        if kwargs_str is None:
            return dict()

        items = kwargs_str.split(":")

        if len(items) % 2 != 0:
            raise ValueError(
                f"Malformed keyword arguments {kwargs_str}: should be k1:v1:k2:v2_...."
            )

        return {items[idx]: items[idx + 1] for idx in range(0, len(items), 2)}

