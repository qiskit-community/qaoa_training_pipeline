#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class inheriting from ParamsProvider used to provide initial QAOA angles via configuration"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qaoa_training_pipeline.framework.param_result import ParamResult
from qaoa_training_pipeline.framework.params_provider import ParamsProvider
from qaoa_training_pipeline.training.functions import FUNCTIONS

if TYPE_CHECKING:
    from qaoa_training_pipeline.training.functions import BaseAnglesFunction


class FromConfigParamsProvider(ParamsProvider):
    """Class for providing initial QAOA angles or parameters from a configuration file"""

    def __init__(self, params0: list, qaoa_angles_function: BaseAnglesFunction):
        """Initialize the parameter provider.

        Args:
            params0: the initial parameters to pass to the next pipeline component
        """
        super().__init__(qaoa_angles_function=qaoa_angles_function)
        self._params0 = params0

    def provide_params(self) -> ParamResult:
        """Provide QAOA angles to the next element in the pipeline.

        Returns:
            ParamResult object containing the QAOA angles
        """
        return ParamResult(self._params0, 0, self, None)

    @classmethod
    def from_config(cls, config: dict) -> "FromConfigParamsProvider":
        """Create an instance of the parameter provider

        Args:
            config: Dictionary containing the initial parameters to initialize the class

        Returns:
            An instance of FromConfigParamsProvider.

        """
        qaoa_angles_function = FUNCTIONS[config["qaoa_angles_function"]]
        return cls(
            config["params0"],
            qaoa_angles_function(**config.get("qaoa_angles_function_init", {})),
        )

    def to_config(self) -> dict:
        return {
            "provider_name": self.__class__.__name__,
            "provider_init": {"params0": self._params0},
        }
