#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to store result data."""

from dataclasses import dataclass
import platform
from typing import Optional, TYPE_CHECKING
import numpy as np

from qaoa_training_pipeline.training.history_mixin import HistoryMixin

if TYPE_CHECKING:
    from qaoa_training_pipeline.training.base_trainer import BaseTrainer


@dataclass
class ParamResult:
    """A class to store the results of a parameter optimization.

    This class ensures that we have elementary information such as information on the
    platform in addition to training duration and parameters. The class includes a
    `qaoa_training_pipeline_version` variable. This variable should be updated manually
    in each new commit to the repository. The addeded functionality is tracked in
    a table in the main README.md.
    """

    data: dict

    def __init__(
        self,
        optimized_params: list,
        duration: float,
        trainer: "BaseTrainer",
        energy: Optional[float] = None,
    ):
        """Initialize the data class."""
        self.data = {}

        self.data["system_info"] = {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "qaoa_training_pipeline_version": 27,
        }

        # Convert, e.g., np.float to float
        self.data["optimized_params"] = [float(val) for val in optimized_params]
        self.data["optimized_qaoa_angles"] = [
            float(val) for val in trainer.qaoa_angles_function(optimized_params)
        ]
        self.data["train_duration"] = duration
        self.data["energy"] = float(energy) if isinstance(energy, np.floating) else energy
        self.data["trainer"] = trainer.to_config()

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        """Return the keys of the underlying dict"""
        return self.data.keys()

    def update(self, other: dict):
        """Update the data with the given dictionary."""
        self.data.update(other)

    def add_history(self, history_mixin: HistoryMixin):
        """Add the history to the data."""
        self.data["energy_history"] = history_mixin.energy_history
        self.data["parameter_history"] = history_mixin.parameter_history
        self.data["energy_evaluation_time"] = history_mixin.energy_evaluation_time

    # pylint: disable=(too-many-positional-arguments
    @classmethod
    def from_scipy_result(cls, result, params0, train_duration, sign, trainer) -> dict:
        """Standardizes results from SciPy such that it can be serialized."""

        param_result = cls(result.pop("x").tolist(), train_duration, trainer, sign * result.pop("fun"))

        result = dict(result)

        param_result["x0"] = params0

        # Serialize the success boolean to avoid json issues
        if "success" in result:
            success = result["success"]
            param_result["success"] = f"{success}"

        if isinstance(trainer, HistoryMixin):
            param_result.add_history(trainer)

        return param_result
