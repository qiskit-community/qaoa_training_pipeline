#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This class allows us to load existing optimized parameters."""

import glob
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.exceptions import TrainingError
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult


class OptimizedParametersLoader(BaseTrainer):
    """Class to load parameters from a file.

    In a slight abuse of notation, this class is not really a trainer. However,
    having it inherit from the BaseTrainer ensures that it is usable in the
    train pipeline.
    """

    def __init__(self):
        """Initialize a class instance."""
        super().__init__(None)

    @property
    def minimization(self):
        """Raises a warning as a loader neither minimizes nor maximizes."""
        raise ValueError(f"{self.__class__.__name__} neither minimizes nor maximizes.")

    # pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
        params0: list[float] | None = None,
        folder: str | None = None,
        file_pattern: str | None = None,
    ) -> ParamResult:
        """Load from a file.

        We will load from a file and take the parameters from the last trainer in the
        trainer chain. Crucially, this methods requires the file to be formatted following
        the output of `train.py` in this package.

        Args:
            cost_op: The operator for which we train. It is not needed here.
            folder: The folder where to find the parameters to load.
            file_pattern: The pattern to match to identify the file. This is a simple
                if file_pattern in file_name then load the data in the file.
            mixer: Not needed for now.
            initial_state: Not needed for now.
            ansatz_circuit: Not needed for now.
        """
        if folder is None:
            raise ValueError(
                f"class {self.__class__.__name__} requires a folder name to load the parameters."
            )
        if file_pattern is None:
            raise ValueError(
                f"class {self.__class__.__name__} requires a file pattern to load the parameters."
            )
        # 1. look for the file in the folder
        data, loaded_file_name = None, None
        for file_name in glob.glob(folder + "*.json"):
            if file_pattern in file_name:
                loaded_file_name = file_name

                with open(loaded_file_name, "r") as fin:
                    data = json.load(fin)

                break

        if data is None:
            raise TrainingError(f"No file found in {folder} containing {file_pattern}.")

        # 2. extract the last point from the file.
        keys = []
        for key in data.keys():
            try:
                keys.append(int(key))
            except ValueError:
                pass

        max_key = str(max(keys))

        param_result = ParamResult(data[max_key]["optimized_params"], np.nan, self, None)
        param_result["from_file"] = loaded_file_name

        return param_result

    @classmethod
    def from_config(cls, config: dict) -> "OptimizedParametersLoader":
        """Create an instance from a config."""
        return cls()

    def to_config(self) -> dict:
        """Creates a serializable dictionary to keep track of how results are created.

        Note: This data structure is not intended for us to recreate the class instance.
        """
        return {"trainer_name": self.__class__.__name__}

    def parse_train_kwargs(self, args_str: str | None = None) -> dict:
        """Parse the train args, i.e., get file and folder names.

        The string should have the format `folder:folder_name:file_pattern:pattern`.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key in ["folder", "file_pattern"]:
                train_kwargs[key] = str(val)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs
