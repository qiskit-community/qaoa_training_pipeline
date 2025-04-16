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
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.exceptions import TrainingError


class OptimizedParametersLoader(BaseTrainer):
    """Class to load parameters from a file.

    In a slight abuse of notation, this class is not really a trainer. However,
    having it inherite from the BaseTrainer ensures that it is usable in the
    train pipeline.
    """

    def __init__(self):
        """Initialize a class instance."""
        super().__init__(None)

    @property
    def minimization(self):
        """Raises a warning as a loader neither minimizes nor maximizes."""
        raise ValueError(f"{self.__class__.__name__} neither minimizes nor maximizes.")

    # pylint: disable=arguments-differ
    def train(
        self,
        cost_op: SparsePauliOp,
        folder: str,
        file_pattern: str,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> Dict[str, Any]:
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

        return {
            "optimized_params": data[max_key]["optimized_params"],
            "from_file": loaded_file_name,
            "trainer": self.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict) -> "OptimizedParametersLoader":
        """Create an intance from a config."""
        return cls()

    def to_config(self) -> dict:
        """Creates a serializeable dictionary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        return {"trainer_name": self.__class__.__name__}

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse the train args, i.e., get file and folder names.

        Note: since underscore is a common delimiter in file names here we will use
        the @ character to split folder and file name. Therefore we expect strings
        of the form "my_folder_name@my_file_pattern."
        """
        if args_str is None:
            return dict()

        split_args = args_str.split("@")

        return {
            "folder": split_args[0],
            "file_pattern": split_args[1],
        }
