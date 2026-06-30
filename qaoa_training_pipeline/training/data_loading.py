#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods to extra data from databases."""

from abc import ABC, abstractmethod
import json
import numpy as np


class BaseDataLoader(ABC):
    """A base class to define the interface of QAOA angle functions."""

    @abstractmethod
    def __call__(self) -> dict:
        """Extract data from a database."""

    def to_config(self) -> dict:
        """Creates a serializable dictionary of the class."""
        return {"function_name": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the loader from a config."""


class TrivialDataLoader(BaseDataLoader):
    """Allows us to pass data to classes without going through files."""

    def __init__(self, data):
        """The data loader holds the data."""
        self._data = data

    def __call__(self):
        """Load the data."""
        return self._data

    def to_config(self) -> dict:
        """Creates a serializable dictionary of the class."""
        config = super().to_config()
        config["data"] = self._data

        return config

    @classmethod
    def from_config(cls, config: dict) -> BaseDataLoader:
        """Initialize the loader from a config."""
        return cls(config["data"])


class LoadFromJson(BaseDataLoader):
    """Loads data from a json file.

    The loader expects that json file to contain keys that can be converted
    to a tuple of floats that correspond to features of problem instances.
    For example, `2, 6, 9, 3.0, -0.5, 0.6` is a valid key which will be
    converted to the tuple of features `(2.0, 6.0, 9.0, 3.0, -0.5, 0.6)`.
    """

    def __init__(self, file_name: str, nested: bool = False):
        """Setup the loader by specifying the file from which to load."""
        self._file_name = file_name
        self._nested = nested

    def __call__(self):
        """Load the data."""
        with open(self._file_name, "r") as fin:
            json_data = json.load(fin)

        if self._nested:
            return self._load_nested(json_data)
        else:
            return self._load_flat(json_data)

    def _load_nested(self, json_data: dict) -> dict:
        """Load nested structure and flatten to tuple keys.

        Transforms: {degree: {reps: {beta, gamma, AR}}}
        To: {(reps, degree): {qaoa_angles: [beta + gamma]}}

        Note: This transformation on the data needs to be used when using
        the fixed angles conjecture method
        """
        data = dict()

        for outer_key, inner_dict in json_data.items():
            # Skip non-dict entries (like "description")
            if not isinstance(inner_dict, dict):
                continue

            # Convert outer key to float
            try:
                outer_val = float(outer_key)
            except ValueError:
                # Skip keys that can't be converted (like "description")
                continue

            for inner_key, angles_dict in inner_dict.items():
                # Convert inner key to int (for reps/qaoa_depth)
                inner_val = int(inner_key)

                # Create tuple key: (qaoa_depth, degree)
                tuple_key = (inner_val, outer_val)

                # Transform angles to TransferTrainer format
                if "beta" in angles_dict and "gamma" in angles_dict:
                    qaoa_angles = angles_dict["beta"] + angles_dict["gamma"]
                    data[tuple_key] = {"qaoa_angles": np.atleast_2d(qaoa_angles)}

                    # Preserve additional metadata if present
                    if "AR" in angles_dict:
                        data[tuple_key]["approximation_ratio"] = angles_dict["AR"]
                else:
                    # If not in fixed-angle format, store as-is
                    data[tuple_key] = angles_dict

        return data

    def _load_flat(self, json_data):
        """Load the features from a given database to tuple keys

        Note: This is needed to use parameter transfer from a given dataset of pre-trained
        QAOA angles to a given problem instance.
        """
        data = dict()
        for key, val in json_data.items():
            tuple_key = tuple(float(val) for val in key.split(","))
            data[tuple_key] = val
        return data

    def to_config(self):
        config = super().to_config()
        config["file_name"] = self._file_name

        return config

    @classmethod
    def from_config(cls, config) -> "LoadFromJson":
        """Setup the loader from a config file."""
        return cls(config["file_name"], config["nested"])


DATA_LOADERS = {
    "LoadFromJson": LoadFromJson,
    "TrivialDataLoader": TrivialDataLoader,
}
