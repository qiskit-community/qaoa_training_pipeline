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

    def __init__(self, file_name: str):
        """Setup the loader by specifying the file from which to load."""
        self._file_name = file_name

    def __call__(self):
        """Load the data."""
        with open(self._file_name, "r") as fin:
            json_data = json.load(fin)

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
        return cls(config["file_name"])


DATA_LOADERS = {
    "LoadFromJson": LoadFromJson,
    "TrivialDataLoader": TrivialDataLoader,
}
