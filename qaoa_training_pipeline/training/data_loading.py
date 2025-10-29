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
    def __call__(self):
        """Extract data from a database."""

    def to_config(self) -> dict:
        """Creates a serializeable dictionary of the class."""
        return {"function_name": self.__class__.__name__}
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the loader from a config."""


class LoadFromJson(BaseDataLoader):
    """Loads data from a json file."""

    def __init__(self, file_name: str):
        """Setup the loader by specifying the file from which to load."""
        self._file_name = file_name

    def __call__(self):
        """Load the data."""
        with open(self._file_name, "r") as fin:
            data = json.load(fin)

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
}