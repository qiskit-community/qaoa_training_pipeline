#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base trainer interface."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.training.functions import BaseAnglesFunction, IdentityFunction


class BaseTrainer(ABC):
    """An interface that all the trainers should satisfy"""

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        qaoa_angles_function: Optional[Callable] = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            evaluator: The class with which the energy should be evaluated.
            qaoa_angles_function: A function to convert optimization parameters into QAOA
                angles. By default, this is the identity function. Ideally, this argument is
                an instance of `BaseAnglesFunction` but we allow any callable here that maps
                optimization parameters to QAOA angles.

        """
        self._evaluator = evaluator
        self._qaoa_angles_function = qaoa_angles_function or IdentityFunction()

    @property
    def evaluator(self) -> BaseEvaluator:
        """Return the evaluator of the trainer."""
        return self._evaluator

    @property
    def qaoa_angles_function(self) -> BaseAnglesFunction:
        """Return the QAOA angles function of the trainer."""
        return self._qaoa_angles_function

    @property
    @abstractmethod
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""

    @abstractmethod
    def train(
        self,
        cost_op: SparsePauliOp,
        *args,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        **kwargs,
    ) -> ParamResult:
        """Performs the training."""
        raise NotImplementedError("Sub-classes must implement `train`.")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict):
        """Return an instance of the class based on a config."""
        raise NotImplementedError("Sub-classes must implement `from_config`.")

    def to_config(self) -> dict:
        """Creates a serializeable dictionary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": self._evaluator.to_config(),
            "qaoa_angles_function": self._qaoa_angles_function.to_config(),
        }

    @abstractmethod
    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Extract training key word arguments from a string."""

    @staticmethod
    def extract_train_kwargs(kwargs_str: Optional[str] = None) -> dict:
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
    def extract_list(list_str: str, dtype: type = float) -> List:
        """Extract a list of elements from a string in format v0/v1/v2"""
        return [dtype(val) for val in list_str.split("/")]
