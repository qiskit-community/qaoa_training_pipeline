#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base trainer interface."""

from abc import ABC, abstractmethod
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.param_result import ParamResult


class BaseTrainer(ABC):
    """An interface that all the trainers should satisfy"""

    def __init__(self, evaluator: Optional[BaseEvaluator] = None) -> None:
        """Initialise the trainer."""
        self._evaluator = evaluator

    @property
    def evaluator(self) -> BaseEvaluator:
        """Return the evaluator of the trainer."""
        return self._evaluator

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
        }

    @abstractmethod
    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Extract training key word arguments from a string."""
