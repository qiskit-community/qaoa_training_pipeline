#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Defines the methods that an evaluator should have."""

from abc import ABC, abstractmethod
from typing import Dict, Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class BaseEvaluator(ABC):
    """Defines the interface for the evaluators.

    This makes the evaluators pluggable in parameter trainers. This base class
    is designed to be as light-weight as possible to (i) not constrain development
    and (ii) base classes are notoriously hard to remove from code.
    """

    # pylint: disable=too-many-positional-arguments
    @abstractmethod
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: Sequence[float],
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the energy for the given arguments.

        Args:
            cost_op: The cost operator that defines the cost Hamiltonian.
            params: The parameters for QAOA. The length of this sequence will
                determine the depth of the QAOA.
            mixer: The mixer operator. Defaults to None.
            initial_state: The initial state circuit. Defaults to None.
            ansatz_circuit: The ansatz circuit for the cost operator. Defaults to None.

        Returns:
            The energy as a real value.
        """

    def get_results_from_last_iteration(self) -> Dict:
        """Function to access results of the simulation at the last iteration

        This function (which should be overriden by derived class) should be used
        to retrieve results from the last call of the evaluator that are different
        from the energy.
        For instance, if there is an approximate simulator, it would be useful to
        know the accuracy of the energy estimate at the last simulation step.
        This function enables to access these data.

        By default, an empty dictionary is returned (if the method is not defined
        in the derived class)

        Returns:
            Dict: dictionary with results from the last optimizer iteration
        """
        return {}

    def to_config(self) -> dict:
        """Json serializable config to keep track of how results are generated."""
        return {"name": self.__class__.__name__}

    @classmethod
    # pylint: disable=unused-argument
    def parse_init_kwargs(cls, init_kwargs: str | None = None) -> dict:
        """A hook that sub-classes can implement to parse initialization kwargs."""
        return dict()
