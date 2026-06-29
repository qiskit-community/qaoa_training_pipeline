#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
This module defines the ProblemParamsProvider abstract base class, which serves as the foundation
for QAOA parameter providers that need the cost operator to generate/retrieve the angles. 


Subclasses implement the abstract class to provide QAOA angles in different ways,
such as by database look-up, or diverse QAOA angles training methods, all using the problem 
cost operator.
"""


from abc import abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.framework.params_provider import ParamsProvider
from qaoa_training_pipeline.framework.param_result import ParamResult


class ProblemParamsProvider(ParamsProvider):
    """Abstract base class to provide QAOA angles dependent on the problem.

    ProblemParamsProvider defines the interface for parameter providers in the QAOA training
    pipeline that need information on how the QAOA ansatz is constructed.

    Abstract methods that sub-classes implement:
        - provide_params: Provides QAOA angles to the next element of the pipeline.
    """

    @abstractmethod
    # pylint: disable=too-many-positional-arguments,arguments-differ
    def provide_params(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit,
        initial_state: QuantumCircuit,
        ansatz_circuit: QuantumCircuit,
    ) -> ParamResult:
        """Provide QAOA angles to the next element in the pipeline, based on the problem.

        This abstract method must be implemented by subclasses. It defines how QAOA
        angles are generated or retrieved using elements that enter the QAOA ansatz
        and the cost function.

        Args:
            cost_op: The problem cost operator.
            mixer: The mixer operator.
            initial_state: The initial state of the QAOA ansatz.
            ansatz_circuit: The ansatz that generates the cost layer.

        Returns:
            ParamResult object containing the QAOA angles and associated metadata.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `provide_params`.")

