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

from qaoa_training_pipeline.qaoa_training_pipeline.params_provider import ParamsProvider
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.training.functions import BaseAnglesFunction
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class ProblemParamsProvider(ParamsProvider):
    """Abstract base class for providing QAOA angles that are problem dependent
    in the training pipeline.

    ProblemParamsProvider defines the interface for parameter providers in the QAOA training
    pipeline that need the problem cost operator.

    Abstract methods that sub-classes implement:
        - provide_params: Provides QAOA angles to the next element of the pipeline.
    """

    def __init__(
        self,
        *,
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ):
        """Initialize the parameter provider.

        Args:
            qaoa_angles_function: Optional function to transform QAOA angles to a different
            basis, e.g. Fourier. If None, uses IdentityFunction (no transformation).
        """
        super().__init__(qaoa_angles_function=qaoa_angles_function)

    @abstractmethod
    def provide_params(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit,
        initial_state: QuantumCircuit,
        ansatz_circuit: QuantumCircuit,
    ) -> ParamResult:
        """Provide QAOA angles to the next element in the pipeline, based on the problem 
        cost operator.

        This abstract method must be implemented by subclasses to define how QAOA
        angles are generated or retrieved using the cost operator provided.

        Args:
            cost_op: The problem cost operator.
            mixer: The mixer operator.
            initial_state: The initial state of the problem.
            ansatz_circuit: The ansatz circuit used to generate the QAOA angles.

        Returns:
            ParamResult object containing the QAOA angles and associated metadata.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `provide_params`.")
