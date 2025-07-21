#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pauli propagation-based QAOA evaluator."""

import importlib

from qiskit.circuit.library import QAOAAnsatz

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


# Safely import Julia if is is installed.
jl_loader = importlib.util.find_spec("juliacall")
HAS_JL = jl_loader is not None

if HAS_JL:
    from juliacall import Main as jl


class MPSEvaluator(BaseEvaluator):
    """Evaluator based on the Pauli propagation method.
    
    This class requires that the system has the Pauli propagation toolkit
    https://github.com/MSRudolph/PauliPropagation.jl installed. Note that this
    toolkit also require Julia. Therefore, it is not supported by the default
    requirements of the QAOA training pipeline and people need to install
    PauliPropagation and Julia themselves.
    """

        # pylint: disable=too-many-positional-arguments
    def __init__(self):
        """Initialize the Pauli propagation evaluator.
        
        TODO: variables that control the approximation should go here as is done in MPSEvaluator.
        """

        # Importing Julia can cause the kernel to crash, typically, on windows.
        # Therefore we first gracefully check for it before importing
        if not HAS_JL:
            raise ImportError(
                f"{self.__class__.__name__} requires Julia and the PauliPropagation.jl package."
                f"Please install Julia and the PauliPropagation.jl package."
                f"See https://github.com/MSRudolph/PauliPropagation.jl for more details."
            )


        # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> float:
        """Evaluate the QAOA circuit parameters."""

        if ansatz_circuit is not None:
            raise NotImplementedError(
                f"Custom Ansatz circuits are currently not supported in {self.__class__.__name__}."
            )

        circuit = QAOAAnsatz(cost_op, reps=reps, initial_state=initial_state, mixer_operator=mixer)

        pp_circuit = self._convert_to_pp(cirucit)

        # TODO Call the Pauli propagation circuit here
        energy = ...

        return energy

    def self._convert_to_pp(circuit: QuantumCircuit) -> "The PP structure":
        """Convert the Qiskit circuit into the representation required by Pauli propagation.
        
        Args:
            circuit: The Qiskit cirucit.

        Returns:
            A representation of the circuit on which we can call the Pauli propagation code.
        """

        # TODO Implement this.
