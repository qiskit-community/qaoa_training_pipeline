#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluator based on the estimator primitive."""

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


class AerEvaluator(BaseEvaluator):
    """Evaluates the energy using Qiskit Aer.

    This is an abstract class that implements an interface to Qiskit Aer.
    """

    def __init__(self, estimator):
        """Initialize the primitive based on the given input."""
        super().__init__()

        self.primitive = estimator

    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: Sequence[float],
        mixer: BaseOperator | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the expectation value of a cost operator given a set of parameters.

        Args:
            cost_op (SparsePauliOp): The cost operator that defines :math:`H_C`.
            params (List[float]): The parameters for QAOA. The length of this list will
                determine the depth of the QAOA.
            mixer (Optional[QuantumCircuit], optional): the mixer of the QAOA circuit.
                Defaults to None.
            initial_state (Optional[QuantumCircuit], optional): the initial state of the QAOA.
            ansatz_circuit (Optional[Union[QuantumCircuit, SparsePauliOp]]: the circuit to be used
                as ansatz. Can be given either directly as a `QuantumCircuit`, or as a `SparsePauliOp`,
                and the circuit is constructed from the corresponding time-evolution operator.

        """

        if isinstance(ansatz_circuit, SparsePauliOp):
            ansatz_op = ansatz_circuit
        elif ansatz_circuit is None:
            ansatz_op = cost_op
        else:
            raise NotImplementedError(
                "Custom ansatz circuits in format"
                f"{ansatz_circuit.__class__.__name__} are not yet supported."
            )

        circuit = qaoa_ansatz(
            ansatz_op,
            reps=len(params) // 2,
            mixer_operator=mixer,
            initial_state=initial_state,
        )

        # Prevents the edge case where the cost op is the identity.
        if len(circuit.parameters) != len(params):
            raise ValueError("The QAOA Circuit does not have the correct number of parameters. ")

        result = self.primitive.run([(circuit, cost_op, params)]).result()

        return float(result[0].data.evs)
