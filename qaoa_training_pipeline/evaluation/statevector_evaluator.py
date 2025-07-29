#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Statevector-based QAOA evaluator."""

from typing import Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


class StatevectorEvaluator(BaseEvaluator):
    """Evaluates the energy of a QAOA circuit with Qiskit's StatevectorSimulator.

    This evaluator naturally does not scale to large problem sizes but is useful
    when working with small-scale problem instances.
    """

    def __init__(self, statevector_init_args: Optional[Dict] = None) -> None:
        """Initialize the statevector evaluator.

        Args:
            statevector_init_args: The arguments to initialize the StatevectorSimulator with.
        """
        super().__init__()

        self._init_args = statevector_init_args or {}
        self.primitive = StatevectorEstimator(**self._init_args)

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> float:
        """Evaluate the expectation value of a cost operator given a set of parameters.

        Args:
            cost_op (SparsePauliOp): The cost operator that defines :math:`H_C`.
            params (List[float]): The parameters for QAOA. The length of this list will
                determine the depth of the QAOA.
            mixer (Optional[QuantumCircuit], optional): the mixer of the QAOA circuit.
                Defaults to None.
            initial_state (Optional[QuantumCircuit], optional): the initial state of the QAOA.
        """

        if ansatz_circuit is not None:
            raise NotImplementedError("Custom ansatz circuits are not yet supported.")

        circuit = qaoa_ansatz(
            cost_op,
            reps=len(params) // 2,
            mixer_operator=mixer,
            initial_state=initial_state,
        )

        result = self.primitive.run([(circuit, cost_op, params)]).result()

        return float(result[0].data.evs)

    def to_config(self) -> dict:
        config = super().to_config()
        config["statevector_init_args"] = self._init_args

        return config

    @classmethod
    def from_config(cls, config: dict) -> "StatevectorEvaluator":
        """Initialize the evaluator from a configuration dictionary."""
        return cls(**config)
