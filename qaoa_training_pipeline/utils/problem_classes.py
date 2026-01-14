#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to create cost operators of known problem classes from input."""

from typing import Optional

from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.applications import StableSet
from qiskit_optimization.converters import QuadraticProgramToQubo

from qaoa_training_pipeline.utils.data_utils import input_to_operator
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph

from qaoa_training_pipeline.utils.labs.labs_utils import get_terms_offset


class MaxCut:
    """Produce max-cut operators from input graphs."""

    # pylint: disable=unused-argument
    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "MaxCut":
        """Create the class. Note that the input string is not used."""
        return cls()

    def cost_operator(self, input_data: dict):
        """Create the cost operator from the input.

        The (weighted) edges of the graph are converted to ZZ with a prefoactor of -0.5.

        Args:
            input: A dict specifying a graph where the keys are tuples of ints representing
                edges or hyper-edge and the values are the weights.
        """
        return input_to_operator(input_data, pre_factor=-0.5)


class MaxIndependentSet:
    """Produce max independent set operators from input graphs."""

    DEFAULT_PENALTY = 2.0

    def __init__(self, penalty: Optional[float] = None):
        """Create the maximum independent set class.

        Args:
            penalty: The penalty to use for the cost operator. Defaults to 2.0.
        """
        self._penalty = penalty or self.DEFAULT_PENALTY

    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "MaxIndependentSet":
        """Create the class from a string.

        Args:
            input_str: If given, it will be converted to a float representing the
                penalty in from of the edge constraints.
        """
        try:
            penalty = float(input_str)
        except ValueError:
            penalty = cls.DEFAULT_PENALTY

        return cls(penalty)

    def cost_operator(self, input_data: dict):
        """Create the cost operator from the input graph given as dict.

        Args:
            input_data: A dict specifying a graph where the keys are tuples of ints representing
                edges or hyper-edge and the values are the weights.
        """
        graph = dict_to_graph(input_data)
        qp_ = StableSet(graph).to_quadratic_program()
        cost_op, _ = QuadraticProgramToQubo(penalty=self._penalty).convert(qp_).to_ising()

        return cost_op


class LABS:
    """Produce LABS cost operators for a given problem size N.

    Note: LABS has quartic (4-body) terms in the Hamiltonian, so it is NOT compatible
    with EfficientDepthOneEvaluator, which only supports quadratic (2-body) terms.
    """

    DEFAULT_N = 10  # Default problem size if not specified  # pylint: disable=invalid-name

    def __init__(self, num_qubits: Optional[int] = None):
        """Create the LABS problem class.
        Args:
            num_qubits: The problem size (number of spins).
        """
        self._num_qubits = num_qubits or self.DEFAULT_N
        if self._num_qubits <= 0:
            self._num_qubits = self.DEFAULT_N

    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "LABS":
        """Create the class from a string specifying N.

        Args:
            input_str: String specifying problem size (e.g., "15" for N=15).
        """
        num_qubits = None
        if input_str:
            try:
                num_qubits = int(input_str)
            except ValueError:
                pass  # Will use DEFAULT_N
        return cls(num_qubits)

    def cost_operator(
        self, input_data: Optional[dict] = None  # pylint: disable=unused-argument
    ) -> SparsePauliOp:
        """Create the cost operator from the problem size N.

        Note: The input_data is ignored for LABS, as the problem
        is fully defined by N provided at initialization.

        The Hamiltonian is NEGATED so that lower LABS energy corresponds to
        higher eigenvalue, matching the convention used by MaxCut/MIS
        (best solution = highest eigenvalue = maximize).

        Args:
            input_data: An optional dict, ignored.

        Returns:
            A SparsePauliOp representing the negated LABS Hamiltonian -H_C.
        """
        num_qubits = self._num_qubits
        terms, _ = get_terms_offset(
            num_qubits
        )  # We ignore the offset since it drops out in the final calculation

        pauli_list = []

        for weight, nodes in terms:
            # Create a base of all 'I's
            paulis = ["I"] * num_qubits
            for idx in nodes:
                if idx >= num_qubits:
                    # This should not happen with a correct get_terms_offset
                    raise IndexError(f"Node index {idx} out of bounds for N={num_qubits}")
                # Place 'Z' at the specified indices
                paulis[idx] = "Z"

            # Reverse the string for Qiskit's little-endian convention
            pauli_string = "".join(paulis)[::-1]
            # Negate weight so best LABS solution has highest eigenvalue
            pauli_list.append((pauli_string, -weight))

        # The return value is the SparsePauliOp (negated)
        return SparsePauliOp.from_list(pauli_list)

    @staticmethod
    def post_process_result(cost_op: SparsePauliOp, result) -> dict:
        """Post-process training result to add LABS-specific metrics.

        Computes labs_energy, p_opt, tts for the main result and any nested results
        (e.g., from RecursionTrainer).

        Note: Since the Hamiltonian is negated, the energy values need to be
        negated back to get the true LABS energy.

        Args:
            cost_op: The LABS cost operator (negated)
            result: Training result (ParamResult or dict)

        Returns:
            Updated result dict with LABS metrics
        """
        from qaoa_training_pipeline.utils.labs.labs_utils import (  # pylint: disable=import-outside-toplevel
            process_labs_post_optimization,
            true_optimal_energy,
        )

        # Process the main result (handles negated Hamiltonian internally)
        result = process_labs_post_optimization(
            cost_op=cost_op, param_result=result, hamiltonian_negated=True
        )

        # Add optimal energy for reference
        if cost_op.num_qubits in true_optimal_energy:
            result["optimal_energy"] = true_optimal_energy[cost_op.num_qubits]

        # Process nested results (e.g., from RecursionTrainer with numeric keys 2, 3, etc.)
        data = result.data if hasattr(result, "data") else result
        for key in list(data.keys()):
            is_numeric_key = isinstance(key, int) or (isinstance(key, str) and key.isdigit())
            if is_numeric_key:
                nested = data[key]
                if isinstance(nested, dict) and "optimized_qaoa_angles" in nested:
                    nested = process_labs_post_optimization(
                        cost_op=cost_op, param_result=nested, hamiltonian_negated=True
                    )
                    if cost_op.num_qubits in true_optimal_energy:
                        nested["optimal_energy"] = true_optimal_energy[cost_op.num_qubits]
                    data[key] = nested

        return result


PROBLEM_CLASSES = {
    "maxcut": MaxCut,
    "mis": MaxIndependentSet,
    "labs": LABS,
}
