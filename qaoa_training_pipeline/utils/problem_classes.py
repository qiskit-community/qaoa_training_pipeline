#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to create cost operators of known problem classes from input."""

from typing import Optional

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.applications import StableSet
from qiskit_optimization.converters import QuadraticProgramToQubo

from qaoa_training_pipeline.utils.data_utils import input_to_operator
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph


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
            assert input_str
            penalty = float(input_str)
        except (ValueError, AssertionError):
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

    DEFAULT_N = 10  # pylint: disable=invalid-name

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

    def _build_hamiltonian_terms(self, num_spins: int) -> tuple[list[tuple], int]:
        """Build the LABS Hamiltonian terms and compute the constant offset.

        The LABS energy is defined as E = Σₖ Cₖ² where Cₖ = Σᵢ sᵢ·sᵢ₊ₖ (autocorrelation).
        Expanding the square gives ZZ (quadratic) and ZZZZ (quartic) Pauli terms.

        Parameters
        ----------
        num_spins : int
            Number of spins (problem size N)

        Returns
        -------
        terms : list of tuples
            Each tuple is (weight, indices) where indices are the qubit positions
            for Z operators. E.g., (4, (0, 1, 2, 3)) means 4·Z₀Z₁Z₂Z₃
        offset : int
            Constant energy offset from identity terms (excluded from Hamiltonian)
        """
        all_terms = set()
        offset = 0

        for lag in range(1, num_spins):
            offset += num_spins - lag
            for i in range(num_spins - lag):
                for j in range(i + 1, num_spins - lag):
                    if j == i + lag:
                        indices = (i, j + lag)
                        all_terms.add((2, indices))
                    else:
                        indices = tuple(sorted((i, i + lag, j, j + lag)))
                        all_terms.add((4, indices))

        return list(all_terms), offset

    def cost_operator(
        self, input_data: Optional[dict] = None  # pylint: disable=unused-argument
    ) -> SparsePauliOp:
        """Create the cost operator from the problem size N.

        Note: The input_data is ignored for LABS, as the problem
        is fully defined by N provided at initialization.

        Note: LABS is a minimization problem. Use energy_minimization=True
        in ScipyTrainer config when optimizing LABS.

        Args:
            input_data: An optional dict, ignored.

        Returns:
            A SparsePauliOp representing the LABS Hamiltonian H_C.
        """
        num_qubits = self._num_qubits
        terms, offset = self._build_hamiltonian_terms(num_qubits)

        pauli_list = []

        for weight, nodes in terms:
            paulis = ["I"] * num_qubits
            for idx in nodes:
                if idx >= num_qubits:
                    raise IndexError(f"Node index {idx} out of bounds for N={num_qubits}")
                paulis[idx] = "Z"

            pauli_string = "".join(paulis)[::-1]
            pauli_list.append((pauli_string, weight))

        if offset > 0:
            identity_string = "I" * num_qubits
            pauli_list.append((identity_string, offset))

        return SparsePauliOp.from_list(pauli_list)

    @staticmethod
    def is_labs_problem(cost_op, tolerance: float = 1e-10) -> bool:
        """Check if a SparsePauliOp represents a LABS problem.

        This function verifies if the cost operator matches the structure of a LABS problem
        by checking:
        1. All terms are either quadratic (2 Z operators) or quartic (4 Z operators)
        2. Quartic terms have weight 4.0 (LABS-specific)
        3. The problem has at least some quartic terms (distinguishes from pure quadratic problems)

        Note: This is a heuristic check. Other non-quadratic problems might also pass this test
        if they have the same structure. For a definitive check, you would need to verify
        the exact term structure matches what _build_hamiltonian_terms() would produce.

        Parameters
        ----------
        cost_op : SparsePauliOp
            The cost operator to check
        tolerance : float
            Numerical tolerance for weight comparison

        Returns
        -------
        bool
            True if the cost operator appears to be a LABS problem, False otherwise
        """
        has_quartic_terms = False
        has_invalid_terms = False

        for pauli, coeff in cost_op.to_list():
            z_count = pauli.count("Z")

            if z_count not in {2, 4}:
                if z_count > 0:
                    has_invalid_terms = True
                    break

            if z_count == 4:
                has_quartic_terms = True
                abs_coeff = abs(coeff)
                if abs(abs_coeff - 4.0) > tolerance:
                    has_invalid_terms = True
                    break

        return has_quartic_terms and not has_invalid_terms

    @staticmethod
    def compute_energy(spins: np.ndarray) -> float:
        """Compute the LABS energy for a spin configuration.

        The LABS energy is E = Σₖ Cₖ² where Cₖ is the autocorrelation at lag k:
        Cₖ = Σᵢ sᵢ·sᵢ₊ₖ

        Parameters
        ----------
        spins : np.ndarray
            Array of spins with values in {-1, +1}

        Returns
        -------
        float
            The LABS energy value
        """
        num_spins = len(spins)
        total_energy = 0.0

        for lag in range(1, num_spins):
            autocorr = np.dot(spins[:-lag], spins[lag:])
            total_energy += autocorr * autocorr

        return total_energy

    @staticmethod
    def compute_energy_from_bits(bits: np.ndarray) -> float:
        """Compute LABS energy from a bitstring (0/1 encoding).

        Converts {0, 1} bits to {+1, -1} spins and computes energy.

        Parameters
        ----------
        bits : np.ndarray
            Array of bits with values in {0, 1}

        Returns
        -------
        float
            The LABS energy value
        """
        spins = 1 - 2 * bits
        return LABS.compute_energy(spins)


PROBLEM_CLASSES = {
    "maxcut": MaxCut,
    "mis": MaxIndependentSet,
    "labs": LABS,
}
