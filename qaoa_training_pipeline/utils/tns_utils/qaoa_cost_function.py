# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These classes are used to construct the MPO representation of the QAOA cost function."""

from typing import Optional, List, Tuple

from qiskit.quantum_info import SparsePauliOp, Pauli

from quimb.tensor import (
    # SpinHam1D,
    MatrixProductOperator,
)

from .symbolic_mpo import SymbolicMPOConstruction


class MPOWrapper:
    """Wrapper around a `quimb` `MatrixProductOperator` object"""

    def __init__(self, input_mpo: MatrixProductOperator):
        """Class constructor

        Args:
            input_mpo (MatrixProductOperator): Matrix Product Operator
                encoded in the `quimb` format
        """
        self._matrix_product_operator = input_mpo

    @property
    def mpo(self) -> MatrixProductOperator:
        """Getter for the MPO"""
        return self._matrix_product_operator

    def get_bond_dimensions(self) -> List[int]:
        """Getter for the matrix product operator bond dimensions

        Returns:
            List[int]: maximum bond dimension for each inner bond
        """
        return [
            self._matrix_product_operator.ind_sizes()[inner_ind]
            for inner_ind in self._matrix_product_operator.inner_inds()
        ]

    def get_max_bond_dimension(self) -> int:
        """Getter for the *maximum* bond dimension.

        Returns:
            int: maximum value of the MPO bond dimension
        """
        return max(self.get_bond_dimensions())


class QAOACostFunction:
    """Matrix Product Operator representation of the QAOA cost function

    Limitations:
      * Supports only up to four-body cost functions
    """

    def __init__(
        self,
        sparse_pauli: SparsePauliOp,
        truncation: Optional[float] = None,
        max_bond: Optional[int] = None,
    ):
        """Class constructor

        Args:
            sparse_pauli (SparsePauliOp): Pauli representation of the cost function.
            truncation (Optional[float]): truncation threshold for the MPO construction.
            max_bond (Optional[int]): maximum bond dimension of the MPO.

        Raise:
            NotImplementedError: for cost functions including five- and higher-order
            many-body couplings.
        """
        # MPO constructor
        self._n_qubits = sparse_pauli.num_qubits
        self._sparse_pauli_op = sparse_pauli
        # self._mpo_builder = SpinHam1D(S=1 / 2, cyclic=False)
        # self._long_range_terms = []
        self._symbolic_mpo = SymbolicMPOConstruction(self._n_qubits)
        self.add_sparse_pauli_op(sparse_pauli)

        # Lazy construction of the MPO.
        self._mpo = None

        # Parameters for the MPO construction.
        self._truncation = truncation
        self._max_bond = max_bond

    @property
    def mpo(self):
        """Return the MPOWrapper of the cost function."""
        if self._mpo is None:
            self._mpo = self.return_mpo_representation()

        return self._mpo

    def add_sparse_pauli_op(self, new_sparse_pauli: SparsePauliOp) -> None:
        """Adds a new sparse Pauli op to the existing one.

        Args:
            new_sparse_pauli (SparsePauliOp): sparse Pauli operator to be added

        Raise:
            KeyError: if the number of qubits of the sparse Pauli don't match
        """
        if new_sparse_pauli.num_qubits != self._n_qubits:
            raise KeyError("Number of qubits not coherent")

        # NOTE: We keep for the moment the old implementation of the MPO construction
        #       for having a reference implementation in case there are problems
        # .     with the symbolic constructor.
        #
        # from quimb import pauli
        # for i_label, i_coeff in new_sparse_pauli.to_list():
        #     # Isolates the Coefficient
        #     pair_representation = self._convert_string_to_label(i_label)

        #     # The constant term is added wlog to the first site
        #     if len(pair_representation) == 0:
        #         self._mpo_builder[0] += i_coeff, "I"

        #     # One-body contribution. The factor of 2 comes from the fact that the
        #     # `_mpo_builder` object works with S_x/S_y/S_z operators, while in
        #     # QAOA one works with sigma_x, sigma_y, sigma_z. For this reason,
        #     # we add a factor of 2 (and 4 below) to compensate for the 1/2 factor
        #     # included in each spin operator
        #     elif len(pair_representation) == 1:
        #         self._mpo_builder[pair_representation[0][0]] += (
        #             i_coeff * 2,
        #             pair_representation[0][1],
        #         )

        #     # Nearest-neighbour two-body term.
        #     elif (
        #         len(pair_representation) == 2
        #         and abs(pair_representation[0][0] - pair_representation[1][0]) <= 1
        #     ):
        #         # Note that here the sites *must* be sorted in increasing index.
        #         self._mpo_builder[pair_representation[1][0], pair_representation[0][0]] += (
        #             i_coeff * 4,
        #             pair_representation[1][1],
        #             pair_representation[0][1],
        #         )

        #     # Non-nearest-neighbour two-body and higher-order couplings.
        #     else:
        #         self._long_range_terms.append(
        #             [
        #                 [i[0] for i in reversed(pair_representation)],  # All the qubits
        #                 [i[1] for i in reversed(pair_representation)],  # All Paulis
        #                 i_coeff,
        #             ]
        #         )

        # Updates the symbolic MPO
        for i_terms in new_sparse_pauli.to_list():
            self._symbolic_mpo.add_term(Pauli(i_terms[0][::-1]), i_terms[1])

    def return_mpo_representation_symbolic(self) -> MPOWrapper:
        """Returns the Matrix Product Operator (MPO) representation of the cost function
        using the symbolic MPO constructor

        Returns:
            MPOWrapper: wrapper around the MPO representation of the cost function
        """

        returned_mpo = self._symbolic_mpo.generate_mpo_representation()

        # Final truncation
        if self._truncation is not None:
            returned_mpo.compress(cutoff=self._truncation)

        if self._max_bond is not None:
            returned_mpo.compress(max_bond=self._max_bond)

        return MPOWrapper(returned_mpo)

    def return_mpo_representation(self) -> MPOWrapper:
        """Returns the Matrix Product Operator (MPO) representation of the cost function

        Returns:
            MPOWrapper: wrapper around the MPO representation of the cost function
        """

        return self.return_mpo_representation_symbolic()
        # NOTE: As noted above, the code here below is kept for back-compatibility.
        # returned_mpo = self._mpo_builder.build_mpo(self._n_qubits)

        # # Possibly add the long-range terms.
        # for i_long_range in self._long_range_terms:
        #     list_of_operators = [np.zeros([1, 2, 2], dtype=complex)]
        #     for _ in range(self._n_qubits - 2):
        #         list_of_operators.append(np.zeros([1, 1, 2, 2], dtype=complex))
        #     list_of_operators.append(np.zeros([1, 2, 2], dtype=complex))
        #     list_of_operators[0][0, :, :] = pauli("I")
        #     list_of_operators[-1][0, :, :] = pauli("I")
        #     for i_qubit in range(1, self._n_qubits - 1):
        #         list_of_operators[i_qubit][0, 0, :, :] = pauli("I")
        #     for i_qubit, i_operator in zip(i_long_range[0], i_long_range[1]):
        #         if i_qubit in (0, self._n_qubits - 1):
        #             list_of_operators[i_qubit][0, :, :] = pauli(i_operator)
        #         else:
        #             list_of_operators[i_qubit][0, 0, :, :] = pauli(i_operator)
        #     long_range_mpo = MatrixProductOperator(list_of_operators)
        #     long_range_mpo.multiply(i_long_range[2], inplace=True)
        #     returned_mpo.add_MPO(long_range_mpo, inplace=True)

        # # Final truncation
        # if self._truncation is not None:
        #     returned_mpo.compress(cutoff=self._truncation)

        # if self._max_bond is not None:
        #     returned_mpo.compress(max_bond=self._max_bond)

        # return MPOWrapper(returned_mpo)

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits on which the operator is defined"""
        return self._n_qubits

    @property
    def sparse_pauli(self) -> SparsePauliOp:
        """Getter for the `SparsePauliOp` object associated with the cost function"""
        return self._sparse_pauli_op

    def _convert_string_to_label(self, label: str) -> List[Tuple[int, str]]:
        """Converts a Pauli string to a sparse representation.

        The sparse representation is obtained by isolating the terms that are
        different from the identity, and associating them with the corresponding
        qubit index.

        Args:
            label (str): label of an element of a `SparsePauliOp` object.

        Returns:
            List[Tuple[int, str]]: list of (pauli operators, integer)
        """
        return [(self.n_qubits - idx - 1, chr) for idx, chr in enumerate(label) if chr != "I"]
