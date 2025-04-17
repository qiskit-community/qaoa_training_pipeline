#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Symbolic MPO utilities."""

import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp

from quimb.tensor import MatrixProductOperator, Tensor


class SymbolicMPOConstruction:
    r"""Class enabling constructing a matrix product operator (MPO) through its
    symbolic representation. "Symbolic" meas that the MPO is constructed based
    on the Pauli representation of the target operator, without converting the
    latter to its numeric representation.

    Remember that the MPO representation of an operator :math:`O` is obtained as:

    .. math::
        O = \sum_{\sigma,\sigma'} \sum_{b_1,\ldots,b_{N-1}}
            O_{1,b_1}^{\sigma_1,\sigma_1'}
            O_{b_1,b_2}^{\sigma_2,\sigma_2'} \cdots
            O_{b_{N-1},1}^{\sigma_N,\sigma_N'}

    This symbolic representation of the MPO is obtained by first constructing the
    matrices :math:`O_{b_{i-1},b_i}` and keeping their entries as symbolic operators
    (i.e., elementary Paulis) and, only at a later stage, replacing these Paulis
    with the corresponding matrix representation.

    The symbolic MPO representation enables constructing the MPO representation of
    arbitrarily long-range interacting Hamiltonians based on the scheme described
    in Phys. Rev. A 78, 012356 (2018). In fact, the sum given above can be expressed,
    by summing over the :math:`\sigma` and :math:`\sigma'` indices, in the following
    form:

    .. math::
        O = \sum_{b_1,\ldots,b_{N-1}}
              \hat{O}_{1,b_1} \hat{O}_{b_1,b_2} \cdots
              \hat{O}_{b_{N-1},1}

    i.e., the MPO is a product of operator-valued matrices that, contracted together,
    return the overall operator :math:`O`. The algorithm presented in Phys. Rev. A 78,
    012356 (2018) defines a recipe for, given an MPO, extend it to include a new term
    that can have support on arbitrarily many qubit. This is radically different from
    a naive approach where one term is added to the MPO by constructing first the
    corresponding MPO representation, and summing it with the original MPO. With the
    symbolic procedure, the only mathematical operation to be implemented is the update
    of the matrices :math:`\hat{O}_{b_i,b_{i+1}}`, while in the second case, a series
    of tensor addition/compression is necessary.
    """

    def __init__(self, n_sites: int, constant_factor: float = 0.0):
        """Class constructor

        Args:
            n_sites (int): number of sites of the MPO
            constant_factor (float, optional): proportionality constant for
                the identity term. Defaults to 0.
        """

        self._n_sites = n_sites
        self._mpo_maps = [{} for _ in range(self._n_sites)]
        self._constant_factor = constant_factor

        # Add a zero contribution to initialize the maps
        self._mpo_maps[0][("idl", "idr")] = SparsePauliOp(["I"], [0.0])
        self._mpo_maps[0][("idl", "idl")] = SparsePauliOp(["I"], [1.0])

        for i_site in range(1, self._n_sites - 1):
            self._mpo_maps[i_site][("idl", "idl")] = SparsePauliOp(["I"], [1.0])
            self._mpo_maps[i_site][("idl", "idr")] = SparsePauliOp(["I"], [0.0])
            self._mpo_maps[i_site][("idr", "idr")] = SparsePauliOp(["I"], [1.0])

        self._mpo_maps[self._n_sites - 1][("idl", "idr")] = SparsePauliOp(["I"], [0.0])
        self._mpo_maps[self._n_sites - 1][("idr", "idr")] = SparsePauliOp(["I"], [1.0])

    def add_term(self, pauli: Pauli, coeff: complex):
        """Add a Pauli term to the Matrix Product Operator.

        Args:
            pauli (Pauli): Pauli operator.
            coeff (complex): scalar scaling factor associated with the term.
        """

        list_of_chars = [str(i) for i in pauli[::-1]]
        pauli_weight = len([i for i in list_of_chars if i != "I"])

        # Zero-th order term. Note that a term must be already there because
        # we have added it in the class constructor.
        if pauli_weight == 0:
            self._mpo_maps[0][("idl", "idr")] += SparsePauliOp(["I"], [coeff])

        # One-body terms
        elif pauli_weight == 1:
            position = next((i for i, x in enumerate(list_of_chars) if x != "I"), -1)
            assert position != -1

            self._mpo_maps[position][("idl", "idr")] += SparsePauliOp(
                [list_of_chars[position]], [coeff]
            )

        # Two-body terms
        elif pauli_weight == 2:
            positions = [i for i, x in enumerate(list_of_chars) if x != "I"][:2]
            tag = "f_" + list_of_chars[positions[0]] + str(positions[0])
            self._mpo_maps[positions[0]][("idl", tag)] = SparsePauliOp(
                [list_of_chars[positions[0]]], [1.0]
            )
            for i_site in range(positions[0] + 1, positions[1]):
                self._mpo_maps[i_site][(tag, tag)] = SparsePauliOp(["I"], [1.0])
            self._mpo_maps[positions[1]][(tag, "idr")] = SparsePauliOp(
                [list_of_chars[positions[1]]], [coeff]
            )

        # HOBO-like terms
        else:
            positions = [i for i, x in enumerate(list_of_chars) if x != "I"]

            # Determines how many fork and merges we need to implement
            if pauli_weight % 2 == 0:
                n_forks = (pauli_weight - 1) // 2 + 1
                n_merge = (pauli_weight - 1) // 2
            else:
                n_forks = (pauli_weight - 1) // 2
                n_merge = (pauli_weight - 1) // 2

            # Prepares the fork/merge tags
            fork_tags = []
            merge_tags = []
            current_string = ""
            for i_fork in range(n_forks):
                current_string += "f_" + list_of_chars[positions[i_fork]] + str(positions[i_fork])
                fork_tags.append(current_string)

            current_string = ""
            for i_merge in range(n_merge):
                current_string = (
                    "m_"
                    + list_of_chars[positions[-i_merge - 1]]
                    + str(positions[-i_merge - 1])
                    + current_string
                )
                merge_tags.append(current_string)
            merge_tags.reverse()
            overall_tags = fork_tags + merge_tags

            # Add the term to the symbolic operator
            previous_tag = "idl"
            for i_term in range(pauli_weight - 1):
                current_tag = overall_tags[i_term]
                target_coeff = coeff if previous_tag[0] == "f" and current_tag[0] == "m" else 1.0
                self._mpo_maps[positions[i_term]][(previous_tag, current_tag)] = SparsePauliOp(
                    [list_of_chars[positions[i_term]]], [target_coeff]
                )
                for i_site in range(positions[i_term] + 1, positions[i_term + 1]):
                    self._mpo_maps[i_site][(current_tag, current_tag)] = SparsePauliOp(["I"], [1.0])
                previous_tag = current_tag

            # Final term
            self._mpo_maps[positions[pauli_weight - 1]][(overall_tags[-1], "idr")] = SparsePauliOp(
                [list_of_chars[positions[pauli_weight - 1]]], [1.0]
            )

    def generate_mpo_representation(self) -> MatrixProductOperator:
        """Returns the quimb `MatrixProductOperator` representation of the class.

        Remember that the `SymbolicMPOConstruction` stores the MPO representation
        of the underlying operator as a set of operator-value matrices, where the
        operators are qiskit `SparsePauliOp` objects. This format has to be converted
        to an actual MPO with scalar entries before doing any calculation -- this
        method achieves this.

        Note that no compression is applied at this stage.

        Returns:
            MatrixProductOperator: actual MPO associated with the symbolic
                representation stored in this class.
        """

        # Creates an empty MPO
        returned_mpo = MatrixProductOperator.new(
            L=self._n_sites,
            cyclic=False,
            site_tag_id="site{}",
            upper_ind_id="b{}",
            lower_ind_id="k{}",
        )

        # Retrieves the tags
        map_per_site = []
        max_bond_dim = []
        for i_site in range(self._n_sites):
            row_set = set(j[0] for j in self._mpo_maps[i_site].keys())
            map_per_site.append({x: i for i, x in enumerate(row_set)})
            max_bond_dim.append(len(row_set))
            if i_site == self._n_sites - 1:
                row_set = set(j[1] for j in self._mpo_maps[i_site].keys())
                max_bond_dim.append(len(row_set))
                map_per_site.append({x: i for i, x in enumerate(row_set)})

        # First tensor
        target_tensor = np.zeros([2, 2, max_bond_dim[1]], dtype=complex)

        for i_entry, i_op in self._mpo_maps[0].items():
            i_row = map_per_site[0][i_entry[0]]
            i_col = map_per_site[1][i_entry[1]]
            assert i_row == 0
            target_tensor[:, :, i_col] = i_op.to_matrix()

        tensor_to_be_added = Tensor(target_tensor, inds=["b0", "k0", "BD0"], tags=["site0"])
        returned_mpo |= tensor_to_be_added

        # Middle tensor
        for i_site in range(1, self._n_sites - 1):
            target_tensor = np.zeros(
                [2, 2, max_bond_dim[i_site], max_bond_dim[i_site + 1]], dtype=complex
            )

            for i_entry, i_op in self._mpo_maps[i_site].items():
                i_row = map_per_site[i_site][i_entry[0]]
                i_col = map_per_site[i_site + 1][i_entry[1]]
                target_tensor[:, :, i_row, i_col] = i_op.to_matrix()

            tensor_to_be_added = Tensor(
                target_tensor,
                inds=[
                    "b" + str(i_site),
                    "k" + str(i_site),
                    "BD" + str(i_site - 1),
                    "BD" + str(i_site),
                ],
                tags=["site" + str(i_site)],
            )
            returned_mpo |= tensor_to_be_added

        # Final tensor
        target_tensor = np.zeros([2, 2, max_bond_dim[self._n_sites - 1]], dtype=complex)

        for i_entry, i_op in self._mpo_maps[self._n_sites - 1].items():
            i_row = map_per_site[self._n_sites - 1][i_entry[0]]
            i_col = map_per_site[self._n_sites][i_entry[1]]
            assert i_col == 0
            target_tensor[:, :, i_row] = i_op.to_matrix()

        tensor_to_be_added = Tensor(
            target_tensor,
            inds=[
                "b" + str(self._n_sites - 1),
                "k" + str(self._n_sites - 1),
                "BD" + str(self._n_sites - 2),
            ],
            tags=["site" + str(self._n_sites - 1)],
        )
        returned_mpo |= tensor_to_be_added

        return returned_mpo
