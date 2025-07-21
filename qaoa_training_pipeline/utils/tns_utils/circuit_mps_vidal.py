#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These classes wrap Tensor Network circuit functionalities."""

from typing import Optional, List, Tuple

from math import sqrt, cos, sin

import numpy as np

from quimb.tensor import (
    bonds,
    CircuitMPS,
    MatrixProductState,
    Tensor,
    TensorNetwork,
    tensor_network_gate_inds,
    tensor_split,
)

from .multi_qubit_gates import QAOAManyBodyCorrelator


class CircuitMPSVidalCanonization:
    """MPS representation of a circuit in the inverse Vidal representation.

    Specifically, this class uses the so-called "inverse Vidal canonization gauge" to
    represent the circuit (see PRB, 101, 235123 (2020) for additional details).
    This means, that the MPS is in the following format:

         M - D - M - D - M - D - M
         |       |       |       |

    where D is a diagonal matrix containing the inverse of the Schmidt values for
    each bond.
    This construction is well-suited for time-evolution/circuit simulation because
    it does not require selecting the canonization center. In fact, every qubit is
    a canonization center, which means that the M tensors fulfill the following
    normalization condition:

        + - M - +
        |   |   |   =  1
        + - M - +

    The left-canonical gauge can be obtained from the Vidal one by contracting each
    diagonal D tensor with the M corresponding tensor on the right, while the right-canonical
    gauge can be obtained by contracting the diagonal D tensor with the corresponding
    M tensor on the left.
    """

    # The important gates are saved as static class members.
    _h_gate = np.array([[1.0 / sqrt(2), 1.0 / sqrt(2)], [1.0 / sqrt(2), -1.0 / sqrt(2)]])

    # The swap two-qubit gate is expressed as a 2x2x2x2 matrix because
    # quimb requires tensors to have an independent dimension for each
    # index of the tensor.
    _swap_gate = np.array(
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
        ]
    )

    def __init__(
        self,
        input_mps: CircuitMPS,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
    ):
        """Class constructor.

        Args:
            input_mps (CircuitMPS): quimb CircuitMPS object containing the circuit representation
                of the MPS.
            truncation_threshold (Optional[float]): truncation parameter for the SVD
            max_bond_dim (Optional[int]): maximum bond dimension of the MPS
        """
        # Storage of input data
        self._truncation_threshold = truncation_threshold
        self._max_bond_dim = max_bond_dim
        self._compression_dict = {
            "cutoff": truncation_threshold,
            "max_bond": max_bond_dim,
            # Leaving this parameter as arguments in case we need to fine-tune the
            # numerics used in the SVD truncation.
            # "method": "svd",
            # "cutoff_mode": "rsum2",
        }
        input_circuit = input_mps.psi

        # Generates the MPS representation in the Vidal form
        self._n_qubits = input_circuit.L
        input_circuit.right_canonicalize(inplace=True)
        self._bond_indices = list(input_circuit.inner_inds())
        self._phys_indices = input_circuit.outer_inds()
        list_of_schmidt_values = []
        list_of_lambdas = list(input_circuit)

        # Loops over the sites and calculates the SVD
        for i_qubit in range(self._n_qubits - 1):
            local_tensor = list_of_lambdas[i_qubit] @ list_of_lambdas[i_qubit + 1]
            list_of_splitted_indices = (
                [self._phys_indices[i_qubit]]
                if i_qubit == 0
                else [self._phys_indices[i_qubit], self._bond_indices[i_qubit - 1]]
            )
            (u_mat, s_mat, v_mat) = local_tensor.split(
                list_of_splitted_indices, absorb=None, get="tensors"
            )
            new_index = s_mat.inds[0]
            diagonal_tensor = Tensor(np.diag(s_mat.data), inds=[new_index, "BD_tmp"])
            u_mat = u_mat @ diagonal_tensor
            v_mat = v_mat @ diagonal_tensor
            v_mat.reindex_({"BD_tmp": self._get_bond_label_right(i_qubit + 1)})
            u_mat.reindex_({"BD_tmp": self._get_bond_label_left(i_qubit)})
            for i_element in range(diagonal_tensor.data.shape[0]):
                diagonal_tensor.data[i_element, i_element] = (
                    1.0 / diagonal_tensor.data[i_element, i_element]
                )
            diagonal_tensor.reindex_(
                {
                    new_index: self._get_bond_label_left(i_qubit),
                    "BD_tmp": self._get_bond_label_right(i_qubit + 1),
                }
            )
            list_of_schmidt_values.append(diagonal_tensor)
            list_of_lambdas[i_qubit] = u_mat
            list_of_lambdas[i_qubit + 1] = v_mat
            # Final update of the indices
            self._bond_indices[i_qubit] = self._get_bond_label_right(i_qubit + 1)

        # Retagging
        for idx, i_lambda in enumerate(list_of_lambdas):
            i_lambda.drop_tags()
            i_lambda.add_tag(self._get_tensor_tag(idx))
        for idx, i_schmidt in enumerate(list_of_schmidt_values):
            i_schmidt.drop_tags()
            i_schmidt.add_tag(self._get_schmidt_tag(idx))
        # Final construction of the TN
        self._tn = TensorNetwork(
            [list_of_lambdas[0]]
            + [item for t in zip(list_of_schmidt_values, list_of_lambdas[1:]) for item in t]
        )

    @classmethod
    def construct_empty_circuit(
        cls,
        n_qubits: int,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
    ):
        """Empty class constructor.

        This constructor generates an empty circuit, without any gates.

        Args:
            n_qubits (int): number of qubits of the circuit
            truncation_threshold (Optional[float]): truncation parameter for the SVD
            max_bond_dim (Optional[int]): maximum bond dimension of the MPS
        """
        empty_circuit = CircuitMPS(n_qubits)
        return cls(empty_circuit, truncation_threshold, max_bond_dim)

    @staticmethod
    def _x_gate() -> np.ndarray:
        """Returns the matrix representation of the X gate

        Returns:
            np.ndarray: X gate as numpy tensor
        """
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    @staticmethod
    def _y_gate() -> np.ndarray:
        """Returns the matrix representation of the Y gate

        Returns:
            np.ndarray: Y gate as numpy tensor
        """
        return np.array([[0.0, -1j], [1j, 0.0]])

    @staticmethod
    def _z_gate() -> np.ndarray:
        """Returns the matrix representation of the Z gate

        Returns:
            np.ndarray: Z gate as numpy tensor
        """
        return np.array([[1.0, 0.0], [0.0, -1.0]])

    @staticmethod
    def _rx_gate(theta: float) -> np.array:
        """Returns the matrix representation of the RX gate

        Args:
            theta (float): rotation angle.

        Returns:
            np.array: RX gate as a numpy array.
        """
        return np.array(
            [[cos(theta / 2.0), -1j * sin(theta / 2.0)], [-1j * sin(theta / 2.0), cos(theta / 2.0)]]
        )

    @staticmethod
    def _ry_gate(theta: float) -> np.array:
        """Returns the matrix representation of the RY gate

        Args:
            theta (float): rotation angle.

        Returns:
            np.array: RY gate as a numpy array.
        """
        return np.array(
            [[cos(theta / 2.0), -sin(theta / 2.0)], [sin(theta / 2.0), cos(theta / 2.0)]]
        )

    @staticmethod
    def _rz_gate(theta: float) -> np.array:
        """Returns the matrix representation of the RZ gate

        Args:
            theta (float): rotation angle.

        Returns:
            np.array: RZ gate as a numpy array.
        """
        return np.array(
            [
                [cos(theta / 2.0) - 1.0j * sin(theta / 2.0), 0],
                [0, cos(theta / 2.0) + 1.0j * sin(theta / 2.0)],
            ],
            dtype=complex,
        )

    @staticmethod
    def _rzz_gate(theta: float) -> np.array:
        """Returns the matrix representation of the RZZ gate

        Args:
            theta (float): rotation angle.

        Returns:
            np.array: RZZ gate as a numpy array.
        """
        return np.array(
            [
                [
                    [[cos(theta / 2.0) - 1j * sin(theta / 2.0), 0.0], [0.0, 0.0]],
                    [[0.0, cos(theta / 2.0) + 1j * sin(theta / 2.0)], [0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0], [cos(theta / 2) + 1j * sin(theta / 2.0), 0.0]],
                    [[0.0, 0.0], [0.0, cos(theta / 2.0) - 1j * sin(theta / 2.0)]],
                ],
            ],
            dtype=complex,
        )

    def _get_tensor_tag(self, i_qubit: int) -> str:
        """Gets the tag associated with one of the core tensors

        Args:
            i_qubit (int): qubit for which the tag is returned

        Raises:
            ValueError: if `i_qubit` is < 0 or larger than the
                lattice size.

        Returns:
            str: tag associated with the tensor
        """
        if i_qubit > self._n_qubits - 1 or i_qubit < 0:
            raise ValueError("Qubit index not valid")
        return "Lambda_" + str(i_qubit)

    def _get_schmidt_tag(self, i_qubit: int) -> str:
        """Gets the tag associated with one of the Schmidt tensors

        Args:
            i_qubit (int): qubit for which the tag is returned.
                Note that the tag is returned for the Schmidt tensor
                located between `i_qubit` and `i_qubit + 1`

        Raises:
            ValueError: if `i_qubit` is < 0 or larger than
                the number of qubits - 1.

        Returns:
            str: tag associated with the tensor
        """
        if i_qubit > self._n_qubits - 2 or i_qubit < 0:
            raise ValueError("Qubit index not valid")
        return "Schmidt_" + str(i_qubit)

    def _get_bond_label_left(self, i_qubit: int) -> str:
        """Gets the label for the BD connecting a tensor with the Schmidt values from the left.

        Given the MPS in the Vidal form, the function returns the label associated with
        the index marked with an asterisk, where i_qubit is the site marked with a +

          +
          0 --- o --- 0 --- o --- 0
          |  *        |           |

        Args:
            i_qubit (int): index of the qubit for which the label is returned

        Raises:
            ValueError: if i_qubit > n_qubit - 2 or i_qubit < 0

        Returns:
            str: label associated with the specific label
        """
        if i_qubit > self._n_qubits - 2 or i_qubit < 0:
            raise ValueError("Qubit index not valid")
        return "BD_" + str(i_qubit)

    def _get_bond_label_right(self, i_qubit: int) -> str:
        """Gets the label for the BD connecting a tensor with the Schmidt values from the right.

        Given the MPS in the Vidal form, the function returns the label associated with
        the index marked with an asterisk, where i_qubit is the site marked with a +

                      +
          0 --- o --- 0 --- o --- 0
          |        *  |           |

        Args:
            i_qubit (int): index of the qubit for which the label is returned

        Raises:
            ValueError: if i_qubit > n_qubit - 1 or i_qubit < 1

        Returns:
            str: label associated with the specific label
        """
        if i_qubit > self._n_qubits - 1 or i_qubit < 1:
            raise ValueError("Qubit index not valid")
        return "BD_" + str(i_qubit - 1) + "_2"

    @property
    def n_qubits(self) -> int:
        """Getter for the number of qubits"""
        return self._n_qubits

    @property
    def tensor_network(self) -> TensorNetwork:
        """Getter for the underlying tensor network"""
        return self._tn

    def check_if_all_tensors_are_normalized(self, threshold: float = 1.0e-15) -> bool:
        """Checks if all the tensors composing the tensor network are normalized.

        In the inverse Vidal gauge, the three-index tensors composing the network
        (i.e., not the ones storing the Schmidt values) must be normalized according
        to the Hilbert-Schmidt norm. This function enables to check whether this is
        the case

        Args:
            threshold (float): tolerance for the normalization check.

        Returns:
            bool: True if all tensors are normalized, false otherwise.
        """
        return all(
            abs(self._tn[self._get_tensor_tag(i)].norm() - 1.0) < threshold
            for i in range(self._n_qubits)
        )

    def apply_x_gate(self, i_qubit: int):
        """Apply an X gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._x_gate(), i_qubit)

    def apply_y_gate(self, i_qubit: int):
        """Apply a Y gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._y_gate(), i_qubit)

    def apply_z_gate(self, i_qubit: int):
        """Apply a Z gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._z_gate(), i_qubit)

    def apply_rx_gate(self, i_qubit: int, theta: float):
        """Apply an Rx rotation gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
            theta (float): rotation angle.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._rx_gate(theta), i_qubit)

    def apply_ry_gate(self, i_qubit: int, theta: float):
        """Apply an Ry rotation gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
            theta (float): rotation angle.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._ry_gate(theta), i_qubit)

    def apply_rz_gate(self, i_qubit: int, theta: float):
        """Apply an Rz rotation gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
            theta (float): rotation angle.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._rz_gate(theta), i_qubit)

    def apply_h_gate(self, i_qubit: int):
        """Apply an H gate onto the circuit.

        Args:
            i_qubit (int): index of the qubit on which the gate is applied.
        """
        self._apply_single_qubit_gate(CircuitMPSVidalCanonization._h_gate, i_qubit)

    def _apply_single_qubit_gate(self, gate_matrix_representation: float, i_qubit: int):
        """Apply a single-qubit gate onto the circuit.

        Args:
            gate_matrix_representation (float): matrix representation of
                the gate to be applied.
            i_qubit (int): index of the qubit on which the gate is applied.
        """
        if i_qubit < 0:
            raise ValueError("Qubit index cannot be a negative number")
        if i_qubit >= self._n_qubits:
            raise ValueError("Qubit index too large")
        tensor_network_gate_inds(
            self._tn,
            G=gate_matrix_representation,
            inds=[self._phys_indices[i_qubit]],
            contract=True,
            inplace=True,
        )

    def apply_gate(self, name: str, params: List[float], i_qubit: int):
        """Applies a gate by name and params.

        This method is designed to be similar to `quimb.CircuitMPS.apply_gate`.

        Args:
            name: the name of the gate. Will be converted to lower case characters.
            params: the parameters of the gate to apply.
            i_qubit: the qubit on which to apply the gate.
        """
        name = name.lower()
        if name in ["rx", "ry", "rz"]:
            args = [i_qubit, params[0]]
        elif name in ["h", "x", "y", "z"]:
            args = [i_qubit]
        else:
            raise ValueError(f"{self.__class__.__name__} does not support the {name} gate.")

        getattr(self, f"apply_{name}_gate")(*args)

    def apply_swap_gate(self, i_qubit: int) -> np.ndarray:
        """Applies a swap gate between nearest-neighbouring sites

        The swap gate is applied between qubits `i_qubit` and `i_qubit + 1`

        Args:
            i_qubit (int): index of the qubit on which the Swap gate is applied

        Returns:
            np.ndarray: array containing the overall set of retained Schmidt
            values.
        """
        return self._apply_two_qubit_gate(i_qubit, CircuitMPSVidalCanonization._swap_gate)

    def apply_rzz_gate(self, i_qubit: int, j_qubit: int, theta: float) -> List[np.ndarray]:
        """Applies a RZZ gate between non-neighbouring qubits.

        Args:
            i_qubit (int): first qubit of the RZZ gate
            j_qubit (int): second qubit of the RZZ gate
            theta (float): rotation angle

        Returns:
             List[np.ndarray]: tuple storing the Schmidt values that
                are retained at each application of a two-qubit gate.
                If `i_qubit` and `j_qubit` are not nearest-neighbour,
                the Schmidt values are returned also for the SWAP gates
                that are implicitly applied to bring the qubits close.
        """
        min_qubit = min(i_qubit, j_qubit)
        max_qubit = max(i_qubit, j_qubit)
        list_of_schmidt_values = []

        for i_swap in range(min_qubit, max_qubit - 1):
            list_of_schmidt_values.append(self.apply_swap_gate(i_swap))

        list_of_schmidt_values.append(self.apply_rzz_gate_nn(max_qubit - 1, theta))

        for i_swap in range(max_qubit - 2, min_qubit - 1, -1):
            list_of_schmidt_values.append(self.apply_swap_gate(i_swap))

        return list_of_schmidt_values

    def apply_rzz_gate_nn(self, i_qubit: int, theta: float) -> np.ndarray:
        """Apply a nearest-neighbour two-qubit Rzz gate onto the circuit.

        Note that the two-qubit gate is applied to the pair of qubits
        (i_qubit, i_qubit+1). If one wants to apply a ZZ gate between
        not neighbouring qubits, an appropriate series of SWAP gates has
        to be applied.

        Args:
            i_qubit (int): index of the first qubit on which the gate is applied.
            theta (float): rotation angle.

        Returns:
            np.ndarray: array containing the overall set of retained Schmidt
            values.
        """
        return self._apply_two_qubit_gate(i_qubit, CircuitMPSVidalCanonization._rzz_gate(theta))

    def get_s_diagonal_elements_values(self, i_site: int) -> np.ndarray:
        """Gets the Schmidt values for a given bond

        Args:
            i_site (int): site for which the Schmidt values are
                calculated (specifically, they are calculated for
                the bond joining sites `i_site` and `i_site + 1`)

        Returns:
            np.ndarray: Schmidt values, stored in a 1D array.

        Raises:
            ValueError: if the index is out-of-bound.
        """

        if i_site < 0 or i_site >= self._n_qubits - 1:
            raise ValueError("Index out of bound for the Schmidt values")

        return np.diag(self._tn[self._get_schmidt_tag(i_site)].data)

    def apply_hyperedge(self, hyper_edge: QAOAManyBodyCorrelator, scaling_factor: float) -> None:
        r"""Applies a HOBO-like term to the MPS, expressed in Vidal's form.

        The hyperedge can be readily converted into an MPO, which can then be
        applied onto the MPS based on the following contraction strategy.

                O -- X -- O -- X -- O -- X -- O
                |         |         |         |
                @ ------- @ ------- @ ------- @
                |         |         |         |

        The contraction of the tensor network is done following these steps:

         - first, the following contraction is calculated:

                     -- O -- X --     -- + -- + --
                        |                |    |
                        @ -------  =     |    | --
                        |                + -- +
                                         |

         - we recover the Vidal form by applying the SVD as we would do
           for a two-qubit gate, with the key difference that now we have
           two indices on the left, rather than one.

         - we repeat the procedure for all sites.

        The procedure is essentially equivalent to the zip-up algorithm proposed
        in New. J. Phys., 12, 005026 (2010). As discussed in the paper, one has to
        be a bit more careful when applying the conventional truncation scheme to
        the zip-up algorithm. In fact, the entries of the diagonal tensor (represented
        as --X--) do not represent the Schmidt values of the wave function. Therefore,
        setting the truncation threshold to a specific value :math:`\eta` does not
        guarantee that the approximation error is bound by :math:`\eta`. Nevertheless,
        it has been heuristically observed that the approximation error monotonically
        decreases with :math:`\eta` -- therefore one can just repeat simulations for
        decreasing values of :math:`\eta` and monitor the convergence of the target
        observables.

        Args:
            hyper_edge (QAOAManyBodyCorrelator): hyperedge associated with the
                Hamiltonian term that should be applied.
            scaling_factor (float): scaling factor appearing in the exponential.
        """

        # Gets the MPO representation of the hyperedge
        mpo_hyper = hyper_edge.get_mpo_representation(scaling_factor=scaling_factor)
        mpo_inner_inds = mpo_hyper.inner_inds()

        self._tn = self._tn & mpo_hyper

        for i_site in range(self._n_qubits - 1):
            # For the first site, we first contract the Lambda tensor with the
            # MPO -- this is done at the end of the loop for the subsequent sites.
            if i_site == 0:
                self._tn.contract_ind("k0")

            self._tn.contract_ind(self._get_bond_label_left(i_site))
            (tid,) = self._tn._get_tids_from_tags(self._get_schmidt_tag(i_site), which="all")
            tensor = self._tn.pop_tensor(tid)
            left_inds = (
                ["b0"] if i_site == 0 else ["b" + str(i_site), self._get_bond_label_right(i_site)]
            )
            # pylint: disable=invalid-name
            U, S, V = tensor_split(
                tensor,
                left_inds=left_inds,
                get="tensors",
                absorb=None,
                **self._compression_dict,
            )
            new_index = S.inds[0]

            # Inverts the overlap matrix
            diagonal_elements = S.data
            inverse_diagonal_elements = np.array(
                [1 / i if abs(i) > 1.0e-15 else i for i in diagonal_elements]
            )
            U.multiply_index_diagonal(new_index, diagonal_elements, inplace=True)
            V.multiply_index_diagonal(new_index, diagonal_elements, inplace=True)
            U.drop_tags(self._get_schmidt_tag(i_site))
            V.drop_tags()
            V.add_tag("V")
            self._tn.add_tensor(U)
            self._tn.add_tensor(V)
            self._tn.insert_operator(
                np.diag(inverse_diagonal_elements.data),
                self._get_tensor_tag(i_site),
                "V",
                tags=self._get_schmidt_tag(i_site),
                inplace=True,
            )

            # Contracts the V part into the next site
            self._tn.contract_ind(self._get_bond_label_right(i_site + 1))
            self._tn.contract_ind(mpo_inner_inds[i_site])

            # Final retagging
            bnd_index_l = bonds(
                self._tn[self._get_tensor_tag(i_site)], self._tn[self._get_schmidt_tag(i_site)]
            ).popright()
            self._tn.reindex_({bnd_index_l: self._get_bond_label_left(i_site)})
            bnd_index_r = bonds(self._tn[self._get_schmidt_tag(i_site)], self._tn["V"]).popright()
            self._tn.reindex_({bnd_index_r: self._get_bond_label_right(i_site + 1)})
            self._tn.drop_tags("V")

        # Final retagging of bra indices to ket indices
        self._tn.reindex_({"b" + str(i): "k" + str(i) for i in range(self._n_qubits)})

    def get_mps_representation(self, canonization_center: int = 0) -> MatrixProductState:
        """Transforms the MPS in the inverse Vidal canonical form to a normal MPS.

        The transformation is performed by absorbing, for the sites on the left of the
        canonization center, the diagonal matrix with the singular values into the
        preceding tensor. Conversely, for the tensors at the right of the canonization
        center, we merge them with the diagonal matrix lying on the left.

        Args:
            canonization_center (int, optional): canonization center of the final MPS.
                Defaults to 0.

        Returns:
            MatrixProductState: conventional MPS representation of the tensor network
                in the Vidal form
        """
        matrix_product_state = self._tn.copy(deep=True, virtual=False)
        for i_site in range(self._n_qubits):
            if i_site < canonization_center:
                matrix_product_state.contract_ind(self._get_bond_label_left(i_site))
            elif i_site > canonization_center:
                matrix_product_state.contract_ind(self._get_bond_label_right(i_site))

        # Final transposition
        for idx in range(self._n_qubits):
            ten = matrix_product_state[self._get_tensor_tag(idx)]
            if idx < canonization_center:
                if idx == 0:
                    ten.transpose(self._get_bond_label_right(idx + 1), "k" + str(idx), inplace=True)
                else:
                    ten.transpose(
                        self._get_bond_label_right(idx),
                        self._get_bond_label_right(idx + 1),
                        "k" + str(idx),
                        inplace=True,
                    )
            elif idx > canonization_center:
                if idx == self._n_qubits - 1:
                    ten.transpose(self._get_bond_label_left(idx - 1), "k" + str(idx), inplace=True)
                else:
                    ten.transpose(
                        self._get_bond_label_left(idx - 1),
                        self._get_bond_label_left(idx),
                        "k" + str(idx),
                        inplace=True,
                    )
            else:
                if idx == 0:
                    ten.transpose(self._get_bond_label_left(idx), "k" + str(idx), inplace=True)
                elif idx == self._n_qubits - 1:
                    ten.transpose(self._get_bond_label_right(idx), "k" + str(idx), inplace=True)
                else:
                    ten.transpose(
                        self._get_bond_label_right(idx),
                        self._get_bond_label_left(idx),
                        "k" + str(idx),
                        inplace=True,
                    )

        return MatrixProductState(
            [matrix_product_state[self._get_tensor_tag(i)].data for i in range(self._n_qubits)]
        )

    def _apply_two_qubit_gate(self, i_qubit: int, array: np.array) -> np.ndarray:
        """Applies a two-qubit gate on the circuit.

        Note that the two-qubit gate is assumed to be nearest-neighbour,
        so it is applied on gates `i_qubit` and `i_qubit + 1`

        Args:
            i_qubit (int): qubit on which the gate is applied
            array (np.array): 4x4 array representation of the gate

        Returns:
            np.ndarray: array containing the overall set of retained Schmidt
            values.

        Raises:
            ValueError: if the qubit index is unphysical
        """
        # Standard checks
        if i_qubit < 0:
            raise ValueError("Qubit index cannot be a negative number")
        if i_qubit >= self._n_qubits - 1:
            raise ValueError("Qubit index too large")

        # Merges the tensors on which the gate is applied, including the lambda term.
        self._tn.contract_ind(self._get_bond_label_left(i_qubit))
        self._tn[self._get_tensor_tag(i_qubit)].drop_tags(self._get_schmidt_tag(i_qubit))
        info = {}
        tensor_network_gate_inds(
            self._tn,
            G=array,
            inds=[self._phys_indices[i_qubit], self._phys_indices[i_qubit + 1]],
            contract="reduce-split",
            inplace=True,
            info=info,
            absorb=None,
            **self._compression_dict,
        )

        # We renormalize the S matrix to ensure that the state is normalized even
        # if we neglect non-zero singular values
        diagonal_elements = info[("singular_values", self._get_bond_label_right(i_qubit + 1))]
        sqrt_normalization = sqrt(sum(i**2 for i in diagonal_elements))
        diagonal_elements_normalized = np.array(diagonal_elements) / sqrt_normalization
        inverse_diagonal_elements = np.array(
            [1 / i if abs(i) > 1.0e-15 else i for i in diagonal_elements_normalized]
        )
        self._tn[self._get_tensor_tag(i_qubit)].multiply_index_diagonal(
            self._get_bond_label_right(i_qubit + 1), diagonal_elements_normalized, inplace=True
        )
        self._tn[self._get_tensor_tag(i_qubit + 1)].multiply_index_diagonal(
            self._get_bond_label_right(i_qubit + 1), diagonal_elements_normalized, inplace=True
        )
        self._tn.insert_operator(
            np.diag(inverse_diagonal_elements.data),
            self._get_tensor_tag(i_qubit),
            self._get_tensor_tag(i_qubit + 1),
            tags=self._get_schmidt_tag(i_qubit),
            inplace=True,
        )
        bnd = bonds(
            self._tn[self._get_tensor_tag(i_qubit + 1)], self._tn[self._get_schmidt_tag(i_qubit)]
        )
        if len(bnd) != 1:
            raise ValueError("Expected tensors joined by single index, multiple index found")
        bnd_index = bnd.popright()
        self._tn.reindex_(
            {self._get_bond_label_right(i_qubit + 1): self._get_bond_label_left(i_qubit)}
        )
        self._tn.reindex_({bnd_index: self._get_bond_label_right(i_qubit + 1)})
        return diagonal_elements
