# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These classes manage the compilation of multi-qubit gates."""

from itertools import product
from cmath import exp
from math import log10
from typing import Optional, Tuple, List
import numpy as np

from quimb.tensor import (
    MatrixProductOperator,
    Tensor,
    tensor_split,
    tensor_network_gate_inds,
)


class QAOAManyBodyCorrelator:
    r"""This class represents a term of the following form:

    .. math::
        \exp^{-\mathrm{i} t Z_i Z_j \cdots Z_k}

    and offers functionalities to represent it as a matrix product operator,
    such that it can then be applied to a tensor network (and, specifically,
    to a matrix product state).

    The matrix product operator is constructed as follows:

      1. assuming that the operator acts on :math:`n` qubits, we first construct
         the dense representation of the operator as a :math:`2^n \times 2^n`
         matrix. Graphically, such a matrix can be represented as:

            |   |   |
         +-------------+
         |             |
         +-------------+
            |   |   |

         in the case in which :math:`n=3`.

      2. through successive SVD, we factorize the tensor as a matrix product operator

            |   |   |         |       |       |
         +-------------+    +---+   +---+   +---+
         |             | =  |   | - |   | - |   |
         +-------------+    +---+   +---+   +---+
            |   |   |         |       |       |

      3. the resulting matrix product operator represents the original gate if the
         indices :math:`i,j,\ldots` are consecutive. If this is not the case, we
         apply SWAP gates to ensure that the MPO indices match the gate indices.
    """

    def __init__(self, i_qubits: Tuple[int], n_qubits: int, time_step: float):
        """Class constructor

        Args:
            i_qubits (Tuple[int]): qubits on which the gate acts.
            n_qubits (int): number of qubits of the circuit.
            time_step (float): scaling factor appearing at the exponent.

        Raises:
            ValueError: if one requests to have operators with > 4 qubits,
                if some of the qubits are repeated, or if any of the indices
                is larger than the number of qubits.
        """

        if len(i_qubits) > 4:
            raise ValueError("Multi-qubit operations supported only up to 4 qubits")
        if len(set(i_qubits)) != len(i_qubits):
            raise ValueError("Some of the qubit indices are repeated!")
        if any(i >= n_qubits for i in i_qubits):
            raise ValueError("Qubit index exceeds the number of qubits")

        # As the class is limited to product of Z gates, the order in which
        # the qubits appear is not relevant. For this reason, we can sort
        # the qubits, which makes the MPO construction easier.
        # Note that this sorting does not introduce any SWAP overhead.
        self._i_qubits = sorted(i_qubits)
        self._n_qubits = n_qubits
        self._time_step = time_step
        self._order = len(self._i_qubits)
        self._mpo_representation = None
        self._current_bond_dim = None
        self._current_trunc_thresh = None

    @property
    def n_qubits(self) -> int:
        """Getter for the number of qubits"""
        return self._n_qubits

    @property
    def i_qubits(self) -> List[int]:
        """Getter for the qubits on which the gate acts"""
        return self._i_qubits

    def _is_initial(self) -> bool:
        """Helper function checking if the gate is applied on the first qubit.

        Returns:
            bool: true if the gate is applied on the first qubit,
                false otherwise
        """
        return self._i_qubits[0] == 0

    def _is_final(self) -> bool:
        r"""Helper function checking if the gate is applied on the last qubit.

        Note that, when constructing the MPO, it is first assumed that
        the gate acts on neighbouring qubits. This means that, e.g., if the
        multi-qubit gate acts on qubits (0, 2, 3), the dense representation of
        the gate is first constructed for qubits (0, 1, 2). The corresponding
        MPO is constructed through repeated singular value decompositions of
        this dense operator, and identity operators are added for all sites
        :math:`i \geq 3`. Lastly, SWAP gates are applied to map the sites
        (0, 1, 2) to the qubits on which the gates acts -- i.e., (0, 2, 3)
        in this example.

        This function returns true if the last qubit is the final one
        in the *neighbouring* sorting. So, if the circuit has 4 qubits,
        a gate acting on qubits (0, 1, 3) will return False to this function
        call, because the dense representation will be constructed for qubits
        (0, 1, 2). Conversely, a gate acting on qubits (1, 2, 3) will return
        True.

        Returns:
            bool: true if the gate is applied on the last qubit,
                false otherwise
        """
        return self._i_qubits[0] + len(self._i_qubits) == self._n_qubits

    def get_dense_representation(self, scaling_factor: float) -> np.array:
        """Get the dense representation of the propagator.

        Note that the dense representation is calculated only *within* the
        qubits on which the propagator acts -- i.e., if the propagator acts
        on qubits (2, 3, 4), qubits (0, 1) are ignored.

        Args:
            scaling_factor (float): scalar to be used to scale the
                exponent operator.

        Returns:
            np.array: Tensor representation of the propagator.
        """

        dimensions = [2] * len(self._i_qubits) * 2

        # The dense tensor that we construct must be contracted with
        # identities from the left (if the gate is not applied onto
        # the first qubit) and from the right (if the gate is not
        # applied onto the last qubit) to extend it on the qubits
        # on which the original gate does not acts. We need additional
        # "virtual" indices to represent the contraction with the
        # identity operators.
        if not self._is_initial():
            dimensions.insert(0, 1)
        if not self._is_final():
            dimensions.append(1)

        # Generates the raw data
        data = np.zeros(dimensions, dtype=complex)
        for combination_row in product([0, 1], repeat=self._order):
            exponent_term = scaling_factor
            for i in combination_row:
                if i == 1:
                    exponent_term *= -1.0
            index = combination_row + combination_row
            if not self._is_initial():
                index = (0,) + index
            if not self._is_final():
                index = index + (0,)
            data[index] = exp(-1j * self._time_step * exponent_term)

        return data

    def get_mpo_representation(
        self,
        scaling_factor: Optional[float] = 1.0,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
    ) -> MatrixProductOperator:
        """Get the matrix product operator (MPO) representation of the gate.

        The MPO representation is obtained by first constructing the MPO
        representation of the gate assuming that they are neighbouring,
        and, then, applying the proper network of SWAP gates to bring the
        gates in the correct position.

        Args:
            scaling_factor (Optional[float]): scalar to optionally
                scale the angle in the exponent of the multi-qubit
                operation. Defaults to 1.
            truncation_threshold (Optional[float]): threshold for the
                truncated Singular Value Decompositon.
                Defaults to None, in which case this is effectively 0.
            max_bond_dim (Optional[int]): maximum bond dimension that
                is allowed when compressing the MPO.
                Defaults to None, in which case no boundaries on the bond
                dimension are applied.

        Returns:
            MatrixProductOperator: MPO representation of the gate
        """

        requires_new_calculation = True

        # If an MPO representation of the gate is found, may consider not recalculating it.
        if self._mpo_representation:
            # Checks the truncation threshold -- if it is the same, then can skip and, therefore,
            # can set `truncation_threshold` to False
            if truncation_threshold is not None and self._current_trunc_thresh is not None:
                requires_new_calculation = (
                    log10(truncation_threshold / self._current_trunc_thresh) > 1
                )

            # If a calculation is not needed up to now, we may still need to do that if a
            # different bond dimension has been requested.
            if not requires_new_calculation:
                if max_bond_dim is not None and self._current_bond_dim is not None:
                    requires_new_calculation = self._current_bond_dim != max_bond_dim

        if requires_new_calculation:
            # Prepares the container for the MPO
            self._mpo_representation = MatrixProductOperator.new(
                L=self._n_qubits,
                cyclic=False,
                site_tag_id="site{}",
                upper_ind_id="b{}",
                lower_ind_id="k{}",
            )

            # Generates the quimb Tensor associated with the gate
            inds = ["b" + str(i + self._i_qubits[0]) for i in range(self._order)] + [
                "k" + str(i + self._i_qubits[0]) for i in range(self._order)
            ]
            if not self._is_initial():
                inds.insert(0, "BD" + str(self._i_qubits[0] - 1))
            if not self._is_final():
                inds.append("BD" + str(self._i_qubits[0] + len(self._i_qubits) - 1))
            gate_tensorized = Tensor(self.get_dense_representation(scaling_factor), inds=inds)

            # Constructs the Matrix Product Operator
            for i_qubit in range(self._i_qubits[0]):
                if i_qubit == 0:
                    one_tensor = np.zeros([2, 2, 1], dtype=float)
                    one_tensor[0, 0, 0] = 1.0
                    one_tensor[1, 1, 0] = 1.0
                    identity_tensor = Tensor(
                        one_tensor, inds=["b0", "k0", "BD0"], tags=["site" + str(i_qubit)]
                    )
                else:
                    one_tensor = np.zeros([2, 2, 1, 1], dtype=float)
                    one_tensor[0, 0, 0, 0] = 1.0
                    one_tensor[1, 1, 0, 0] = 1.0
                    identity_tensor = Tensor(
                        one_tensor,
                        inds=[
                            "b" + str(i_qubit),
                            "k" + str(i_qubit),
                            "BD" + str(i_qubit - 1),
                            "BD" + str(i_qubit),
                        ],
                        tags=["site" + str(i_qubit)],
                    )
                self._mpo_representation |= identity_tensor

            # Actual "meat" of the MPO
            for i in range(self._i_qubits[0], self._i_qubits[0] + len(self._i_qubits) - 1):
                left_inds = (
                    ["b0", "k0"] if i == 0 else ["b" + str(i), "k" + str(i), "BD" + str(i - 1)]
                )
                (l, r) = tensor_split(
                    gate_tensorized,
                    left_inds=left_inds,
                    get="tensors",
                    absorb="both",
                    bond_ind="BD" + str(i),
                )
                l.add_tag("site" + str(i))
                self._mpo_representation |= l
                if i != self._i_qubits[0] + len(self._i_qubits) - 2:
                    gate_tensorized = r
                else:
                    r.add_tag("site" + str(i + 1))
                    self._mpo_representation |= r

            for i_qubit in range(self._i_qubits[0] + len(self._i_qubits), self._n_qubits):
                if i_qubit == self._n_qubits - 1:
                    one_tensor = np.zeros([2, 2, 1], dtype=float)
                    one_tensor[0, 0, 0] = 1.0
                    one_tensor[1, 1, 0] = 1.0
                    identity_tensor = Tensor(
                        one_tensor,
                        inds=["b" + str(i_qubit), "k" + str(i_qubit), "BD" + str(i_qubit - 1)],
                        tags=["site" + str(i_qubit)],
                    )
                else:
                    one_tensor = np.zeros([2, 2, 1, 1], dtype=float)
                    one_tensor[0, 0, 0, 0] = 1.0
                    one_tensor[1, 1, 0, 0] = 1.0
                    identity_tensor = Tensor(
                        one_tensor,
                        inds=[
                            "b" + str(i_qubit),
                            "k" + str(i_qubit),
                            "BD" + str(i_qubit - 1),
                            "BD" + str(i_qubit),
                        ],
                        tags=["site" + str(i_qubit)],
                    )
                self._mpo_representation |= identity_tensor

            # Swap the third site
            for i_swap_index in range(len(self._i_qubits) - 1, 0, -1):
                for i in range(self._i_qubits[0] + i_swap_index, self._i_qubits[i_swap_index]):
                    self._swap_sites(
                        self._mpo_representation,
                        i,
                        True,
                        "right",
                        truncation_threshold,
                        max_bond_dim,
                    )
                    self._swap_sites(
                        self._mpo_representation,
                        i,
                        False,
                        "right",
                        truncation_threshold,
                        max_bond_dim,
                    )

            # Final update of parameters
            self._current_bond_dim = max_bond_dim
            self._current_trunc_thresh = truncation_threshold

        return self._mpo_representation

    def _swap_sites(
        self,
        input_mpo: MatrixProductOperator,
        site: int,
        is_upper: bool,
        absorb_mod: str,
        truncation_threshold: Optional[float] = None,
        max_bond_dimension: Optional[float] = None,
    ):
        """Applies a swap gate onto the MPS

        The swap is applied on the pair of sites (site, site+1)

        Args:
            input_mpo (MatrixProductOperator): Matrix Product Operator on which
                the swap gates are applied.
            site (int): sites onto which the swap is applied.
            is_upper (bool): whether to apply onto the upper of lower indices.
            absorb_mod (str): where to absorb the non-unitary part.
            truncation_threshold (Optional[float]): threshold for the
                truncated Singular Value Decompositon.
                Defaults to None, in which case this is effectively 0.
            max_bond_dim (Optional[int]): maximum bond dimension that
                is allowed when compressing the MPO.
                Defaults to None, in which case no boundaries on the bond
                dimension are applied.
        """
        swap_gate = np.zeros([2, 2, 2, 2], dtype=float)
        for i_phys in range(2):
            for j_phys in range(2):
                swap_gate[i_phys, j_phys, j_phys, i_phys] = 1.0
        if site > self._n_qubits - 2:
            raise KeyError("Qubit index for swap outsite range")

        # Note that for the swap operator the transpose/conjugate
        # business is irrelevant
        inds = (
            ("b" + str(site), "b" + str(site + 1))
            if is_upper
            else ("k" + str(site), "k" + str(site + 1))
        )

        # Final appliation of the gate
        tensor_network_gate_inds(
            input_mpo,
            G=swap_gate,
            inds=inds,
            contract="reduce-split",
            inplace=True,
            absorb=absorb_mod,
        )
