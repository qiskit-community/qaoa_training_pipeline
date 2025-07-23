#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These classes wrap Tensor Network circuit functionalities."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, List, Tuple

from math import sqrt

from networkx import Graph
import numpy as np

from quimb.tensor import CircuitMPS, MatrixProductState, tensor_network_gate_inds
from quimb.tensor.circuit import parse_to_gate

from qiskit import QuantumCircuit
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qaoa_training_pipeline.utils.circuit_utils import split_circuit
from .qaoa_cost_function import QAOACostFunction
from .circuit_mps_vidal import CircuitMPSVidalCanonization
from .multi_qubit_gates import QAOAManyBodyCorrelator


class QAOACircuitTNSRepresentation(ABC):
    """Abstract class representing the TNS representation of a QAOA circuit.

    This class is abstract because it represents an interface to the specific
    TNS method used to simulate the QAOA circuit. It implements all the functionalities
    that are independent of the specificities of the TNS, such as, e.g., the
    construction of the cost function.

    Limitations:
      * Mixer is hardcoded to be a sum of X gates.
      * The circuit ansatz is constructed starting from the adjecency matrix
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        n_qubits: int,
        adjacency_matrix: np.array,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
        swap_strategy: Optional[SwapStrategy] = None,
        list_of_hyperedges: Optional[List[Tuple[List[int], float]]] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        store_intermediate_schmidt_values: bool = False,
    ):
        """Class initialization.

        Args:
            n_qubits (int): number of qubits of the circuit.
            adjacency_matrix (np.array): adjacency matrix in the QAOA problem.
            truncation_threshold (Optional[float], optional): threshold of the singular value
                decomposition-based compression of the circuit. Defaults to None.
            max_bond_dim (Optional[int], optional): maximum value for the bond dimension of
                the matrix product state representation of the circuit. Defaults to None.
            swap_strategy (SwapStrategy): The swap strategy to use. If None (the default)
                then we do not apply the gates in the cost layer following a swap strategy.
            list_of_hyperedges (Optional[List[Tuple[List[int], float]]], optional):
                list of hyperedges, represented as a list of tuples, where the first
                element of the tuple is a list containing the indices of the hyperedge,
                and the second element is the corresponding weight. Defaults to None.
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
            store_intermediate_schmidt_values (bool): whether the Schmidt values associated with
                each application of a two-qubit gate should be stored. Defaults to `False`.
        """
        self._n_qubits = n_qubits
        self._adj_matrix = adjacency_matrix
        self._threshold = truncation_threshold
        self._max_bond = max_bond_dim
        self._swap_strat = swap_strategy
        self._swap_layer_pairs = None
        self._initial_states = None if initial_state is None else split_circuit(initial_state)
        self._store_schmidt = store_intermediate_schmidt_values

        if self._store_schmidt:
            self._list_of_schmidt = []

        self._mixers = None
        if mixer is not None:
            if len(mixer.parameters) > 1:
                raise ValueError("Currently, only mixer operators with one parameter are allowed.")
            self._mixers = split_circuit(mixer)

        # Order the edges according to the distance in the swap strategy.
        if swap_strategy is not None:
            self._swap_layer_pairs = defaultdict(list)
            for i_qubit in range(self._n_qubits):
                for j_qubit in range(0, i_qubit):
                    if abs(self._adj_matrix[i_qubit, j_qubit]) > 1.0e-16:
                        distance = self._swap_strat.distance_matrix[i_qubit, j_qubit]
                        self._swap_layer_pairs[distance].append(
                            (min(i_qubit, j_qubit), max(i_qubit, j_qubit))
                        )

        # The high-order many-body correlator are converted to instances of the
        # `QAOAManyBodyCorrelator` class here.
        self._list_of_hyperedges: List[QAOAManyBodyCorrelator] = []
        if list_of_hyperedges is not None:
            if len(list_of_hyperedges) != 0:

                if self._store_schmidt:
                    raise ValueError("Schmidt value generation for HOBO not supported")

                for i in list_of_hyperedges:
                    self._list_of_hyperedges.append(
                        QAOAManyBodyCorrelator(i[0], self._n_qubits, i[1])
                    )

    @classmethod
    # pylint: disable=too-many-positional-arguments
    def construct_from_graph(
        cls,
        graph: Graph,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
        swap_strategy: Optional[SwapStrategy] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        store_intermediate_schmidt_values: bool = False,
    ):
        """Constructor from a networkx graph.

        Args:
            graph (Graph): input graph, expressed as a networkx `Graph` object. This graph
                expresses the network of gates that the MPS will apply.
            truncation_threshold (Optional[float], optional):
                Threshold for the SVD truncation in the Matrix Product State construction.
                Defaults to None, in which case no SVD thresholding is applied.
            max_bond_dim (Optional[int], optional):
                Maximum allowed value for the Matrix Product State bond dimension.
                Defaults to None, in which case the bond dimension growth is governed
                by the choice for `truncation_threshold`.
            swap_strategy (SwapStrategy): The swap strategy to use. If None (the default)
                then we do not apply the gates in the cost layer following a swap strategy.
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
            store_intermediate_schmidt_values (bool): whether the Schmidt values associated with
                each application of a two-qubit gate should be stored. Defaults to `False`.

        Returns:
            QAOACircuitMPSRepresentation: instance of the QAOACircuitMPSRepresentation
                class associated with the input `Graph`
        """

        n_qubits = max(i[0] for i in graph.adjacency()) + 1
        adjacency_matrix = np.zeros((n_qubits, n_qubits), dtype=float)
        for i_node in graph.adjacency():
            for i_edge in i_node[1].keys():
                adjacency_matrix[i_node[0], i_edge] = i_node[1][i_edge]["weight"]

        return cls(
            n_qubits,
            adjacency_matrix,
            truncation_threshold,
            max_bond_dim,
            swap_strategy=swap_strategy,
            mixer=mixer,
            initial_state=initial_state,
            store_intermediate_schmidt_values=store_intermediate_schmidt_values,
        )

    @classmethod
    # pylint: disable=too-many-positional-arguments
    def construct_from_list_of_edges(
        cls,
        list_of_edges: List[Tuple[List[int], float]],
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
        swap_strategy: Optional[SwapStrategy] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        store_intermediate_schmidt_values: bool = False,
    ):
        """Constructor taking as input explicitly a list of edges/weight.

        Note that this is the only constructor that can support high-order binary
        optimization (HOBO) problems.

        Args:
            list_of_edges (List[Tuple[List[int], float]]): _description_
            truncation_threshold (Optional[float], optional): _description_. Defaults to None.
            max_bond_dim (Optional[int], optional): _description_. Defaults to None.
            swap_strategy (Optional[SwapStrategy]): A SWAP strategy with which to apply
                swap gates. This can result in a more efficient implementation of the cost
                operator, especially for dense problems. Defaults to None.
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
        """
        n_qubits = max(max(i[0]) for i in list_of_edges) + 1
        adjacency_matrix = np.zeros((n_qubits, n_qubits), dtype=float)
        list_of_higher_order_terms = []

        # Loop over the edges and classify them
        for i_edge in list_of_edges:
            # First-order terms are in the diagonal of the adjacency matrix
            if len(i_edge[0]) == 1:
                adjacency_matrix[i_edge[0][0], i_edge[0][0]] = i_edge[1]
            # Second-order edges
            elif len(i_edge[0]) == 2:
                adjacency_matrix[i_edge[0][0], i_edge[0][1]] = i_edge[1]
                adjacency_matrix[i_edge[0][1], i_edge[0][0]] = i_edge[1]
            else:
                list_of_higher_order_terms.append(i_edge)

        return cls(
            n_qubits,
            adjacency_matrix,
            truncation_threshold,
            max_bond_dim,
            swap_strategy=swap_strategy,
            list_of_hyperedges=list_of_higher_order_terms,
            mixer=mixer,
            initial_state=initial_state,
            store_intermediate_schmidt_values=store_intermediate_schmidt_values,
        )

    @property
    def n_qubits(self) -> int:
        """Getter for the number of qubits"""
        return self._n_qubits

    @abstractmethod
    def get_underlying_tn(self):
        """Getter for the underlying tensor network"""
        pass

    @abstractmethod
    def get_schmidt_values(self, i_site: int):
        """Gets the Schmidt values for a given site"""
        pass

    @abstractmethod
    def compute_expectation_value_single_pauli_string(self, pauli_string: str) -> float:
        """Calculates the expectation value of a single Pauli string."""
        pass

    def get_intermediate_schmidt_values(self) -> List[List[float]]:
        """Getter for the intermediate Schmidt values."""
        return self._list_of_schmidt

    def compute_cost_function(self, cost_function: QAOACostFunction) -> complex:
        """Calculates the QAOA cost associated with the circuit

        Args:
            cost_function (QAOACostFunction): cost function for the QAOA problem

        Returns:
            complex: < self | cost_function | self >

        Raises:
            KeyError: if the cost function and the mps have different number of qubits
        """
        if cost_function.n_qubits != self.n_qubits:
            raise KeyError(
                "Number of qubits of cost function does not match the number of qubits of the circuit"
            )
        mpo_cost_function = cost_function.mpo.mpo
        psi = self.get_underlying_tn()
        psi_dagger = psi.H
        psi_dagger.reindex_(dict(zip(mpo_cost_function.lower_inds, mpo_cost_function.upper_inds)))
        tensor_network_to_contract = psi_dagger & mpo_cost_function & psi
        return tensor_network_to_contract.contract(..., optimize="auto-hq") / psi.norm() ** 2

    def apply_split_circuits(self, circuits: List[QuantumCircuit], scaling_factor: float = None):
        """Applies initial states and mixer layers of single-qubit gates.

        Args:
            circuits: Each circuit is a single-qubit. This comes as an output of the method
                `split_circuit`.
            scaling_factor: If the circuits have a parameter then we bind with this value. Here,
                the scaling factor is typically the free parameter in the mixer operator of
                a custom mixer.
        """
        for i_qubit, circuit in enumerate(circuits):
            if scaling_factor is not None:
                circuit = circuit.assign_parameters([scaling_factor], inplace=False)

            for inst in circuit.data:
                self._mps_representation.apply_gate(inst.name, inst.params, i_qubit)

    def apply_qaoa_layer(self, betas: List[float], gammas: List[float]):
        """Applies a full layer of the QAOA ansatz.

        Note that this method is not abstract because it just calls the underlying
        private methods constructing the individual components of the circuit.
        These methods are the ones that must be implemented by the derived class.

        Args:
            betas (List[float]): proportionality term for the mixing factor(s).
            gammas (List[float]): proportionality term for the Hamiltonian evolution(s).
        """

        self._apply_initial_layer()

        rep = 1  # determines if even or odd layer.

        for gamma_value, beta_value in zip(gammas, betas):
            if self._swap_strat is None:
                self._apply_ansatz_layer(gamma_value)
            else:
                self._apply_layer_ansatz_swap_strat(gamma_value, rep)

            self._apply_mixing_layer(beta_value)

            rep += 1

    @abstractmethod
    def _apply_initial_layer(self):
        """Applies the initial layer to all qubits"""
        pass

    @abstractmethod
    def _apply_mixing_layer(self, scaling_factor: float) -> None:
        """Applies the mixing layer onto the circuit.

        Note: This function applies the standard QAOA mixer. Once this is extended
        to accept one-local mixers unique to each qubit we will need to handle how
        the custom mixer interacts with the swap strategies.

        Args:
            scaling_factor (float): scaling term for the mixing.
        """
        pass

    @abstractmethod
    def _apply_ansatz_layer(self, scaling_factor: float):
        """Applies the circuit ansatz onto the circuit.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.

        Returns:
            List[np.ndarray]: list containing the Schmidt values that are retained after
                each two-qubit application. Note that this includes also the "virtual"
                SWAP gates that are implicitly applied for non-nearest two-qubit interactions.
        """
        pass

    @abstractmethod
    def _apply_layer_ansatz_swap_strat(self, scaling_factor: float, rep: int):
        """Applies the circuit ansatz onto the circuit following a SWAP strategy.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.
            rep (int): Indicates if the QAOA layer is even or odd. Depth-one,
                for example, is odd.
        """
        pass


class QAOACircuitMPSRepresentation(QAOACircuitTNSRepresentation):
    """Class representing the Matrix Product State representation of a QAOA circuit.

    Specifically, this class uses the so-called "left-right canonization gauge" to
    represent the circuit. This means, that the MPS is in the following format:

         L - L - L - O - R - R - R
         |   |   |   |   |   |   |

    where tensors represented by an "L" are left normalized, i.e., they represent
    an isometry when the physical (vertical) index is grouped with the row index of
    the tensor. Similarly, the "R" tensors are right normalized, i.e., they represent
    an isometry when the physical index is grouped with the column index of the tensor.
    The "O" tensor is, instead, generic, and is the so-called "canonization center".
    A gauge freedom in the definition of an MPS makes it possible to select any qubit
    as canonization center -- i.e., the canonization center can be shifted through
    successive SVD.

    The choice of this gauge makes it possible to control the accuracy of the TNS
    simulation when two-qubit gates are applied to the circuit.

    Consider the case of a two-qubit gate acting on qubits (i, i+1). When this gate
    is applied onto the MPS representation of the circuit, the MPS structure is broken
    (see doc of `_apply_ansatz_layer`) because the two tensors associated with sites
    `i` and `i+1` are contracted together. The original MPS structure can be restored
    by calculating the SVD decomposition of the tensor resulting from the application of
    the two-qubit gate. However, if no truncation is applied to the SVD, the dimension of
    the MPS will grow exponentially with the circuit depth.
    To avoid this growth, the SVD can be replaced with a truncated SVD by keeping only the
    singular values (and the corresponding singular vectors) that are below a given threshold.
    If the MPS is expressed in the "left-right canonical gauge" and if the canonization
    center is either located at qubit i or at qubit (i+1), then this threshold is proportional
    to the accuracy (measured as fidelity) at which the circuit is approximated.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        n_qubits: int,
        adjacency_matrix: np.array,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
        swap_strategy: Optional[SwapStrategy] = None,
        list_of_hyperedges: Optional[List[Tuple[List[int], float]]] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        store_intermediate_schmidt_values: bool = False,
    ):
        """Class constructor

        Args:
            n_qubits (int): number of qubits of the circuit.
            adjacency_matrix (np.array): adjacency matrix in the QAOA problem.
            truncation_threshold (Optional[float], optional): truncation threshold
                for the singular value decomposition in the matrix product state simulator.
                Defaults to None.
            max_bond_dim (Optional[int], optional): maximum bond dimension of the
                matrix product state representation of the circuit. Defaults to None.
            swap_strategy (SwapStrategy): A strategy through which to apply the
                gates in the cost function layer. This argument is None by default
                in which case we perform SWAPs for each edge applied.
            list_of_hyperedges (Optional[List[Tuple[List[int], float]]], optional):
                list of hyperedges, represented as a list of tuples, where the first
                element of the tuple is a list containing the indices of the hyperedge,
                and the second element is the corresponding weight. Defaults to None.
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
            store_intermediate_schmidt_values (bool): whether the Schmidt values associated with
                each application of a two-qubit gate should be stored. Defaults to `False`.
        """
        self._mps_representation = CircuitMPS(n_qubits)
        self._canonization_center = 0
        super().__init__(
            n_qubits,
            adjacency_matrix,
            truncation_threshold,
            max_bond_dim,
            swap_strategy=swap_strategy,
            list_of_hyperedges=list_of_hyperedges,
            mixer=mixer,
            initial_state=initial_state,
            store_intermediate_schmidt_values=store_intermediate_schmidt_values,
        )

    def get_underlying_tn(self) -> MatrixProductState:
        """Getter for the MPS representation"""
        return self._mps_representation.psi

    def get_schmidt_values(self, i_site: int) -> List[float]:
        """Gets the Schmidt values for a given site.

        Args:
            i_site (int): site for which the Schmidt values are calculated.
                Note that the Schmidt values are returned between site
                `i_site` and `i_site + 1`

        Returns:
            List[float]: list containing the Schmidt values for the
                selected bond.
        """
        return self._mps_representation.psi.schmidt_values(i_site + 1)

    def compute_expectation_value_single_pauli_string(self, pauli_string: str) -> float:
        """Calculates the expectation value of a single Pauli string.

        For the conventional canonization, the expectation value can be calculated
        by applying each gate to the corresponding MPS tensor.

        Args:
            pauli_string (str): string representing the Pauli string to be measured.

        Returns:
            float: < Psi | pauli_string | Psi >
        """

        bra_mps = self._mps_representation.copy()
        for i_qubit, i_gate in enumerate(pauli_string):
            if i_gate != "I":
                bra_mps.apply_gate(i_gate, i_qubit)
        return (bra_mps.psi.H & self._mps_representation.psi) ^ all

    def _apply_initial_layer(self) -> None:
        """Applies the initial layer to all qubits"""
        if self._initial_states is not None:
            self.apply_split_circuits(self._initial_states)
        else:
            for i_qubit in range(self.n_qubits):
                self._mps_representation.apply_gate("H", i_qubit)

    def _apply_mixing_layer(self, scaling_factor: float) -> None:
        """Applies the mixing layer onto the circuit.

        Note that the mixing layer is composed of single-qubit gates.
        Since these gates do not alter the structure of the MPS, no
        SVD truncation is needed for bringing the wave function back to
        the MPS format after the gates are applied.

        Args:
            scaling_factor (float): scaling term for the mixing.
        """
        if self._mixers is not None:
            self.apply_split_circuits(self._mixers, scaling_factor)
        else:
            # Single-qubit gate, no truncation. However, we keep track of the canonization
            for i_qubit in range(self._n_qubits):
                self._mps_representation.apply_gate("RX", 2.0 * scaling_factor, i_qubit)

        self._canonization_center = self.n_qubits - 1

    def _apply_one_local(self, scaling_factor: float):
        """Internal helper function to apply one-local terms from the Ansatz."""
        for i_qubit in range(self.n_qubits):
            if self._adj_matrix[i_qubit, i_qubit] != 0.0:
                value = 2.0 * scaling_factor * self._adj_matrix[i_qubit, i_qubit]
                self._mps_representation.apply_gate("RZ", value, i_qubit)

    def _apply_ansatz_layer(self, scaling_factor: float):
        """Applies the circuit ansatz onto the circuit.

        Since the circuit ansatz is, by default, constructed from the Trotter approximation
        of the Ising Hamiltonian underlying the QAOA problem, simulating it requires applying
        two-qubit gates onto the MPS representation of the circuit. This means, in practice,
        that one needs to contract a tensor network of the following type (assuming that the
        two qubits on which the gate acts are neighbours in the one-dimensional sorting that is
        implicitly defined by the MPS):

          |    |
          xxxxxx                      <-- gate
          |    |    |    |    |    |
          0 -- 0 -- 0 -- 0 -- 0 -- 0  <-- MPS

        Contracting the gate with the MPS yields the following, new tensor network

          |    |    |    |    |    |
          yyyyyy -- 0 -- 0 -- 0 -- 0

        that do not correspond anymore to an MPS. To bring back the yyyyyy tensor to an MPS-like
        form, an SVD is applied, possibly truncated based on the `truncation_threshold` and the
        `max_bond_dim` class members.

        If the two-qubit gate is applied on non-neigbouring qubits, a chain of SWAP gates is applied
        onto the MPS to bring the two qubits close one to each other (and back, after the gate has
        been applied). Also for the SWAP gates, we apply the SVD-based procedure outlined above
        to ensure that the circuit can be consistently represented as an MPS.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.
        """

        list_of_coupled_pairs = []
        for i_qubit in range(self.n_qubits):
            for j_qubit in range(0, i_qubit):
                if abs(self._adj_matrix[i_qubit, j_qubit]) > 1.0e-16:
                    list_of_coupled_pairs.append((min(i_qubit, j_qubit), max(i_qubit, j_qubit)))
        list_of_coupled_pairs.sort(key=lambda x: x[0])

        self._apply_one_local(scaling_factor)

        # Now loops over the pairs to apply the Z_i Z_j terms and applies them.
        swap_gate = parse_to_gate("SWAP", 0, 1).build_array()

        for i_pairs in list_of_coupled_pairs:

            # Determines the gate
            j_qubit = min(i_pairs[0], i_pairs[1])
            i_qubit = max(i_pairs[0], i_pairs[1])
            rzz_gate = parse_to_gate(
                "RZZ", 2.0 * scaling_factor * self._adj_matrix[i_qubit, j_qubit], j_qubit, i_qubit
            )

            # Canonizes wrt the *first* site. Note that the SWAP gates should not
            # influence the canonization (the sites are just swapped back and forth)

            for swap_site in range(j_qubit, i_qubit - 1):
                list_of_schmidt = self._apply_two_qubit_gate(swap_site, swap_gate, False)
                if self._store_schmidt:
                    self._list_of_schmidt.append(list_of_schmidt)

            # Applies the original gate
            list_of_schmidt_gate = self._apply_two_qubit_gate(i_qubit - 1, rzz_gate.array, True)
            if self._store_schmidt:
                self._list_of_schmidt.append(list_of_schmidt_gate)

            # Swaps back
            for swap_site in range(i_qubit - 2, j_qubit - 1, -1):
                list_of_schmidt = self._apply_two_qubit_gate(swap_site, swap_gate, True)
                if self._store_schmidt:
                    self._list_of_schmidt.append(list_of_schmidt)

        # Proceeds to the application of the higher-order terms
        for i_higher_order in self._list_of_hyperedges:
            # Gets the MPO representation of the high-order correlator
            mpo_representation = i_higher_order.get_dense_representation(scaling_factor)
            self._mps_representation.apply_gate_raw(mpo_representation, i_higher_order.i_qubits)

    def _apply_two_qubit_gate(self, i_qubit: int, array: np.array, absorb_left: bool) -> np.ndarray:
        """Applies a two-qubit gate on the circuit.

        Note that the two-qubit gate is assumed to be nearest-neighbour,
        so it is applied on qubits `i_qubit` and `i_qubit + 1`

        Args:
            i_qubit (int): qubit on which the gate is applied
            array (np.array): 4x4 array representation of the gate
            absort_left (bool): whether the singular values should be
                absorbed in the left tensor, with index `i_qubit`.
                If `False`, the singular values are absorbed in the tensor
                with index `i_qubit+1`.

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

        # Proceeds to the gate application
        input_mps = self._mps_representation.psi.copy()
        input_mps.shift_orthogonality_center(current=self._canonization_center, new=i_qubit)
        phys_indices = self._mps_representation.psi.outer_inds()
        aux_indices = self._mps_representation.psi.inner_inds()
        tensor_indices = ["I" + str(i) for i in range(self._n_qubits)]

        info = {}
        tensor_network_gate_inds(
            input_mps,
            G=array,
            inds=[phys_indices[i_qubit], phys_indices[i_qubit + 1]],
            contract="reduce-split",
            inplace=True,
            info=info,
            absorb=None,
            cutoff=self._threshold,
            max_bond=self._max_bond,
        )

        # We renormalize the S matrix to ensure that the state is normalized even
        # if we neglect non-zero singular values
        diagonal_elements = info[("singular_values", aux_indices[i_qubit])]
        sqrt_normalization = sqrt(sum(i**2 for i in diagonal_elements))
        diagonal_elements_normalized = np.array(diagonal_elements) / sqrt_normalization
        if absorb_left:
            input_mps[tensor_indices[i_qubit]].multiply_index_diagonal(
                aux_indices[i_qubit], diagonal_elements_normalized, inplace=True
            )
            self._canonization_center = i_qubit
        else:
            input_mps[tensor_indices[i_qubit + 1]].multiply_index_diagonal(
                aux_indices[i_qubit], diagonal_elements_normalized, inplace=True
            )
            self._canonization_center = i_qubit + 1

        # Copies back the MPS
        self._mps_representation = CircuitMPS(self._n_qubits, psi0=input_mps)
        return diagonal_elements

    def _apply_layer_ansatz_swap_strat(self, scaling_factor: float, rep: int):
        """Apply the cost operator using a SWAP strategy.

        Note: The swap strategy is not undone. This is taken into account in the next
        QAOA layer and in the cost operator.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.
            rep (int): the repetition of the layer. This determines if we apply the swap
                strategy in its original order (rep is odd) or in the inverse order (rep
                is even).
        """
        layer_order = list(range(len(self._swap_strat) + 1))
        if rep % 2 == 0:
            layer_order = layer_order[::-1]

        self._apply_one_local(scaling_factor)

        # There are len(layer_order) layers of Rzz gates and len(layer_order) - 1 layers
        # of SWAP gates.
        swap_gate = parse_to_gate("SWAP").build_array()
        for layer_idx in layer_order:

            permutation = self._swap_strat.inverse_composed_permutation(layer_idx)

            # 1. Apply the gates.
            for i_pairs in self._swap_layer_pairs[layer_idx]:
                j_qubit, i_qubit = i_pairs[0], i_pairs[1]
                tn_i = permutation.index(i_qubit)
                tn_j = permutation.index(j_qubit)

                tn_i_qubit = min(tn_i, tn_j)

                rzz_gate = parse_to_gate(
                    "RZZ",
                    2.0 * scaling_factor * self._adj_matrix[i_qubit, j_qubit],
                    j_qubit,
                    i_qubit,
                )
                list_of_schmidt_gate = self._apply_two_qubit_gate(tn_i_qubit, rzz_gate.array, True)
                if self._store_schmidt:
                    self._list_of_schmidt.append(list_of_schmidt_gate)

            if rep % 2 == 0:
                swap_layer_idx = layer_idx - 1
            else:
                swap_layer_idx = layer_idx

            # 2. Apply the SWAPs.
            if 0 <= swap_layer_idx < len(self._swap_strat):
                for swap_pairs in self._swap_strat.swap_layer(swap_layer_idx):

                    if swap_pairs[1] != swap_pairs[0] + 1:
                        raise ValueError("Inconsistency found in SWAP strategy")

                    list_of_schmidt_swap = self._apply_two_qubit_gate(
                        swap_pairs[0], swap_gate, True
                    )
                    if self._store_schmidt:
                        self._list_of_schmidt.append(list_of_schmidt_swap)


class QAOACircuitVidalRepresentation(QAOACircuitTNSRepresentation):
    """Matrix Product State representation of a QAOA circuit.

    This class uses the so-called "inverse Vidal canonization gauge" to
    represent the circuit (see PRB, 101, 235123 (2020) for additional details).
    This means, that the MPS is in the following format:

         M - D - M - D - M - D - M
         |       |       |       |

    where M are tensors with Frobenius norm = 1, and D are tensors that collect
    the Schmidt values that one would obtain by applying the Schmidt decomposition
    to the corresponding bond.

    The advantage of this gauge compared to the "left-right" one is that one does
    not need to shift the canonization center every time a two-qubit gate is applied.
    For this reason, multiple qubits can be applied in parallel.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        n_qubits: int,
        adjacency_matrix: np.array,
        truncation_threshold: Optional[float] = None,
        max_bond_dim: Optional[int] = None,
        swap_strategy: Optional[SwapStrategy] = None,
        list_of_hyperedges: Optional[List[Tuple[List[int], float]]] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        store_intermediate_schmidt_values: bool = False,
    ):
        """Class initialization.

        See the constructor of the `QAOACircuitTNSRepresentation` for additional
        details on the class members.

        Args:
            n_qubits (int): number of qubits of the circuit.
            adjacency_matrix (np.array): adjacency matrix in the QAOA problem.
            truncation_threshold (Optional[float], optional): truncation threshold
                for the singular value decomposition in the matrix product state simulator.
                Defaults to None.
            max_bond_dim (Optional[int], optional): maximum bond dimension of the
                matrix product state representation of the circuit. Defaults to None.
            swap_strategy (SwapStrategy): A strategy through which to apply the
                gates in the cost function layer. This argument is None by default
                in which case we perform SWAPs for each edge applied.
            list_of_hyperedges (Optional[List[Tuple[List[int], float]]], optional):
                list of hyperedges, represented as a list of tuples, where the first
                element of the tuple is a list containing the indices of the hyperedge,
                and the second element is the corresponding weight. Defaults to None.
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
            store_intermediate_schmidt_values (bool): whether the Schmidt values associated with
                each application of a two-qubit gate should be stored. Defaults to `False`.
        """

        self._mps_representation = CircuitMPSVidalCanonization.construct_empty_circuit(
            n_qubits, truncation_threshold, max_bond_dim
        )
        super().__init__(
            n_qubits,
            adjacency_matrix,
            truncation_threshold,
            max_bond_dim,
            swap_strategy=swap_strategy,
            list_of_hyperedges=list_of_hyperedges,
            mixer=mixer,
            initial_state=initial_state,
            store_intermediate_schmidt_values=store_intermediate_schmidt_values,
        )

    def get_underlying_tn(self) -> MatrixProductState:
        """Getter for the MPS representation"""
        return self._mps_representation.get_mps_representation()

    def get_schmidt_values(self, i_site: int) -> List[float]:
        """Gets the Schmidt values for a given site.

        Args:
            i_site (int): site for which the Schmidt values are calculated.
                Note that the Schmidt values are returned between site
                `i_site` and `i_site + 1`

        Returns:
            List[float]: list containing the Schmidt values for the
                selected bond.
        """

        # Here we invert because the Vidal gauge stores the *inverse* of the
        # Schmidt values in the matrix joining two sites.
        return [
            1.0 / i**2 if abs(i) > 1.0e-10 else 0.0
            for i in self._mps_representation.get_s_diagonal_elements_values(i_site)
        ]

    def _apply_initial_layer(self) -> None:
        """Applies the initial layer to all qubits"""
        if self._initial_states is not None:
            self.apply_split_circuits(self._initial_states)
        else:
            for i_qubit in range(self.n_qubits):
                self._mps_representation.apply_h_gate(i_qubit)

    def _apply_mixing_layer(self, scaling_factor: float) -> None:
        """Applies the mixing layer onto the circuit.

        For the moment, the mixing layer is hardcoded being a
        layer of Rx gates.

        Args:
            scaling_factor (float): scaling term for the mixing.
        """
        if self._mixers is not None:
            self.apply_split_circuits(self._mixers, scaling_factor)
        else:
            # Single-qubit gate, no truncation. However, we keep track of the
            # canonization
            for i_qubit in range(self._n_qubits):
                self._mps_representation.apply_rx_gate(i_qubit, 2.0 * scaling_factor)

    def _apply_one_local(self, scaling_factor):
        """Internal helper function to apply one-local terms from the Ansatz."""
        for i_qubit in range(self.n_qubits):
            if abs(self._adj_matrix[i_qubit, i_qubit]) > 1.0e-16:
                value = 2.0 * scaling_factor * self._adj_matrix[i_qubit, i_qubit]
                self._mps_representation.apply_rz_gate(i_qubit, value)

    def _apply_ansatz_layer(self, scaling_factor: float) -> List[np.ndarray]:
        """Applies the circuit ansatz onto the circuit.

        Since the circuit ansatz is, by default, constructed from the Trotter approximation
        of the Ising Hamiltonian underlying the QAOA, simulating it requires applying
        two-qubit gates onto the MPS representation of the circuit. This means, in practice,
        that one contracts a tensor network of the following type (assuming that the
        two qubits on which the gate acts are neighbours in the one-dimensional sorting
        implicitly defined by the MPS):

         M - D - M - D - M - D - M - D - M
         |       |       |       |       |
                 xxxxxxxxx                    <-- gate
                 |       |

        Contracting the gate with the MPS yields the following, new tensor network

         M - D - xxxxxxxxx - D - M - D - M
         |       |       |       |       |

        that does not correspond anymore to an MPS in the Vidal form. One can apply an
        form, an SVD is applied, possibly truncated based on the `truncation_threshold` and the
        `max_bond_dim` class members, to approximate the two-qubit tensor as follows:

          - xxxxxxxx -  ~  - U - D - V -
            |      |         |       |

        However, if this equality is plugged in the original tensor network, the resulting
        MPS is not in the Vidal canonical form. In fact, the canonical form requires that,
        for a given qubit, contracting the corresponding M tensor with the D matrix on the left
        yields a right-normalized tensor and, conversely, that contracting the M tensor with
        the D matrix located on the right yields a left-normalized tensor. Since U is, by
        definition of SVD, left-normalized, and V is right-normalized, their contraction with
        D does not have the desired normalization property. We overcome this limitation by
        expressing the SVD as:

          - xxxxxxxx -  ~  - U - D - D^{-1} - D - V -
            |      |         |                    |
                           |      |         |       |
                           +------+         +-------+
                              M                 N

        Now, the M and N have the desired properties. Therefore, we use them to update the
        MPS, and use D^{-1} to update the matrix with the Schmidt values.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.
        """

        list_of_coupled_pairs = []
        for i_qubit in range(self.n_qubits):
            for j_qubit in range(0, i_qubit):
                if abs(self._adj_matrix[i_qubit, j_qubit]) > 1.0e-16:
                    list_of_coupled_pairs.append((min(i_qubit, j_qubit), max(i_qubit, j_qubit)))
        list_of_coupled_pairs.sort(key=lambda x: x[0])

        self._apply_one_local(scaling_factor)

        # Now loops over the pairs
        for i_pairs in list_of_coupled_pairs:
            j_qubit = i_pairs[0]
            i_qubit = i_pairs[1]
            list_of_schmidt = self._mps_representation.apply_rzz_gate(
                j_qubit, i_qubit, 2.0 * scaling_factor * self._adj_matrix[i_qubit, j_qubit]
            )
            if self._store_schmidt:
                self._list_of_schmidt.append(list_of_schmidt)

        # HOBO term
        for i_hyper_edge in self._list_of_hyperedges:
            self._mps_representation.apply_hyperedge(i_hyper_edge, scaling_factor)

    def _apply_layer_ansatz_swap_strat(self, scaling_factor: float, rep: int):
        """Applies the circuit ansatz onto the circuit following a SWAP strategy.

        Args:
            scaling_factor (float): scaling term appearing in the exponent.
        """

        layer_order = list(range(len(self._swap_strat) + 1))
        if rep % 2 == 0:
            layer_order = layer_order[::-1]

        self._apply_one_local(scaling_factor)

        # There are len(layer_order) layers of Rzz gates and len(layer_order) - 1 layers
        # of SWAP gates.
        for layer_idx in layer_order:
            permutation = self._swap_strat.inverse_composed_permutation(layer_idx)
            # 1. Apply the gates.
            for node0, node1 in self._swap_layer_pairs[layer_idx]:
                tn_j_qubit = min(permutation.index(node0), permutation.index(node1))

                list_of_schmidt = self._mps_representation.apply_rzz_gate_nn(
                    tn_j_qubit,
                    2.0 * scaling_factor * self._adj_matrix[node0, node1],
                )
                if self._store_schmidt:
                    self._list_of_schmidt.append(list_of_schmidt)

            if rep % 2 == 0:
                swap_layer_idx = layer_idx - 1
            else:
                swap_layer_idx = layer_idx

            # 2. Apply the SWAPs.
            if 0 <= swap_layer_idx < len(self._swap_strat):
                for swap_pairs in self._swap_strat.swap_layer(swap_layer_idx):
                    list_of_schmidt_swap = self._mps_representation.apply_swap_gate(swap_pairs[0])
                    if self._store_schmidt:
                        self._list_of_schmidt.append(list_of_schmidt_swap)

    def compute_expectation_value_single_pauli_string(self, pauli_string: str) -> float:
        """Calculates the expectation value of a single Pauli string.

        Args:
            pauli_string (str): Pauli string to be measured

        Returns:
            float: expectation value of the Pauli operator over the TN.
        """

        mps_representation = CircuitMPS(psi0=self.get_underlying_tn().copy())
        for i_qubit, i_gate in enumerate(pauli_string):
            if i_gate != "I":
                mps_representation.apply_gate(i_gate, i_qubit)
        return (mps_representation.psi.H & self.get_underlying_tn()) ^ all
