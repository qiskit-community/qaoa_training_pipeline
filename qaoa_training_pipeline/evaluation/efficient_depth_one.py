#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to efficiently evaluate depth-one circuits."""

from dataclasses import dataclass, field
from itertools import pairwise

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy import sparse

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.circuit_utils import split_circuit
from qaoa_training_pipeline.utils.graph_utils import circuit_to_graph, operator_to_graph

# cspell: words correlator correlators kron einsum reduceat bincount indptr tril

# Eigenvalues of Z for the computational basis states |0> and |1>.
_Z = np.array([1.0, -1.0])

# The entry rho[(a_i, a_j), (b_i, b_j)] of a two-qubit density matrix is multiplied by the
# neighbour product P(a_i - b_i, a_j - b_j). We store P in an array indexed by
# [a_i - b_i + 1, a_j - b_j + 1] and these index arrays broadcast it to the axes
# (a_i, a_j, b_i, b_j) of the density matrix.
_BITS = np.arange(2)
_S_IDX = _BITS.reshape(2, 1, 1, 1) - _BITS.reshape(1, 1, 2, 1) + 1
_T_IDX = _BITS.reshape(1, 2, 1, 1) - _BITS.reshape(1, 1, 1, 2) + 1


@dataclass
class _Structure:
    """Data that only depends on the cost operator and the ansatz circuit."""

    num_qubits: int

    # Off-diagonal part of the ansatz graph, i.e., the Rzz gates, in CSR format.
    neighbours: sparse.csr_array

    # Sum over w_ik of the ansatz graph, including the diagonal, i.e., the Rz gates.
    row_sums: np.ndarray

    # The two-local terms of the cost operator and the ansatz weight on these edges.
    edges_i: np.ndarray
    edges_j: np.ndarray
    edge_weights: np.ndarray
    edge_circuit_weights: np.ndarray

    # Contiguous groups of edges for which we compute the neighbour products at once.
    chunks: list[slice]

    # The one-local terms of the cost operator.
    diag_idx: np.ndarray
    diag_weights: np.ndarray

    # Cached neighbour data. Only filled for the edges if everything fits in a single chunk.
    edge_pairs: dict[int, tuple] = field(default_factory=dict)
    diag_pairs: tuple | None = None


class EfficientDepthOneEvaluator(BaseEvaluator):
    r"""Class to efficiently evaluate the energy of a depth-one QAOA.

    A description of the method that this class implements can be found in
    the [warm-start QAOA paper](https://quantum-journal.org/papers/q-2021-06-17-479/).
    This description is in Appendix F. The method computes the one- and two-local
    correlators from 2x2 and 4x4 density matrices. This works for any problem
    connectivity, even for dense graphs, which sets it apart from light-cone based
    methods that typically perform well on sparse graphs.

    The implementation is vectorized over all the terms of the cost operator. It relies on
    the fact that every gate that precedes the mixer is diagonal. For a diagonal unitary
    :math:`U={\rm diag}(d)` the conjugation :math:`U\rho U^\dagger` acts elementwise, i.e.,
    :math:`\rho_{ab}\to d_a\rho_{ab}d_b^*`. Tracing out a neighbour :math:`k` that is in
    :math:`|1\rangle` with probability :math:`c_k` therefore multiplies each entry of the
    density matrix of qubits :math:`i` and :math:`j` by

    .. math::

        (1 - c_k) + c_k\exp\left(-4i\gamma[(a_i-b_i)w_{ik} + (a_j-b_j)w_{jk}]\right).

    The products of these factors over the neighbours only take four independent values
    per edge. Finally, the one-local mixers :math:`M_i` are absorbed in the observables
    since :math:`\langle Z_iZ_j\rangle={\rm Tr}[(A_i\otimes A_j)\rho_{ij}]` with
    :math:`A_i = M_i^\dagger Z M_i`. This also supports warm-start mixers that differ
    from qubit to qubit.

    Data that does not depend on the QAOA parameters, such as the graphs and the initial
    state, is cached between calls to :meth:`evaluate`. The cache is keyed on the identity
    of the arguments. Therefore, modifying a cost operator or a circuit in place between
    two calls is not detected.

    Limitations:
      * Quadratic forms, i.e., sum over :math:`Z_i Z_j` and :math:`Z_i`.
      * Depth-one QAOA.
      * One-local mixers and initial states.
    """

    def __init__(self, max_pairs: int = 2**20) -> None:
        """Initialize the class.

        Args:
            max_pairs: The maximum number of (edge, neighbour) pairs that are processed at
                once. This bounds the peak memory of an evaluation, which is roughly
                150 bytes per pair. If all pairs of a problem fit within this bound
                they are cached between evaluations.
        """
        super().__init__()

        if max_pairs < 1:
            raise ValueError(f"max_pairs must be a positive integer. Received {max_pairs}.")

        self._max_pairs = max_pairs

        # The caches. The keys hold references to the arguments of `evaluate` so that
        # their ids cannot be reused by other objects while they are cached.
        self._structure_key: tuple | None = None
        self._structure: _Structure | None = None
        self._initial_state_key: tuple | None = None
        self._amplitudes: np.ndarray | None = None
        self._mixer_key: tuple | None = None
        self._mixer_gates: list[list] | None = None
        self._observables_key: tuple | None = None
        self._observables_beta: float | None = None
        self._observables: np.ndarray | None = None

    # pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list[float],
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the energy.

        Args:
            cost_op: The cost operator that defines :math:`H_C`. This operator is converted to
                a graph which describes the correlators to measure to build-up :math:`H_c` and
                their weights. The cost operator can only be made of one- and two-local
                :math:`Z` terms.
            params: The parameters for QAOA. They are a list of length two and
                correspond to [beta, gamma].
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. The mixer must be made of
                single-qubit gates only and have a single parameter which is beta. Different
                qubits may have different mixers, as in warm-start QAOA.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
                It must be made of single-qubit gates only.
            ansatz_circuit: The ansatz circuit for the cost operator. This is internally converted
                to a graph that describes the structure of the Ansatz circuit. This argument is
                optional. If it is not given then we assume that it corresponds to the full graph
                given as argument. This is the default case of QAOA. This argument allows us to
                work with a different circuit Ansatz than the default from QAOA.

        Raises:
            ValueError: If there are not exactly two parameters or if the mixer or the initial
                state do not match the number of qubits of the cost operator.
            NotImplementedError: If the mixer or the ansatz are of an unsupported type.

        Returns:
            The energy for the given graph.
        """
        if len(params) != 2:
            raise ValueError("Efficient depth one only supports two parameters.")

        beta, gamma = float(params[0]), float(params[1])

        structure = self._get_structure(cost_op, ansatz_circuit)
        amplitudes = self._get_amplitudes(structure.num_qubits, initial_state)
        observables = self._get_observables(beta, structure.num_qubits, mixer)

        prob_1 = np.abs(amplitudes[:, 1]) ** 2

        energy = self._two_local_energy(structure, amplitudes, prob_1, observables, gamma)
        energy += self._one_local_energy(structure, amplitudes, prob_1, observables, gamma)

        return energy

    @staticmethod
    def mixer(beta: float) -> np.ndarray:
        """The mixer operator.

        Args:
            beta: The rotation angle of the mixer.

        Returns:
            The single-qubit mixer :math:`\\exp(-i\\beta X)` as a 2x2 matrix.
        """
        exp_m = np.exp(1.0j * beta)
        exp_p = np.exp(-1.0j * beta)

        mixer = np.array(
            [
                [0.5 * exp_p + 0.5 * exp_m, 0.5 * (exp_p - exp_m)],
                [0.5 * (exp_p - exp_m), 0.5 * exp_m + 0.5 * exp_p],
            ],
            dtype=complex,
        )

        return mixer

    @staticmethod
    def _is_cached(key: tuple | None, *objects) -> bool:
        """Return True if the key was created from exactly these objects."""
        return (
            key is not None
            and len(key) == len(objects)
            and all(cached is obj for cached, obj in zip(key, objects))
        )

    def _get_structure(
        self,
        cost_op: SparsePauliOp,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None,
    ) -> _Structure:
        """Return the parameter-independent data of the cost operator and the ansatz."""
        if self._structure is not None and self._is_cached(
            self._structure_key, cost_op, ansatz_circuit
        ):
            return self._structure

        num_qubits = cost_op.num_qubits
        assert num_qubits is not None, "The cost operator must have a number of qubits."
        cost_graph = _to_csr(operator_to_graph(cost_op), num_qubits)

        if ansatz_circuit is None:
            circuit_graph = cost_graph
        elif isinstance(ansatz_circuit, QuantumCircuit):
            circuit_graph = _to_csr(circuit_to_graph(ansatz_circuit), num_qubits)
        else:
            raise NotImplementedError(
                f"ansatz_circuit of type {type(ansatz_circuit).__name__} is not supported. "
                "Only QuantumCircuit is supported."
            )

        # The two-local and one-local terms of the cost operator.
        lower = sparse.tril(cost_graph, k=-1, format="coo")
        keep = lower.data != 0.0
        edges_i, edges_j = lower.row[keep].astype(np.int64), lower.col[keep].astype(np.int64)
        cost_diag = cost_graph.diagonal()
        diag_idx = np.flatnonzero(cost_diag)

        # Split the ansatz graph into the Rz gates (diagonal) and the Rzz gates (off-diagonal).
        row_sums = np.asarray(circuit_graph.sum(axis=1)).ravel()
        coo = circuit_graph.tocoo()
        off_diag = (coo.row != coo.col) & (coo.data != 0.0)
        neighbours = sparse.csr_array(
            (coo.data[off_diag], (coo.row[off_diag], coo.col[off_diag])),
            shape=(num_qubits, num_qubits),
        )
        neighbours.sum_duplicates()
        neighbours.sort_indices()

        # Group the edges such that each group has at most `max_pairs` (edge, neighbour) pairs,
        # unless a single edge has more pairs than that.
        degrees = np.diff(neighbours.indptr)
        lengths = degrees[edges_i] + degrees[edges_j]
        chunk_ids = (np.cumsum(lengths) - lengths) // self._max_pairs
        bounds = [0, *(np.flatnonzero(np.diff(chunk_ids)) + 1), len(edges_i)]
        chunks = [slice(start, stop) for start, stop in pairwise(bounds)]

        self._structure = _Structure(
            num_qubits=num_qubits,
            neighbours=neighbours,
            row_sums=row_sums,
            edges_i=edges_i,
            edges_j=edges_j,
            edge_weights=lower.data[keep],
            edge_circuit_weights=_lookup(neighbours, edges_i, edges_j),
            chunks=chunks,
            diag_idx=diag_idx,
            diag_weights=cost_diag[diag_idx],
        )
        self._structure_key = (cost_op, ansatz_circuit)

        return self._structure

    def _get_amplitudes(self, num_qubits: int, initial_state: QuantumCircuit | None) -> np.ndarray:
        """Return the single-qubit amplitudes of the initial state as an (n, 2) array."""
        if (
            self._amplitudes is not None
            and len(self._amplitudes) == num_qubits
            and self._is_cached(self._initial_state_key, initial_state)
        ):
            return self._amplitudes

        if initial_state is None:
            amplitudes = np.full((num_qubits, 2), np.sqrt(0.5), dtype=complex)
        else:
            splits = split_circuit(initial_state)

            if len(splits) != num_qubits:
                raise ValueError(
                    f"The initial state has {len(splits)} qubits but the cost operator "
                    f"has {num_qubits}."
                )

            amplitudes = np.array([Operator(sub_circuit).data[:, 0] for sub_circuit in splits])

        self._amplitudes = amplitudes
        self._initial_state_key = (initial_state,)

        return self._amplitudes

    def _get_observables(
        self, beta: float, num_qubits: int, mixer: QuantumCircuit | None
    ) -> np.ndarray:
        r"""Return the observables :math:`A_i = M_i^\dagger Z M_i` as an (n, 2, 2) array.

        Currently, the code only supports one-local mixers. I.e. mixers made of
        arbitrary single-qubit rotations.
        """
        if (
            self._observables is not None
            and len(self._observables) == num_qubits
            and self._is_cached(self._observables_key, mixer)
            and self._observables_beta == beta
        ):
            return self._observables

        if mixer is None:
            mixers = self.mixer(beta)[None, :, :]
        elif isinstance(mixer, QuantumCircuit):
            if mixer.num_parameters > 1:
                raise ValueError(
                    f"The mixer must have a single parameter. It has {mixer.num_parameters}."
                )

            if self._mixer_gates is None or not self._is_cached(self._mixer_key, mixer):
                self._mixer_gates = [_prepare_gates(sub) for sub in split_circuit(mixer)]
                self._mixer_key = (mixer,)

            if len(self._mixer_gates) != num_qubits:
                raise ValueError(
                    f"The mixer has {len(self._mixer_gates)} qubits but the cost operator "
                    f"has {num_qubits}."
                )

            mixers = np.array([_gates_to_matrix(gates, beta) for gates in self._mixer_gates])
        else:
            raise NotImplementedError(
                f"Mixer of type {type(mixer).__name__} is not supported. "
                "Only QuantumCircuit mixers are supported."
            )

        observables = np.einsum("eba,b,ebc->eac", mixers.conj(), _Z, mixers)

        self._observables = np.broadcast_to(observables, (num_qubits, 2, 2))
        self._observables_key = (mixer,)
        self._observables_beta = beta

        return self._observables

    def _two_local_energy(
        self,
        structure: _Structure,
        amplitudes: np.ndarray,
        prob_1: np.ndarray,
        observables: np.ndarray,
        gamma: float,
    ) -> float:
        """Compute the energy of the ZiZj terms of the cost operator."""
        energy = 0.0

        for chunk_idx, chunk in enumerate(structure.chunks):
            edges_i, edges_j = structure.edges_i[chunk], structure.edges_j[chunk]
            num_edges = len(edges_i)

            if num_edges == 0:
                continue

            pairs = structure.edge_pairs.get(chunk_idx)
            if pairs is None:
                pairs = _edge_neighbours(structure.neighbours, edges_i, edges_j)
                if len(structure.chunks) == 1:
                    structure.edge_pairs[chunk_idx] = pairs

            counts, k, w_ik, w_jk = pairs

            # Neighbour products P(s, t) for (s, t) in (1, 0), (0, 1), (1, 1), and (1, -1).
            # The other combinations are complex conjugates, and P(0, 0) = 1.
            c_k = prob_1[k][:, None]
            phase_i, phase_j = np.exp(-4.0j * gamma * w_ik), np.exp(-4.0j * gamma * w_jk)
            phases = np.stack(
                [phase_i, phase_j, phase_i * phase_j, phase_i * phase_j.conj()], axis=1
            )
            products = _segment_prod((1.0 - c_k) + c_k * phases, counts)

            # Neighbour products indexed by [s + 1, t + 1].
            p_st = np.ones((num_edges, 3, 3), dtype=complex)
            for col, (s_i, t_j) in enumerate([(1, 0), (0, 1), (1, 1), (1, -1)]):
                p_st[:, s_i + 1, t_j + 1] = products[:, col]
                p_st[:, 1 - s_i, 1 - t_j] = products[:, col].conj()

            # The Rzz gate between i and j, i.e., diag(exp(-i gamma w_ij z_i z_j)).
            w_ij = structure.edge_circuit_weights[chunk]
            rzz = np.exp(-1.0j * gamma * w_ij[:, None, None] * np.outer(_Z, _Z)[None, :, :])
            factors = (
                p_st[:, _S_IDX, _T_IDX] * rzz[:, :, :, None, None] * rzz.conj()[:, None, None, :, :]
            )

            # The single-qubit states after the Rz gates. These collect the local terms of
            # the Rzz gates with the qubits k != i, j and the Rz gates of the ansatz.
            w_i = structure.row_sums[edges_i] - w_ij
            w_j = structure.row_sums[edges_j] - w_ij
            q_i = amplitudes[edges_i] * np.exp(-1.0j * gamma * w_i[:, None] * _Z[None, :])
            q_j = amplitudes[edges_j] * np.exp(-1.0j * gamma * w_j[:, None] * _Z[None, :])

            # x[e, a, b] = q_a q_b^* A_{ba} so that the contraction below is Tr[(A_i x A_j) rho].
            x_i = q_i[:, :, None] * q_i.conj()[:, None, :] * observables[edges_i].transpose(0, 2, 1)
            x_j = q_j[:, :, None] * q_j.conj()[:, None, :] * observables[edges_j].transpose(0, 2, 1)

            correlators = np.einsum("eab,ecd,eacbd->e", x_i, x_j, factors)
            energy += float(np.real(structure.edge_weights[chunk] @ correlators))

        return energy

    def _one_local_energy(
        self,
        structure: _Structure,
        amplitudes: np.ndarray,
        prob_1: np.ndarray,
        observables: np.ndarray,
        gamma: float,
    ) -> float:
        """Compute the energy of the Zi terms of the cost operator."""
        idx = structure.diag_idx

        if len(idx) == 0:
            return 0.0

        if structure.diag_pairs is None:
            structure.diag_pairs = _node_neighbours(structure.neighbours, idx)

        counts, k, w_ik = structure.diag_pairs

        c_k = prob_1[k]
        products = _segment_prod((1.0 - c_k) + c_k * np.exp(-4.0j * gamma * w_ik), counts)

        q_i = amplitudes[idx] * np.exp(-1.0j * gamma * structure.row_sums[idx][:, None] * _Z)
        rho = q_i[:, :, None] * q_i.conj()[:, None, :]

        # Entry (a, b) is multiplied by the neighbour product P(a - b).
        rho[:, 1, 0] *= products
        rho[:, 0, 1] *= products.conj()

        correlators = np.einsum("eab,eba->e", observables[idx], rho)

        return float(np.real(structure.diag_weights @ correlators))

    @classmethod
    def from_config(cls, config: dict) -> "EfficientDepthOneEvaluator":
        """Create the evaluator from a configuration dictionary.

        Args:
            config: Configuration dictionary that may contain:
                - max_pairs (int): The maximum number of (edge, neighbour) pairs
                  processed at once.

        Returns:
            An instance of EfficientDepthOneEvaluator.
        """
        return cls(**config)


def _to_csr(graph: nx.Graph, num_qubits: int) -> sparse.csr_array:
    """Convert a graph to a CSR adjacency matrix with a fixed node order."""
    return sparse.csr_array(
        nx.to_scipy_sparse_array(graph, nodelist=range(num_qubits), format="csr")
    )


def _prepare_gates(circuit: QuantumCircuit) -> list:
    """Precompute the matrices of the gates of a single-qubit circuit that have no parameters.

    Returns:
        The gates in circuit order. Each entry is either a 2x2 matrix or a parameterized gate.
    """
    return [
        gate if gate.is_parameterized() else Operator(gate).data
        for gate in (inst.operation for inst in circuit.data)
    ]


def _gates_to_matrix(gates: list, value: float) -> np.ndarray:
    """Multiply the gates of a single-qubit circuit with all its parameters set to `value`."""
    matrix = np.eye(2, dtype=complex)

    for gate in gates:
        if isinstance(gate, np.ndarray):
            matrix = gate @ matrix
        else:
            bound = gate.copy()
            bound.params = [
                (
                    float(param.bind({p: value for p in param.parameters}))
                    if isinstance(param, ParameterExpression)
                    else param
                )
                for param in gate.params
            ]
            matrix = Operator(bound).data @ matrix

    return matrix


def _lookup(matrix: sparse.csr_array, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return the entries matrix[rows[m], cols[m]] as a dense float array."""
    if len(rows) == 0:
        return np.zeros(0)

    return np.asarray(matrix[rows, cols], dtype=float).ravel()


def _concatenated_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return the concatenation of range(start, start + length) for all starts and lengths."""
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + offsets


def _node_neighbours(neighbours: sparse.csr_array, nodes: np.ndarray) -> tuple:
    """List the neighbours k of each node i together with the weight w_ik.

    Returns:
        The number of neighbours per node and, grouped by node, the neighbours k
        and the weights w_ik.
    """
    counts = np.diff(neighbours.indptr)[nodes]
    positions = _concatenated_ranges(neighbours.indptr[nodes], counts)

    return counts, neighbours.indices[positions], neighbours.data[positions]


def _edge_neighbours(neighbours: sparse.csr_array, rows: np.ndarray, cols: np.ndarray) -> tuple:
    """List the neighbours k of each edge (i, j), i.e., k in N(i) or N(j) with k != i, j.

    Returns:
        The number of neighbours per edge and, grouped by edge, the neighbours k and
        the weights w_ik and w_jk. At least one of the weights is non-zero.
    """
    indptr = neighbours.indptr
    deg_i = np.diff(indptr)[rows]
    deg_j = np.diff(indptr)[cols]

    # Each edge gets a segment with first the neighbours of i and then those of j.
    lengths = deg_i + deg_j
    edge = np.repeat(np.arange(len(rows)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    from_i = offset < deg_i[edge]

    own = np.where(from_i, rows[edge], cols[edge])
    other = np.where(from_i, cols[edge], rows[edge])
    positions = indptr[own] + np.where(from_i, offset, offset - deg_i[edge])

    k = neighbours.indices[positions]
    w_own = neighbours.data[positions]
    w_other = _lookup(neighbours, other, k)

    # Drop the edge itself, i.e., k == j for i and vice versa. Neighbours of j that are
    # also neighbours of i have already been listed with the neighbours of i.
    keep = (k != other) & (from_i | (w_other == 0.0))

    w_ik = np.where(from_i, w_own, w_other)[keep]
    w_jk = np.where(from_i, w_other, w_own)[keep]
    counts = np.bincount(edge[keep], minlength=len(rows))

    return counts, k[keep], w_ik, w_jk


def _segment_prod(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Multiply consecutive groups of `counts` rows of `values`. Empty groups give one."""
    out = np.ones((len(counts),) + values.shape[1:], dtype=values.dtype)
    non_empty = counts > 0

    if len(values):
        starts = (np.cumsum(counts) - counts)[non_empty]
        out[non_empty] = np.multiply.reduceat(values, starts, axis=0)

    return out
