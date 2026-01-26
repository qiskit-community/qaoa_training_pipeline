#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to efficiently evaluate depth-one circuits."""

from collections.abc import Sequence

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.circuit_utils import split_circuit
from qaoa_training_pipeline.utils.graph_utils import circuit_to_graph, operator_to_graph


# cspell: words correlator correlators toarray kron
class EfficientDepthOneEvaluator(BaseEvaluator):
    r"""Class to efficiently evaluate the energy of a depth-one QAOA.

    A description of the method that this class implements can be found in
    the [warm-start QAOA paper](https://quantum-journal.org/papers/q-2021-06-17-479/).
    This description is in Appendix F. The method works by computing the two-local
    correlators one by one. The correlators are built-up from 4x4 density matrices.
    This function works for any problem connectivity, even for dense graphs, this sets
    it apart from light-cone based methods which typically perform well on sparse graphs.

    Limitations:
      * Quadratic forms, i.e., sum over :math:`Z_i Z_j`.
      * Depth-one QAOA.

    The limitation to quadratic forms could be removed in the future.
    """

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()
        self._mixers = None
        self._initial_states = None

    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: Sequence[float],
        mixer: BaseOperator | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the energy.

        Args:
            cost_op: The cost operator that defines :math:`H_C`. This operator is converted to
                an adjacency matrix of the graph which describes the correlators to measure to
                build-up :math:`H_c` and their weights. Currently, the cost operator can only be
                made of quadratic terms. This restriction may be removed in the future.
            params: The parameters for QAOA. They are a list of length two and
                correspond to [beta, gamma].
            mixer: A quantum circuit describing the mixer operator. If None is given, the default,
                then we assume that the mixer is the sum of X gates. A current limitation of the
                method is that the mixer is made of single-qubit rotations only.
            initial_state: The initial state. This is given to accommodate, e.g., warm-start QAOA.
            ansatz_circuit: The ansatz circuit for the cost operator. This is internally converted
                to an adjacency matrix of the graph that describes the structure of the
                Ansatz circuit. This argument is optional. If it is not given then we assume that
                it corresponds to the full graph given as argument. This is the default case of
                QAOA. This argument allows us to work with a different circuit Ansatz than the
                default from QAOA.

        Returns:
            The energy for the given graph.
        """
        pauli_list = cost_op.paulis[0]
        length = 0
        if isinstance(pauli_list, PauliList):
            length = len(pauli_list)
        elif isinstance(pauli_list, Pauli):
            length = pauli_list.num_qubits
            assert isinstance(length, int)
        graph = nx.adjacency_matrix(
            operator_to_graph(cost_op),
            nodelist=range(length),
        ).toarray()

        if len(params) != 2:
            raise ValueError("Efficient depth one only supports two parameters.")

        self._set_mixers(params[0], len(graph), mixer)
        self._set_initial_states(len(graph), initial_state)

        if ansatz_circuit is None:
            circuit_graph = graph
        elif isinstance(ansatz_circuit, QuantumCircuit):
            circuit_graph = nx.adjacency_matrix(
                circuit_to_graph(ansatz_circuit),
                nodelist=range(length),
            ).toarray()
        else:
            raise NotImplementedError(
                f"ansatz_circuit of type {type(ansatz_circuit).__name__} is not supported. "
                "Only QuantumCircuit is supported."
            )

        # Compute the energy from the two-qubit correlators
        energy = sum(
            (
                graph[i, j] * self.correlator(i, j, circuit_graph, params[1])
                if graph[i, j] != 0.0
                else 0.0
            )
            for i in range(len(graph))
            for j in range(0, i)
        )

        # Compute the energy from the single-qubit terms
        energy += sum(
            (graph[i, i] * self.single_z(i, circuit_graph, params[1]) if graph[i, i] != 0 else 0.0)
            for i in range(len(graph))
        )

        return np.real(energy)

    @staticmethod
    def mixer(beta: float) -> np.ndarray:
        """The mixer operator.

        Args:
            beta: The rotation angle of the mixer.
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

    def _set_initial_states(
        self, num_qubits: int, initial_state: QuantumCircuit | None = None
    ) -> None:
        """Set the initial state."""
        if initial_state is None:
            equal_pop = np.array([[np.sqrt(0.5)], [np.sqrt(0.5)]], dtype=complex)
            self._initial_states = [equal_pop] * num_qubits
        else:
            splits = split_circuit(initial_state)

            self._initial_states = []
            for sub_circuit in splits:
                data = np.array(Operator(sub_circuit).data)
                data = data[:, 0].reshape(2, 1)
                self._initial_states.append(data)

    def _set_mixers(self, beta: float, num_qubits: int, mixer: BaseOperator | None = None) -> None:
        """Set the mixer operators per qubit.

        Currently, the code only supports one-local mixers. I.e. mixers made of
        arbitrary single-qubit rotations.
        """
        if mixer is None:
            mix = self.mixer(beta)
            self._mixers = [mix] * num_qubits
        elif isinstance(mixer, QuantumCircuit):
            splits = split_circuit(mixer.assign_parameters([beta], inplace=False))
            self._mixers = [Operator(sub_circ).data for sub_circ in splits]
        else:
            raise NotImplementedError(
                f"Mixer of type {type(mixer).__name__} is not supported. "
                "Only QuantumCircuit mixers are supported."
            )

    def single_z(self, idx, circuit_graph: np.ndarray, gamma: float):
        """Compute the energy of <Zi>"""
        n = len(circuit_graph)

        # Initial state of the qubits as generated by a one-local circuit.
        assert self._initial_states, "_initial states must be initialized before calling single_z"
        qi = self._initial_states[idx].copy()

        # (i) Apply the U1 gates of the Rzz gates that come from qubit k neq i
        # (ii) and the single-qubit Rz gate from the cost operator on qubit i
        # (i) and (ii) are conceptually different but code-wise it is more compact to combine.
        wi = sum(circuit_graph[idx, k] for k in range(n))

        # Apply Rz(2 gamma \sum_k w_ik) to qubit i
        qi[0] *= np.exp(-1.0j * (2 * wi * gamma) / 2)
        qi[1] *= np.exp(1.0j * (2 * wi * gamma) / 2)

        # Switch to density matrices and compute the effect of
        # the two-qubit CP gate that comes from qubit k neq i,j
        rho_i = qi * qi.T.conj()

        for k in range(n):
            if k == idx:
                continue

            phase_i = np.exp(-1.0j * 4 * gamma * circuit_graph[idx, k])

            u1 = np.diag([1.0, phase_i])

            rho_i = 0.5 * rho_i + 0.5 * np.dot(u1, np.dot(rho_i, u1.conj().T))

        # Apply the mixer operator
        assert self._mixers, "_mixers must be defined before calling single_z()"
        mixer_i = self._mixers[idx]

        rho_i = np.dot(mixer_i, np.dot(rho_i, mixer_i.conj().T))

        return np.real(rho_i[0, 0] - rho_i[1, 1])

    def correlator(self, idx1: int, idx2: int, circuit_graph: np.ndarray, gamma: float):
        r"""Computes the correlator <ZiZj>

        Convention: for the mixer we apply rotations of the form :math:`R_{x}(2\beta)`.
        For each term in the cost operator we apply the gates :math:`Rzz(2\gamma w_{ij})`
        this corresponds to a cost operator of the form:

        ..math::

            \sum_{i,j=0}^{n-1}w_{i,j}Z_iZ_j

        Now, we decompose the gate :math:`Rzz(2\gamma w_{ij})` into a controlled-phase
        gate and local phase gates. The following identity holds

        ..math::

            Rzz(\theta) = \exp(i\theta/2)\cdot Rz(\theta)\otimes Rz(\theta)\cdot CP(-2\theta)

        Here, :math:`CP(\theta)=\text{diag}(1,1,1,e^{i\theta})`.

        Args:
            idx1: The first index of the correlator.
            idx2: The second index of the correlator.
            circuit_graph: The circuit graph describes the structure of the QAOA
                cost operator. This graph may be different from the one for which
                we want to compute the correlators because the circuit Ansatz may
                be an approximation of the ideal circuit Ansatz for :math:`exp(-i\gamma H_c)`
            gamma: The gamma of the QAOA.
        """
        n = len(circuit_graph)

        # Initial state of the qubits as equal superposition of 0 and 1, i.e. |+>.
        assert self._initial_states, "_initial_states must be defined before calling single_z()"
        qi = self._initial_states[idx1].copy()
        qj = self._initial_states[idx2].copy()

        # Apply the U1 gates of the Rzz gates that come from qubit k neq i,j
        wi, wj = 0.0, 0.0
        for k in range(n):
            if k in {idx1, idx2}:
                continue

            wi += circuit_graph[idx1, k]
            wj += circuit_graph[idx2, k]

        # Apply and single Rz terms to qubit i
        wi += circuit_graph[idx1, idx1]
        wj += circuit_graph[idx2, idx2]

        # Apply Rz(2 gamma \sum_k w_ik) to qubit i
        qi[0] *= np.exp(-1.0j * (2 * wi * gamma) / 2)
        qi[1] *= np.exp(1.0j * (2 * wi * gamma) / 2)

        # Apply Rz(2 gamma \sum_k w_jk) to qubit j
        qj[0] *= np.exp(-1.0j * (2 * wj * gamma) / 2)
        qj[1] *= np.exp(1.0j * (2 * wj * gamma) / 2)

        # Switch to density matrices and compute the effect of
        # the two-qubit CP gate that comes from qubit k neq i,j
        rho_i, rho_j = qi * qi.T.conj(), qj * qj.T.conj()
        rho_ij = np.kron(rho_i, rho_j)

        for k in range(n):
            if k in {idx1, idx2}:
                continue

            if circuit_graph[idx1, k] == 0.0 and circuit_graph[idx2, k] == 0.0:
                continue

            phase_i = np.exp(-1.0j * 4 * gamma * circuit_graph[idx1, k])
            phase_j = np.exp(-1.0j * 4 * gamma * circuit_graph[idx2, k])
            u1 = np.diag([1.0, phase_j, phase_i, phase_i * phase_j])

            rho_ij = 0.5 * rho_ij + 0.5 * np.dot(u1, np.dot(rho_ij, u1.conj().T))

        # Apply the two-qubit Rzz gate between `i` and `j`
        if circuit_graph[idx1, idx2] != 0.0:
            phase_m = np.exp(-1.0j * (2 * gamma * circuit_graph[idx1, idx2]) / 2)
            phase_p = np.exp(1.0j * (2 * gamma * circuit_graph[idx1, idx2]) / 2)
            u_ij = np.diag([phase_m, phase_p, phase_p, phase_m])
            rho_ij = np.dot(u_ij, np.dot(rho_ij, u_ij.conj().T))

        # Apply the mixer operator
        assert self._mixers, "_mixers must be defined before calling single_z()"
        mixer_i = self._mixers[idx1]
        mixer_j = self._mixers[idx2]
        assert isinstance(mixer_i, np.ndarray)

        mixer_ij = np.kron(mixer_i, mixer_j)
        rho_ij = np.dot(mixer_ij, np.dot(rho_ij, mixer_ij.conj().T))

        return np.real(rho_ij[0, 0] - rho_ij[1, 1] - rho_ij[2, 2] + rho_ij[3, 3])

    @classmethod
    def from_config(cls, config: dict) -> "EfficientDepthOneEvaluator":
        """Create the evaluator."""
        return cls(**config)
