#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Perform a naive light-cone simulation of QAOA."""

from typing import Any, Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from qiskit import transpile

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class LightConeEvaluator(BaseEvaluator):
    r"""Light-cone computation of QAOA.

    This class allows us to compute the energy of QAOA circuits for very large
    graphs that are not dense and for moderate p values. For each correlator
    :math:`\langle ZiZj\rangle` that QAOA needs, this evaluator builds a smaller
    graph from the original graph by considering the light-cone of the QAOA
    circuit. For example, as discussed in Sack & Egger, a random-three-regular
    graph of arbitrary size will have light-cone circuits with at most 14 qubits.
    This allows us to compute exactly the energy for such graphs. The limit in
    size that this class can handle depends on the graph density and the depth
    of the QAOA that should be simulated.

    Note: this class currently does not use the statevector for simulation.
    This could be improved on in subsequent PRs.
    """

    def __init__(self, shots: int = 4096, estimator: Optional[Any] = None):
        """Initialize the light-cone evaluator.

        Args:
            shots: Number of shots to gather.
            estimator: The estimator on which to compute the expectation values.
                This will default to StatevectorEstimator. This object must have
                a `run` method which takes tasks (list of tuples) as an input.
        """
        self.shots = shots

        # Maximum number of qubits that we allow in the light-cone.
        self.qubit_threshold = 32

        self.primitive = estimator or StatevectorEstimator()

        # The graph from which we will take sub-graphs using the light-cone.
        self.graph = None

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> float:
        r"""Evaluate the energy.

        Args:
            cost_op: The cost operator that defines :math:`H_C`. Currently, this cost
                operator can only be quadratic.
            params: The parameters for QAOA. Their length will determine the depth of the
                QAOA. The params are given in the order `[beta0, beta1, ..., gamma0, gamma1, ...].
            mixer: Mixer of QAOA which defaults to the sum of X's. Custom mixers are currently
                not supported.
            initial_state: The initial state of QAOA which defaults to the equal superposition.
                Custom initial states are currently not supported.
            ansatz_circuit: A custom ansatz circuit to use. Currently, custom ansatz circuits are
                not supported.

        Raises:
            ValueError if the number of qubits in the circuit exceeds the maximum
            threshold of the class. By default, this threshold is set to 32 qubits.
        """
        if initial_state is not None:
            raise NotImplementedError(
                "Custom initial states are not yet supported. To support them the function "
                "make_radius_circuit will need to select the corresponding qubits."
            )

        if mixer is not None:
            raise NotImplementedError(
                "Custom mixers are not yet supported. To support them the function "
                "make_radius_circuit will need to select the corresponding qubits."
            )

        if ansatz_circuit is not None:
            raise NotImplementedError(
                "ansatz_circuits are currently not supported. Only circuits that match "
                "the original QAOA construction are used."
            )

        self.graph = operator_to_graph(cost_op)

        tasks, weights = [], []
        for node1, node2, data in self.graph.edges(data=True):
            circ, obs = self.make_radius_circuit((node1, node2), params)
            circ = circ.decompose()

            if circ.num_qubits > self.qubit_threshold:
                raise ValueError(
                    f"The number of qubits in the light-cone of ({node1}, {node2}) "
                    f"exceeds the allowable threshold of {self.qubit_threshold}."
                )

            tasks.append((circ, obs))
            weights.append(data["weight"] if "weight" in data else 1.0)

        results = list(res for res in self.primitive.run(tasks).result())

        # Compute and return the observable.
        energy = sum(w * float(res.data.evs) for res, w in zip(results, weights))

        return np.real(energy)

    def make_radius_circuit(
        self,
        edge: Tuple[int, int],
        params: List[float],
        initial_state: Optional[QuantumCircuit] = None,
        mixer_operator: Optional[QuantumCircuit] = None,
    ) -> Tuple[QuantumCircuit, str]:
        r"""Create the circuit for the given edge.

        This method proceeds by first shrinking the graph of the problem to the light cone.
        Next, it uses `qaoa_ansatz` to create the circuit for this smaller scale version
        of the graph.

        Args:
            edge: The edge that we want to measure.
            params: The parameters of QAOA in order [beta0, beta1, ..., gamma0, gamma1, ...]
                to match with `qaoa_ansatz`.
            initial_state: The initial state of all qubits.
            mixer_operator: The mixer operator.
        """
        edges = self.make_radius_edges(edge, radius=len(params) // 2)

        paulis, src_edge = self.make_sub_correlators(edges, edge, len(self.graph))

        ansatz = qaoa_ansatz(
            cost_operator=SparsePauliOp.from_list(paulis),
            reps=len(params) // 2,
            initial_state=initial_state,
            mixer_operator=mixer_operator,
        )

        # Assumes the same parameter order as qaoa_ansatz.
        ansatz.assign_parameters(params, inplace=True)

        ansatz = transpile(ansatz, basis_gates=["sx", "x", "rzz", "rz", "rx"])

        # Create the observable
        obs = "".join(["I" if idx not in src_edge else "Z" for idx in range(len(paulis[0][0]))])

        return ansatz, obs[::-1]

    def make_radius_edges(self, edge: Tuple[int, int], radius: int) -> Dict[Tuple, float]:
        r"""Make the subset of edges that are a radius p away.

        Args:
            edge: The edge for which we want to construct the `ZiZj` correlator.
            radius: The radius of the light-cone, given by the QAOA layers `p`.

        Returns:
            A dict where the keys are the edges in the graph to include and the values
            are the weights of the edges.
        """
        ego1 = nx.generators.ego_graph(self.graph, edge[0], radius=radius)
        ego2 = nx.generators.ego_graph(self.graph, edge[1], radius=radius)

        # Build the edges we need ton consider taking weights into account.
        edges = {}
        for node1, node2, data in ego1.edges(data=True):
            edges[(node1, node2)] = data.get("weight", 1.0)

        for node1, node2, data in ego2.edges(data=True):
            edges[(node1, node2)] = data.get("weight", 1.0)

        # add back edges that are in G that cross from one ego to the other.
        for node1, node2, data in self.graph.edges(data=True):
            if (
                node1 in ego1.nodes
                and node2 in ego2.nodes
                or node2 in ego1.nodes
                and node1 in ego2.nodes
            ):
                edges[(node1, node2)] = data.get("weight", 1.0)

        # Ensure that we do not double count the edges.
        sub_edges = {}
        for lc_edge in edges:
            if lc_edge not in sub_edges and lc_edge[::-1] not in sub_edges:
                sub_edges[lc_edge] = edges[lc_edge]

        return sub_edges

    @staticmethod
    def make_sub_correlators(
        edges: Dict[Tuple[int, int], float],
        source_edge: Tuple[int, int],
        base_size: int,
    ) -> Tuple[List[Tuple[str, float]], Tuple]:
        r"""Build Paulis from the edges to construct a cost_op for `qaoa_ansatz`.

        First construct an array where each row is a correlator.
        Each I is a 0 and each Z is a 1. Therefore, each row sums to 2 (since QUBOs)
        and the columns that do not sum to 0 have non-idle qubits. We can thus discard
        the qubits that are idle.

        Args:
            edges: A set of edges for which we construct correlators that do not
                contain idle qubits.
            source_edge: The edge for which we want to compute the local correlator.
            base_size: The size of the original problem.

        Returns:
            A list of Pauli correlators, with their weight, as well as the index of the
            source edge.
        """

        correlators, masks, weights = [], [], []
        for edge, weight in edges.items():
            mask = [0] * base_size
            paulis = ["I"] * base_size
            mask[edge[0]], mask[edge[1]] = 1, 1
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            correlators.append(paulis)
            masks.append(mask)
            weights.append([weight] * base_size)  # This is a bit inefficient.

        num_correl = len(correlators)
        correlators = np.array(correlators)
        weights = np.array(weights)

        indices = np.sum(masks, axis=0) > 0  # Columns that sum to 0 are idle qubits

        # Identify the new indices of the source edge so that we can later place measurements
        src_idx1 = source_edge[0] - sum(not idx for idx in indices[0 : source_edge[0]])
        src_idx2 = source_edge[1] - sum(not idx for idx in indices[0 : source_edge[1]])

        # The new length of the correlators is the number of columns that do not sum to 0.
        new_len = sum(indices)
        indices = np.tile(indices, num_correl).reshape(num_correl, len(indices))

        filtered_correl = correlators[indices].reshape((num_correl, new_len))
        filtered_weights = weights[indices].reshape((num_correl, new_len))

        paulis = []
        for idx, pauli in enumerate(filtered_correl):
            paulis.append(("".join(pauli)[::-1], filtered_weights[idx][0]))

        return paulis, (src_idx1, src_idx2)

    @classmethod
    def from_config(cls, config: dict) -> "LightConeEvaluator":
        """Create a light-cone evaluator from a config.

        TODO: The estimator may be an issue here.
        We could deal with that by having some strings to specify it.
        This would work at least for estimators based on simulators which is probably enough
        to start of with.
        """
        return cls(**config)

    def to_config(self) -> Dict:
        """Json serializable config to keep track of how results are generated."""
        config = super().to_config()

        config["primitive"] = self.primitive.__class__.__name__
        config["shots"] = self.shots

        return config
