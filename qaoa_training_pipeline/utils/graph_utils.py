#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions to help work with graphs and cost operators."""

import copy
import json
from typing import Optional, List, Tuple
from collections.abc import Mapping
from collections import defaultdict
import networkx as nx
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.problems.quadratic_objective import ObjSense


def operator_to_graph(
    operator: SparsePauliOp, pre_factor: float = 1.0, include_one_local: bool = True
) -> nx.Graph:
    """Convert a cost operator to a graph.

    Limitations:
    * Restricted to quadratic cost operators given as sums over :math:`Z_iZ_j`.
    * Weighted quadratic cost operators are accepted and result in weighted graphs.

    Args:
        operator: The operator to convert to a graph.
        pre_factor: The prefactor that will be applied to all the edges ontop of any weight
            that the Pauli terms may have. For example, if `pre_factor` is 2 and the term
            `ZiZj` has a weight of -3 then the graph with have an edge between nodes `(i, j)`
            with a weight of -6.
        include_one_local: If this is set to True (the default value), then the one-local
            Pauli terms will be included in the graph as self edges.

    Raises:
        ValueError if the operator is not quadratic.
    """
    graph, edges = nx.Graph(), []

    n_qubits = operator.num_qubits
    graph.add_nodes_from(range(n_qubits))

    for pauli_str, weight in operator.to_list():
        edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]

        if len(edge) == 1:
            if include_one_local:
                edges.append((edge[0], edge[0], pre_factor * np.real(weight)))
        elif len(edge) == 2:
            edges.append((edge[0], edge[1], pre_factor * np.real(weight)))
        else:
            raise ValueError(f"The operator {operator} is not Quadratic.")

    graph.add_weighted_edges_from(edges)

    return graph


def operator_to_list_of_hyperedges(
    operator: SparsePauliOp, pre_factor: float = 1.0
) -> List[Tuple[List[int], float]]:
    """Convert a cost operator into a list of edges

    Unlike `operator_to_graph`, which converts the sparse Pauli operator
    into a networkx graph, here we represent the graph as a list of edges,
    with the corresponding weight. This function should be used if the
    cost operator is not quadratic, but contains higher-order terms.
    In fact, networkx does not support hyperedges

    Args:
        operator (SparsePauliOp): input cost operator
        pre_factor (float, optional): overall scaling factor for the operator.
            Defaults to 1.0.

    Returns:
        List[Tuple[List[int], float]]: list of edges and the corresponding weight
    """

    edges = []
    for pauli_str, weight in operator.to_list():
        edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]
        edges.append([edge, pre_factor * np.real(weight)])
    return edges


def graph_to_operator_with_partial_assignment(
    graph: nx.Graph,
    assignment: Mapping,  # {node: -1, 0, +1}, missing nodes treated as 0
    pre_factor: float = 1.0,
    include_constant: bool = True,
) -> tuple[SparsePauliOp, list[int], list[int]]:
    """
    Convert a graph into a SparsePauliOp with a partial spin assignment applied.

    Each edge (i, j) maps to:
      - Z_i Z_j term if both i and j are free (assignment 0),
      - Z_i term scaled by the fixed spin of j if j is fixed and i is free,
      - Z_j term scaled by the fixed spin of i if i is fixed and j is free,
      - a constant shift (identity term) if both i and j are fixed.

    Vertices that are assigned are removed from the hamiltonian, so that the number of qubits
    in the returning pauli strings is the number of unassigned nodes.
    Args:
        graph: A networkx undirected graph (optionally weighted via edge attr 'weight').
        assignment: Mapping {node: -1, 0, +1} for partial assignment. Missing nodes default to 0.
                    +1/-1 are fixed spins, 0 means free.
        pre_factor: Multiplies the edge weights (default 1.0).
                    For maximum cut conventions, set pre_factor to -0.5
        include_constant: If True, include the constant energy offset arising from edges between
                          two fixed vertices as an identity term. This does not affect optimization,
                          but preserves the exact energy.

    Returns:
        A tuple of:
            `SparsePauliOp` representing the reduced Hamiltonian on the current qubit register.
            list of free nodes
            list of fixed nodes
    """
    # Establish qubit ordering and index map
    nodes = list(graph.nodes())

    free_nodes = [v for v in nodes if assignment.get(v, 0) == 0]
    fixed_nodes = [v for v in nodes if assignment.get(v, 0) != 0]

    num_free_nodes = len(free_nodes)
    free_idx = {v: k for k, v in enumerate(free_nodes)}

    const = 0.0

    def get_fixed_spin(node):
        spin = assignment.get(node, 0)
        if spin not in (-1, 0, +1):
            raise ValueError(f"Assignment for node {node} must be in {{-1, 0, +1}}, got {spin}.")
        return spin

    paulis_coeffs_dict = defaultdict(float)
    for node_u, node_v, data in graph.edges(data=True):
        weight = data["weight"] if "weight" in data else 1.0
        spin_u = get_fixed_spin(node_u)
        spin_v = get_fixed_spin(node_v)

        if spin_u == 0 and spin_v == 0:
            # Free-free edge -> ZZ term
            paulis = ["I"] * num_free_nodes
            paulis[free_idx[node_u]] = "Z"
            paulis[free_idx[node_v]] = "Z"
            # Reverse for qiskit ordering convention
            pauli_str = "".join(paulis)[::-1]
            paulis_coeffs_dict[pauli_str] += pre_factor * weight

        elif spin_u != 0 and spin_v == 0:
            # u fixed, v free -> linear Z_v term with coefficient w * s_u
            paulis = ["I"] * num_free_nodes
            paulis[free_idx[node_v]] = "Z"
            pauli_str = "".join(paulis)[::-1]
            paulis_coeffs_dict[pauli_str] += pre_factor * weight * spin_u

        elif spin_u == 0 and spin_v != 0:
            # u free, v fixed -> linear Z_u term with coefficient w * s_v
            paulis = ["I"] * num_free_nodes
            paulis[free_idx[node_u]] = "Z"
            pauli_str = "".join(paulis)[::-1]
            paulis_coeffs_dict[pauli_str] += pre_factor * weight * spin_v

        else:
            # both fixed -> constant term: w * s_u * s_v
            const += pre_factor * weight * spin_u * spin_v

    pauli_list = [(p, c) for p, c in paulis_coeffs_dict.items() if c != 0.0]
    if include_constant and const != 0.0:
        pauli_list.append(("I" * num_free_nodes, const))  # identity term (no reverse needed)
    hamiltonian_reduced = SparsePauliOp.from_list(pauli_list)
    return hamiltonian_reduced, free_nodes, fixed_nodes


def graph_to_operator(graph: nx.Graph, pre_factor: float = 1.0) -> SparsePauliOp:
    """Convert a graph into a sparse Pauli operator.

    Here, each edge :math:`(i, j)` in the graph will map to a :math:`Z_iZ_j` term in the Paulis.

    Args:
        graph: A networkx graph that is possibly weighted.
        pre_factor: A pre_factor that multiplies the weights in the graph. This prefactor defaults
            to `1.0`. For example, to reproduce the cost operator of a maximum cut problem from a
            graph, this pre-factor must be set to -0.5 as discussed in detail in the Conventions
            section of the README of this repository.

    Returns:
        A `SparcePauliOp` instance that corresponds to the graph.
    """
    pauli_list = []
    for node1, node2, data in graph.edges(data=True):
        paulis = ["I"] * len(graph)
        paulis[node1], paulis[node2] = "Z", "Z"
        weight = data["weight"] if "weight" in data else 1.0
        pauli_list.append(("".join(paulis)[::-1], pre_factor * weight))

    return SparsePauliOp.from_list(pauli_list)


def graph_to_dict(graph: nx.Graph, description: Optional[str] = None) -> dict:
    """Create a json exportable dict for the graph."""
    edge_list = []
    for node1, node2, data in graph.edges(data=True):
        nodes = sorted([node1, node2])
        edge_list.append({"nodes": nodes, "weight": data.get("weight", 1)})

    return {"edge list": edge_list, "Description": description or "No description available."}


def dict_to_graph(input_dict: dict) -> nx.Graph:
    """Create a nx graph from a json exportable dict."""
    graph, edges = nx.Graph(), []

    for edge_data in input_dict["edge list"]:
        edge, weight = edge_data["nodes"], edge_data["weight"]
        edges.append((edge[0], edge[1], weight))

    graph.add_weighted_edges_from(edges)

    return graph


def circuit_to_graph(circuit: QuantumCircuit) -> nx.Graph:
    """Convert a circuit (corresponding to a QAOA cost operator) to a graph.

    This method allows us to convert a network of rzz gates into a graph.
    We assume that for each `Rzz(2 * w * γ, i, j)` gate we have an edge between nodes
    i and j with a weight `w`. Here, `γ` is the QAOA gamma parameter.

    Assumptions:
    * The circuit only contains Rzz operations.
    * Each of the rzz gates is parameterized by gamma.

    Raises:
        ValueError if the circuit contains anything else than a Rzz gates with one parameter.
        The function also raises if a Rzz gate is present multiple times on the same qubits.
        This is designed to make the graph that we generate unambiguous.
    """
    qreg = circuit.qregs[0]
    graph, edges = nx.Graph(), []
    graph.add_nodes_from(range(len(qreg)))
    seen_edges = set()

    for inst in circuit.data:
        iop = inst.operation

        if iop.name not in ["rzz", "rz"]:
            raise ValueError(
                f"Circuit must be composed of Rz or Rzz gates only. Found {inst.operation.name}"
            )

        if len(iop.params) != 1:
            raise ValueError("The Rz/Rzz gates should have one parameter.")

        if not isinstance(iop.params[0], ParameterExpression):
            raise ValueError("The Rzz gates should have one parameter.")

        if len(inst.qubits) == 1:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[0]))

        elif len(inst.qubits) == 2:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[1]))

        else:
            raise ValueError(
                "Instructions with more than 2 qubits cannot be converted to graph edges."
            )

        if edge in seen_edges:
            raise ValueError(f"Circuit contains multiple times the edge {edge}.")

        seen_edges.add(edge)
        seen_edges.add(edge[::-1])

        param_expression = copy.deepcopy(iop.params[0])
        param_expression = param_expression.assign(next(iter(param_expression.parameters)), 1)
        weight = float(param_expression) / 2.0
        edges.append((edge[0], edge[1], weight))

    graph.add_weighted_edges_from(edges)

    return graph


def load_graph(file_name: str) -> nx.Graph:
    """Load a graph from a json file."""
    with open(file_name, "r") as fin:
        data = json.load(fin)

    return dict_to_graph(data)


def solve_max_cut(cost_op: SparsePauliOp, energy: Optional[float] = None):
    """Solve the MaximumCut problem for the given cost op.

    This method allows us to benchmark performance. It requires CPLEX to solve to optimality
    the MaxCut problem for the given cost operator. In addition, if we provide the
    energy then the approximation ratio corresponding to this energy is computed.

    Args:
        cost_op: The cost operator that will be converted to a graph.
        energy: An energy to compare to the minimum and maximum cuts found by CPLEX. If
            this quantity is given then we will convert it to an approximation ratio.
    """
    graph = operator_to_graph(cost_op, pre_factor=-2)

    opt_problem = Maxcut(nx.adjacency_matrix(graph, nodelist=range(graph.order())))

    # Get the maximum cut value
    cplex_result_max = CplexOptimizer().solve(opt_problem.to_quadratic_program())
    max_cut = cplex_result_max.fval

    # Get the minimum cut value
    min_cut_problem = opt_problem.to_quadratic_program()
    min_cut_problem.objective._sense = ObjSense.MINIMIZE
    solcplex_result_min = CplexOptimizer().solve(min_cut_problem)
    min_cut = solcplex_result_min.fval

    approximation_ratio = None
    if energy is not None:
        sum_weights = sum(val[2].get("weight", 1.0) for val in graph.edges(data=True))

        cut_val = energy + 0.5 * sum_weights

        approximation_ratio = (cut_val - min_cut) / (max_cut - min_cut)

    return max_cut, min_cut, approximation_ratio


def make_swap_strategy(edges: List[Tuple[int]], num_qubits: int) -> SwapStrategy:
    """Create the SWAP strategy that implements the graph.

    Create a SWAP strategy for a line that reaches full connectivity and then simplify it.
    If the list of edges does not need full connectivity we remove the unneeded
    layers from the full connectivity strategy. A swap strategy has a distance_matrix `d`
    where entry `[i, j]` is the number of SWAP layers needed to implement a gate between
    qubits `(i, j)`. If all needed gates in the cost operator between the qubits
    satisfy `d[i, j]<d_max` then we can remove all swap layers from `d_max` and above. Indeed,
    these swap layers are not needed since all relevant gates will have been implemented.

    Args:
        edges (List[Tuple[int]]): List of edges, represented as tuple of ints.
        num_qubits (int): number of nodes of the graph.

    Returns:
        A line swap strategy truncated to implement the graph in as few layers as possible.
    """
    tentative_strat = SwapStrategy.from_line(range(num_qubits))

    # Optimize the swap strategy in case we do not need all the layers.
    max_layers = 0
    for edge in edges:
        # Ignore one-local terms
        if len(edge) == 1:
            continue

        n1, n2 = edge
        max_layers = max(max_layers, tentative_strat.distance_matrix[n1, n2])

    layers = [tentative_strat.swap_layer(idx) for idx in range(max_layers)]
    cmap = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
    swap_strat = SwapStrategy(coupling_map=cmap, swap_layers=layers)

    # Validate the SWAP strategy out of paranoia. This is important for the canonization
    # since we require that the indices of the gates are sorted `(i, i+1)`.
    for idx in range(len(swap_strat)):
        for swap in swap_strat.swap_layer(idx):
            if swap[0] + 1 != swap[1]:
                raise ValueError("The gates in the SWAP strategy must be order as (i, i+1).")

    return swap_strat
