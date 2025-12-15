#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Solve the SWAP gate insertion initial mapping problem using SAT."""


from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from threading import Timer
from time import time
from typing import Optional

import networkx as nx
import numpy as np
from networkx.classes.reportviews import DegreeView
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qaoa_training_pipeline.pre_processing.base_processing import BasePreprocessor
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph, graph_to_dict


@dataclass
class SATResult:
    """A data class to hold the result of a SAT solver."""

    satisfiable: bool  # Satisfiable is True if the SAT model could be solved in a given time.
    solution: dict  # The solution to the SAT problem if it is satisfiable.
    mapping: list  # The mapping of nodes in the pattern graph to nodes in the target graph.
    elapsed_time: float  # The time it took to solve the SAT model.


class SATMapper(BasePreprocessor):
    r"""Permute nodes in a graph to best match a line swap strategy.

    The layout is found with a binary search over the layers :math:`l` of a line swap strategy.
    At each considered layer a subgraph isomorphism problem formulated as a SAT is solved by a
    SAT solver. Each instance is whether it is possible to embed the program graph :math:`P`
    into the effective connectivity graph :math:`C_l` that is achieved by applying :math:`l`
    layers of the swap strategy to the coupling map :math:`C_0` of the backend. Since solving
    SAT problems can be hard, a ``time_out`` fixes the maximum time allotted to the SAT solver
    for each instance. If this time is exceeded the considered problem is deemed unsatisfiable
    and the binary search proceeds to the next number of swap layers :math:``l``.

    Warning: for large instances the SATMapper can take a lot of pre-processing time.

    The SAT mapper is based on this paper https://arxiv.org/abs/2212.05666.
    """

    def __init__(self, timeout: int = 60):
        """Initialize the SATMapping.

        Args:
            timeout: The allowed time in seconds for each iteration of the SAT solver. This
                variable defaults to 60 seconds.
        """
        super().__init__()
        self.timeout = timeout

        # Minimum number of swap layers needed.
        self.min_k = None

        # `{k: v}` where node `k` in the original graph maps to node `v`.
        self.edge_map = None

    def __call__(self, input_data: dict):
        """Call the SATMapper.

        This function will return the SAT mapped graph in the same data format as the
        input. This allows the pre-processors to seamlessly integrate in the data
        processing steps of `train.py`.

        Args:
            input_data: The input data to be preprocessed. This data is directly loaded
                from the Json file, typically given to `train.py`. The `__call__` method
                expects that this input dictionary will convert to a graph through the
                method `dict_to_graph` in `graph_utils.py`.
        """
        start = time()
        graph = dict_to_graph(input_data)

        sat_graph = self.remap_graph_with_sat(graph)

        self.duration = time() - start

        return graph_to_dict(sat_graph)

    @classmethod
    def from_str(cls, input_str: str) -> "SATMapper":
        """Initialize the SATMapper from a string.

        Args:
            input_str: The only required input is an int which represents the timeout per
                iteration of the SATMapper. The input is a string so that this pre-processor
                does not constrain the input data type of `from_str` of future pre-processors.
        """
        return cls(int(input_str))

    def to_config(self) -> dict:
        """Serialize the SATMapper to a config dictionary."""
        config = super().to_config()
        config["timeout"] = self.timeout
        config["min_k"] = self.min_k
        config["edge_map"] = self.edge_map

        return config

    # pylint: disable=too-many-locals
    def find_initial_mappings(
        self,
        program_graph: nx.Graph,
        min_layers: Optional[int] = None,
        max_layers: Optional[int] = None,
    ) -> dict[int, SATResult]:
        r"""Find an initial mapping for a line swap strategy.

        Args:
            program_graph (nx.Graph): The program graph with commuting gates, where
                each edge represents a two-qubit gate.
            min_layers (int): The minimum number of swap layers to consider. Defaults to
                the maximum degree of the program graph - 2.
            max_layers (int): The maximum number of swap layers to consider. Defaults to
                the number of qubits in the swap strategy - 2.

        Returns:
            dict[int, SATResult]: A dictionary containing the results of the SAT solver for
                each number of swap layers.
        """
        swap_strategy = SwapStrategy.from_line(list(range(program_graph.order())))

        num_nodes_g1 = len(program_graph.nodes)
        num_nodes_g2 = swap_strategy.distance_matrix.shape[0]
        if num_nodes_g1 > num_nodes_g2:
            return {1: SATResult(False, {}, [], 0)}
        if min_layers is None:
            # use the maximum degree of the program graph - 2 as the lower bound.
            assert isinstance(program_graph.degree, DegreeView)
            min_layers = max((d for _, d in program_graph.degree)) - 2
        if max_layers is None:
            max_layers = num_nodes_g2 - 1

        variable_pool = IDPool(start_from=1)
        variables = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_g2)]
                for i in range(num_nodes_g1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(variables)}
        binary_search_results = {}

        def interrupt(solver):
            # This function is called to interrupt the solver when the timeout is reached.
            solver.interrupt()

        # Make a cnf for the one-to-one mapping constraint
        cnf1 = []
        for i in range(num_nodes_g1):
            clause = variables[i, :].tolist()
            cnf1.append(clause)
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])
        for j in range(num_nodes_g2):
            clause = variables[:, j].tolist()
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])

        # Perform a binary search over the number of swap layers to find the minimum
        # number of swap layers that satisfies the subgraph isomorphism problem.
        assert max_layers, "max_layers must be defined before calling find_initial_mappings()"
        while min_layers < max_layers:
            num_layers = (min_layers + max_layers) // 2

            # Create the connectivity matrix. Note that if the swap strategy cannot reach
            # full connectivity then its distance matrix will have entries with -1. These
            # entries must be treated as False.
            d_matrix = swap_strategy.distance_matrix
            connectivity_matrix = ((-1 < d_matrix) & (d_matrix <= num_layers)).astype(int)
            # Make a cnf for the adjacency constraint
            cnf2 = []
            for e_0, e_1 in program_graph.edges:
                clause_matrix = np.multiply(connectivity_matrix, variables[e_1, :])
                clause = np.concatenate(
                    (
                        [[-variables[e_0, i]] for i in range(num_nodes_g2)],
                        clause_matrix,
                    ),
                    axis=1,
                )
                # Remove 0s from each clause
                cnf2.extend([c[c != 0].tolist() for c in clause])

            cnf = CNF(from_clauses=cnf1 + cnf2)

            with Solver(bootstrap_with=cnf, use_timer=True) as solver:
                # Solve the SAT problem with a timeout.
                # Timer is used to interrupt the solver when the timeout is reached.
                timer = Timer(self.timeout, interrupt, [solver])
                timer.start()
                status = solver.solve_limited(expect_interrupt=True)
                timer.cancel()
                # Get the solution and the elapsed time.
                sol = solver.get_model()
                e_time = solver.time()

                assert sol, "solver from get_model() was undefined"
                assert e_time, "solver ran without defining e_time"
                assert isinstance(status, bool), "solver status returned with non-boolean value"
                sol = {i: x for i, x in enumerate(sol)}
                mapping = []
                if status:
                    # If the SAT problem is satisfiable, convert the solution to a mapping.
                    mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    max_layers = num_layers
                else:
                    # If the SAT problem is unsatisfiable, return the last satisfiable solution.
                    min_layers = num_layers + 1
                binary_search_results[num_layers] = SATResult(status, sol, mapping, e_time)

        return binary_search_results

    def remap_graph_with_sat(self, graph: nx.Graph):
        """Apply the SAT mapping.

        Args:
            graph: The graph to remap.

        Returns:
            The remapped graph. The edge map, and the number of layers of the swap strategy
            that was used to find the initial mapping are both stored locally as internal
            variables to the SATMapper. If no solution is found then an error is raised.
        """
        num_nodes = len(graph.nodes)
        results = self.find_initial_mappings(graph, 0, num_nodes - 1)
        solutions = [k for k, v in results.items() if v.satisfiable]
        if len(solutions):
            min_k = min(solutions)
            edge_map = dict(results[min_k].mapping)
            remapped_graph = nx.relabel_nodes(graph, edge_map)

            self.min_k = min_k
            self.edge_map = edge_map

            return remapped_graph
        else:
            raise ValueError("No solution found.")
