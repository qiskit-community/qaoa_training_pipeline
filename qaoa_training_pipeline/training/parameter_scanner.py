#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to scan param2 and param1 in depth-one QAOA to get the optimal point."""

from time import time
import math
from typing import List, Optional, Tuple, Hashable
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.functions import (
    BaseAnglesFunction,
    IdentityFunction,
    FUNCTIONS,
)
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.extrema_location import Argmax, Argmin
from qaoa_training_pipeline.training.history_mixin import HistoryMixin
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class DepthOneScanTrainer(BaseTrainer, HistoryMixin):
    """Scan the param2 and param1 parameters of QAOA."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        energy_minimization: bool = False,
        qaoa_angles_function: Optional[BaseAnglesFunction] = None,
    ):
        """Initialize the class instance.

        Args:
            evaluator: The evaluator that computes the energy.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
            qaoa_angles_function: A function to convert optimization parameters into QAOA
                angles. By default, this is the identity function. Ideally, this argument is
                an instance of `BaseAnglesFunction` but we allow any callable here that maps
                optimization parameters to QAOA angles.
        """
        BaseTrainer.__init__(self, evaluator=evaluator, qaoa_angles_function=qaoa_angles_function)
        HistoryMixin.__init__(self)

        # Parameters that will be filled by the scanner.
        self._energies = None
        self._params2 = None
        self._params1 = None

        # This could be set in a subsequent PR by other methods, e.g., interpolation.
        # This is a callable that takes as input the 2D energies scan.
        self._energy_minimization = energy_minimization
        self._extrema_locator = Argmin() if energy_minimization else Argmax()

        self._opt_param1 = None
        self._opt_param2 = None

        # Default parameter range over which to scan.
        self._default_range = ((0, np.pi), (0, 2 * np.pi))

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return isinstance(self._extrema_locator, Argmin)

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        parameter_ranges: Optional[List[Tuple[float, float]]] = None,
        num_points: Optional[int] = 15,
    ) -> ParamResult:
        r"""Train the parameters by doing a 2D scan.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\param2 H_C)`.
            parameter_ranges:  The parameter ranges in param1 and param2 over which to scan. If
                this argument is not provided we default to `((np.pi), (0, 2 * np.pi))`.
            num_points: The number of points in the param2 and param1 ranges to take. This
                method will thus evaluate the energy `num_points**2` times.
        """
        self.reset_history()
        start = time()

        parameter_ranges = parameter_ranges or self._default_range

        self._energies = np.zeros((num_points, num_points), dtype=float)

        # By default params1 scans beta and params2 scans gamma
        self._params1 = np.linspace(parameter_ranges[0][0], parameter_ranges[0][1], num_points)
        self._params2 = np.linspace(parameter_ranges[1][0], parameter_ranges[1][1], num_points)

        self._opt_param2, self._opt_param1 = None, None

        for idx1, param1 in enumerate(self._params1):
            for idx2, param2 in enumerate(self._params2):
                estart = time()

                qaoa_angles = self._qaoa_angles_function([param1, param2])

                energy = self._evaluator.evaluate(
                    cost_op,
                    qaoa_angles,
                    mixer,
                    initial_state,
                    ansatz_circuit,
                )
                self._energies[idx1, idx2] = float(np.real(energy))

                self._energy_evaluation_time.append(time() - estart)
                self._energy_history.append(float(np.real(energy)))
                self._parameter_history.append([float(param1), float(param2)])

        min_idx, opt_energy = self._extrema_locator(self._energies)
        min_idxb, min_idxg = min_idx // num_points, min_idx % num_points
        opt_param1, opt_param2 = self._params1[min_idxb], self._params2[min_idxg]

        self._opt_param2 = opt_param2
        self._opt_param1 = opt_param1

        opt_result = ParamResult([opt_param1, opt_param2], time() - start, self, opt_energy)
        opt_result["num_points"] = num_points
        opt_result["parameter_ranges"] = parameter_ranges
        opt_result.add_history(self)

        return opt_result

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        xlabel: str = r"$\gamma$",
        ylabel: str = r"$\beta$",
    ):
        """Make a plot of the training.

        If giving the `axis` then the figure object must also be given. Otherwise,
        you get new objects.

        Args:
            axis: Axis on which to plot.
            fig: The figure object.
            xlabel: Label for the x-axis. This is needed if we are using a function to relate
                the scanned parameters to QAOA anagles.
            ylabel: Label for the y-axis. This is needed if we are using a function to relate
                the scanned parameters to QAOA anagles.
        """

        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        ggs, bbs = np.meshgrid(self._params2, self._params1)
        cset = axis.contourf(ggs, bbs, self._energies, levels=30)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.scatter([self._opt_param2], [self._opt_param1], s=10, marker="*", color="w")
        fig.colorbar(cset, ax=axis, label="Energy")

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "DepthOneScanTrainer":
        """Create an intance from a config."""

        evaluator_cls = EVALUATORS[config["evaluator"]]

        if "qaoa_angles_function" not in config:
            function = IdentityFunction()
        else:
            function_cls = FUNCTIONS[config["qaoa_angles_function"]]
            function = function_cls.from_config(config["qaoa_angles_function_init"])

        return cls(
            evaluator_cls.from_config(config["evaluator_init"]),
            config.get("energy_minimization", None),
            function,
        )

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse the trainig arguments.

        These are given in the form:
        num_points:val:parameter_ranges:low/high/low/high_...
        For instance training with 20 points from 0 to 2pi is given as
        num_points:20:parameter_ranges:0/6.283185/0/6.283185.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key == "num_points":
                train_kwargs[key] = int(val)
            elif key == "parameter_ranges":
                val_ = self.extract_list(val, dtype=float)
                train_kwargs[key] = [
                    (float(val_[idx]), float(val_[idx + 1])) for idx in range(0, len(val_), 2)
                ]
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs


class DepthOneGammaScanTrainer(DepthOneScanTrainer):
    """Scan Gamma values and compute the optimal beta value analytically for each gamma as per https://arxiv.org/pdf/2501.16419 -
    "Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models".
    The gurentee for optimality of the beta parameter here is dependant on using the standard mixer Hm = sum X_j.
    For other mixer hamiltonian the value of the beta computed here might be non optimal.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        energy_minimization: bool = False,
    ):
        """Initialize the class instance.

        Args:
            evaluator: The evaluator that computes the energy.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        super().__init__(
            evaluator=evaluator,
            energy_minimization=energy_minimization,
        )

        # Override parent initialization sice we are only scanning values for gamma and not beta,
        # and put it in a tuple for consistency with parent API
        self._default_range = ((0, 2 * np.pi),)

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        parameter_ranges: Optional[List[Tuple[float, float]]] = None,
        num_points: Optional[int] = 15,
    ) -> ParamResult:
        r"""Train the parameters by doing a 1D scan and setting beta to the analytical
        optimal solution per gamma.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\param2 H_C)`.
            parameter_ranges:  The parameter range for gamma over which to scan. If
                this argument is not provided we default to `(0, 2 * np.pi)`.
            num_points: The number of points in the gamma range to take. This
                method will thus evaluate the energy `num_points**2` times.
        """
        self.reset_history()
        start = time()

        parameter_ranges = parameter_ranges or self._default_range

        self._energies = np.zeros(num_points, dtype=float)

        # By default params1 keep the value for beta and params2 scans gamma
        self._params1 = np.zeros(num_points)
        self._params2 = np.linspace(parameter_ranges[0][0], parameter_ranges[0][1], num_points)

        self._opt_param2, self._opt_param1 = None, None
        graph = operator_to_graph(cost_op)

        for idx, param2 in enumerate(self._params2):
            estart = time()
            param1 = self._beta_star_for_gamma(graph, param2)
            self._params1[idx] = param1

            qaoa_angles = [param1, param2]
            energy = self._evaluator.evaluate(
                cost_op,
                qaoa_angles,
                initial_state,
                ansatz_circuit,
            )
            self._energies[idx] = float(np.real(energy))

            self._energy_evaluation_time.append(time() - estart)
            self._energy_history.append(float(np.real(energy)))
            self._parameter_history.append([float(param1), float(param2)])

        # update the last optimal beta for the last gamma
        param2 = self._params2[-1]
        param1 = self._beta_star_for_gamma(graph, param2)
        self._params1[-1] = param1

        min_idx, opt_energy = self._extrema_locator(self._energies)
        opt_param1, opt_param2 = self._params1[min_idx], self._params2[min_idx]

        self._opt_param2 = opt_param2
        self._opt_param1 = opt_param1

        opt_result = ParamResult([opt_param1, opt_param2], time() - start, self, opt_energy)
        opt_result["num_points"] = num_points
        opt_result["parameter_ranges"] = parameter_ranges
        opt_result.add_history(self)

        return opt_result

    def _beta_star_for_gamma(self, graph: nx.Graph, gamma: float, weight_attr: str = "weight") -> float:
        """
        Compute beta*(gamma) per Theorem 5 (https://arxiv.org/pdf/2501.16419)
        (field-free Ising, QAOA p=1) for a given gamma and graph G.

        Returns beta* in [0, pi].

        Formula: beta* = (1/4) * (atan2(2A(gamma), B(gamma)) + pi).
        """
        a_matrix, b_matrix = self._compute_a_b_matrices_for_gamma(graph, gamma, weight_attr=weight_attr)

        # Handle the degenerate case: if both A and B are ~0, the landscape is flat in beta.
        if abs(a_matrix) < 1e-15 and abs(b_matrix) < 1e-15:
            return 0.0  # any beta works; choose 0 for determinism

        if self._energy_minimization:
            beta = 0.25 * (math.atan2(2.0 * a_matrix, b_matrix) + math.pi)
        else:
            beta = 0.25 * math.atan2(2.0 * a_matrix, b_matrix)

        return beta

    def _edge_weight(
        self, graph: nx.Graph, u: Hashable, v: Hashable, weight_attr: str = "weight"
    ) -> float:
        """Fetch edge weight (u,v) J_uv. default weight is 1.0."""
        data = graph.get_edge_data(u, v, default={})
        return float(data.get(weight_attr, 1.0))

    def _prod_cos_edges_from_node(
        self, graph: nx.Graph, node: Hashable, nbrs: set, gamma: float, weight_attr: str
    ) -> float:
        """
        Compute Prod_{w ∈ nbrs} cos( 2 * J_{node,w} * gamma ).
        Empty product returns 1.0.
        """
        if not nbrs:
            return 1.0
        val = 1.0
        for w in nbrs:
            weight_nw = self._edge_weight(graph, node, w, weight_attr)
            val *= math.cos(2.0 * weight_nw * gamma)
        return val

    def _prod_cos_triangle_terms(
        self,
        graph: nx.Graph,
        u: Hashable,
        v: Hashable,
        mutual_nbrs: set,
        gamma: float,
        weight_attr: str,
        plus: bool,
    ) -> float:
        """
        Compute Ptod_{f ∈ mutual_nbrs} cos( 2*J_{u,f}*gamma + 2*J_{v,f}*gamma ).
        Empty product returns 1.0.
        """
        if not mutual_nbrs:
            return 1.0
        val = 1.0
        for f_vertex in mutual_nbrs:
            weight_u_f = self._edge_weight(graph, u, f_vertex, weight_attr)
            weight_v_f = self._edge_weight(graph, v, f_vertex, weight_attr)
            if plus:
                angle = 2.0 * weight_u_f * gamma + 2.0 * weight_v_f * gamma
            else:
                angle = 2.0 * weight_u_f * gamma - 2.0 * weight_v_f * gamma
            val *= math.cos(angle)
        return val

    def _compute_a_b_matrices_for_gamma(
        self, graph: nx.Graph, gamma: float, weight_attr: str = "weight"
    ) -> tuple[float, float]:
        """
        Compute A(gamma) and B(gamma) per Theorem 5 for an Ising model without fields.

        """
        if not isinstance(graph, nx.Graph) or graph.is_directed():
            raise ValueError("Provide an undirected NetworkX Graph (nx.Graph).")

        a_matrix = 0.0
        b_matrix = 0.0

        # Iterate each undirected edge once
        for u_vertex, v_vertex in graph.edges():
            weight_uv = self._edge_weight(graph, u_vertex, v_vertex, weight_attr)
            g_uv = 2.0 * weight_uv * gamma  # gamma'_{uv} = 2 * J_uv * gamma

            nbrs_u_minus_v = set(graph.neighbors(u_vertex)) - {v_vertex}
            nbrs_v_minus_u = set(graph.neighbors(v_vertex)) - {u_vertex}
            mutual_nbrs_uv = (
                nbrs_u_minus_v & nbrs_v_minus_u
            )  # mutual neighbors of v and u, i.e. nodes that create a traingle with u and v in the graph

            # A(gamma) term
            prod_v = self._prod_cos_edges_from_node(graph, v_vertex, nbrs_v_minus_u, gamma, weight_attr)
            prod_u = self._prod_cos_edges_from_node(graph, u_vertex, nbrs_u_minus_v, gamma, weight_attr)
            a_matrix += (weight_uv / 2.0) * math.sin(g_uv) * (prod_v + prod_u)

            # B(gamma) term
            nbrs_v_minus_mutual_to_u = nbrs_v_minus_u - mutual_nbrs_uv
            nbrs_u_minus_mutul_to_v = nbrs_u_minus_v - mutual_nbrs_uv

            prod_v_without_mutual = self._prod_cos_edges_from_node(
                graph, v_vertex, nbrs_v_minus_mutual_to_u, gamma, weight_attr
            )
            prod_u_without_mutual = self._prod_cos_edges_from_node(
                graph, u_vertex, nbrs_u_minus_mutul_to_v, gamma, weight_attr
            )

            tri_plus = self._prod_cos_triangle_terms(
                graph, u_vertex, v_vertex, mutual_nbrs_uv, gamma, weight_attr, plus=True
            )
            tri_minus = self._prod_cos_triangle_terms(
                graph, u_vertex, v_vertex, mutual_nbrs_uv, gamma, weight_attr, plus=False
            )

            b_matrix += (
                (weight_uv / 2.0)
                * (prod_v_without_mutual * prod_u_without_mutual)
                * (tri_plus - tri_minus)
            )

        return a_matrix, b_matrix

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        xlabel: str = r"$\gamma$",
        ylabel: str = r"$\beta$",
    ):
        """Make a plot of the training.

        If giving the `axis` then the figure object must also be given. Otherwise,
        you get new objects.

        Args:
            axis: Axis on which to plot.
            fig: The figure object.
            xlabel: Label for the x-axis. This is needed if we are using a function to relate
                the scanned parameters to QAOA anagles.
            ylabel: Label for the y-axis. This is needed if we are using a function to relate
                the scanned parameters to QAOA anagles.
        """

        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        sc = axis.scatter(
            self._params2, self._params1, c=self._energies, cmap="viridis", s=40, edgecolor="none"
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        cbar = plt.colorbar(sc)
        cbar.set_label("Energy")
        plt.tight_layout()

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "DepthOneScanTrainer":
        """Create an intance from a config."""

        evaluator_cls = EVALUATORS[config["evaluator"]]

        return cls(
            evaluator_cls.from_config(config["evaluator_init"]),
            config.get("energy_minimization", None),
        )
