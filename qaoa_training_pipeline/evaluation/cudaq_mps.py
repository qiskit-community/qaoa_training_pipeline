#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""CUDA-Q MPS evaluator

This module provides an NVIDIA CUDA-Q backend for evaluating QAOA energy on GPU.
CUDA-Q is imported lazily when the evaluator is instantiated, so importing the
qaoa_training_pipeline package does not require CUDA-Q to be installed.

"""

import os
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator

CUDAQ_MPS_TARGET = "tensornet-mps"
CUDAQ_MPS_ENV_VARS = {
    "max_bond_dim": "CUDAQ_MPS_MAX_BOND",
    "abs_cutoff": "CUDAQ_MPS_ABS_CUTOFF",
    "relative_cutoff": "CUDAQ_MPS_RELATIVE_CUTOFF",
    "svd_algo": "CUDAQ_MPS_SVD_ALGO",
}


@dataclass(frozen=True)
class _ZZTerm:
    """A weighted two-local ZZ term in qubit-index order."""

    u: int
    v: int
    coefficient: float


def _require_cudaq():
    """Import CUDA-Q and raise a clear optional-dependency error if unavailable."""

    try:
        import cudaq  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "CudaQMPSBenchmarkEvaluator requires the optional CUDA-Q package. "
            "Install it with `pip install cudaq` on a supported CUDA-Q platform."
        ) from exc

    return cudaq


def _cudaq_qaoa_kernel(cudaq):
    """Build the CUDA-Q QAOA kernel after CUDA-Q has been imported."""

    # pylint: disable=undefined-variable
    @cudaq.kernel
    def qaoa_problem(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
        """Apply exp(-i alpha Z_i Z_j) via CX-RZ-CX."""

        x.ctrl(qubit_0, qubit_1)
        rz(2.0 * alpha, qubit_1)
        x.ctrl(qubit_0, qubit_1)

    @cudaq.kernel
    def qaoa_kernel(
        qubit_count: int,
        layer_count: int,
        edge_count: int,
        edges_src: List[int],
        edges_tgt: List[int],
        edge_coefficients: List[float],
        params: List[float],
    ):
        """Apply standard QAOA with params ordered as [betas..., gammas...]."""

        qreg = cudaq.qvector(qubit_count)
        h(qreg)

        for layer in range(layer_count):
            gamma = params[layer_count + layer]
            for edge_idx in range(edge_count):
                qaoa_problem(
                    qreg[edges_src[edge_idx]],
                    qreg[edges_tgt[edge_idx]],
                    gamma * edge_coefficients[edge_idx],
                )

            beta = params[layer]
            for qubit_idx in range(qubit_count):
                rx(2.0 * beta, qreg[qubit_idx])

    return qaoa_kernel


class CudaQMPSBenchmarkEvaluator(BaseEvaluator):
    """Evaluator that computes energies for QAOA with CUDA-Q's ``tensornet-mps`` target."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        graph: nx.Graph | None = None,
        list_of_edges: Iterable | None = None,
        n_qubits: int | None = None,
        max_bond_dim: int = 64,
        abs_cutoff: float = 1.0e-5,
        relative_cutoff: float = 1.0e-5,
        precision: str = "fp64",
        auto_setup_target: bool = True,
        require_gpu: bool = True,
        svd_algo: str | None = None,
    ) -> None:
        """Initialize the CUDA-Q evaluator.

            Args:
                graph: Optional NetworkX graph. When provided, it is converted to the
                    repository MaxCut Hamiltonian convention ``-0.5 * weight * Z_i Z_j``.
                list_of_edges: Optional edge list. Entries may be ``(u, v)``, ``(u, v, weight)``,
                    or ``((u, v), weight)`` and are converted with the same MaxCut convention.
                n_qubits: Number of qubits. Required when the graph has isolated highest-index
                    vertices that do not appear in ``list_of_edges``.
                max_bond_dim: Value for ``CUDAQ_MPS_MAX_BOND``.
                abs_cutoff: Value for ``CUDAQ_MPS_ABS_CUTOFF``.
                relative_cutoff: Value for ``CUDAQ_MPS_RELATIVE_CUTOFF``.
                precision: ``"fp64"`` by default; use ``"fp32"`` to call
                    ``cudaq.set_target("tensornet-mps", option="fp32")``.
                auto_setup_target: If true, :meth:`evaluate` calls :meth:`setup_target` before
                    observing the energy.
                require_gpu: If true, fail early when CUDA-Q reports zero available GPUs.

        The evaluator intentionally supports only weighted two-local diagonal ZZ Hamiltonians
        for this first CUDA-Q MaxCut benchmark backend. Custom mixers, custom initial states,
        custom ansatz circuits, hyperedges, one-local terms, identity offsets, and non-Z
        Paulis are rejected.
        """

        super().__init__()
        self._cudaq = _require_cudaq()

        if graph is not None and list_of_edges is not None:
            raise ValueError("Provide either graph or list_of_edges, not both.")

        if precision not in {"fp64", "fp32"}:
            raise ValueError("precision must be either 'fp64' or 'fp32'.")

        if max_bond_dim <= 0:
            raise ValueError("max_bond_dim must be positive.")

        if abs_cutoff < 0.0:
            raise ValueError("abs_cutoff must be non-negative.")

        if relative_cutoff < 0.0:
            raise ValueError("relative_cutoff must be non-negative.")

        if svd_algo is not None:
            svd_algo = svd_algo.upper()
            if svd_algo not in {"GESVD", "GESVDJ", "GESVDP", "GESVDR"}:
                raise ValueError("svd_algo must be one of GESVD, GESVDJ, GESVDP, or GESVDR.")
            self._svd_algo = svd_algo
        else:
            self._svd_algo = "GESVD"

        self._max_bond_dim = max_bond_dim
        self._abs_cutoff = abs_cutoff
        self._relative_cutoff = relative_cutoff
        self._precision = precision
        self._auto_setup_target = auto_setup_target
        self._require_gpu = require_gpu
        self._target_is_configured = False
        self._kernel = None
        self._results_last_iteration = {}
        self._hamiltonian = None
        self._edges_src = None
        self._edges_tgt = None
        self._edge_coefficients = None
        self._terms = None

        self._cost_op: SparsePauliOp | None = None
        if graph is not None:
            self._cost_op = self._cost_operator_from_graph(graph, n_qubits)
        elif list_of_edges is not None:
            self._cost_op = self._cost_operator_from_edges(list_of_edges, n_qubits)

    def setup_target(self, force: bool = False) -> None:
        """Configure CUDA-Q's process-wide ``tensornet-mps`` target.

        This method sets ``CUDAQ_MPS_MAX_BOND``, ``CUDAQ_MPS_ABS_CUTOFF``, and
        ``CUDAQ_MPS_RELATIVE_CUTOFF`` before calling ``cudaq.set_target``. CUDA-Q
        targets are global to the Python process, so calling this method can affect
        other CUDA-Q code running in the same process.
        """

        if self._target_is_configured and not force:
            return

        self._set_mps_environment()
        self._validate_cudaq_runtime()

        try:
            if self._precision == "fp32":
                self._cudaq.set_target(CUDAQ_MPS_TARGET, option="fp32")
            else:
                self._cudaq.set_target(CUDAQ_MPS_TARGET)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                f"Failed to initialize CUDA-Q target '{CUDAQ_MPS_TARGET}'. "
                "This backend requires CUDA-Q, an NVIDIA GPU, and CUDA runtime libraries."
            ) from exc

        self._target_is_configured = True

    # pylint: disable=too-many-positional-arguments,arguments-differ
    def evaluate(
        self,
        cost_op: SparsePauliOp | Sequence[float] | None = None,
        params: list[float] | Sequence[float] | None = None,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate a QAOA MaxCut energy.

        Supports both the repository evaluator API, ``evaluate(cost_op, params)``,
        and graph-initialized use as ``evaluate(betas, gammas)``.
        """

        active_cost_op, qaoa_params = self._resolve_evaluate_arguments(cost_op, params)
        self._validate_qaoa_features(mixer, initial_state, ansatz_circuit)

        if len(qaoa_params) % 2 != 0:
            raise KeyError("Number of parameters must be an even integer")

        layer_count = len(qaoa_params) // 2

        if self._auto_setup_target:
            self.setup_target()

        if self._hamiltonian is None:
            self._terms = self._terms_from_cost_operator(active_cost_op)
            self._hamiltonian = self._cudaq_hamiltonian(self._terms)
            self._edges_src = [term.u for term in self._terms]
            self._edges_tgt = [term.v for term in self._terms]
            self._edge_coefficients = [term.coefficient for term in self._terms]
        if self._kernel is None:
            self._kernel = self._make_graph_qaoa_kernel(
                self._cudaq, active_cost_op.num_qubits, self._terms
            )

        result = self._cudaq.observe(
            self._kernel,
            self._hamiltonian,
            layer_count,
            list(qaoa_params),
        )
        energy = float(np.real(result.expectation()))
        return energy

    def to_config(self) -> dict:
        """Json serializable config to keep track of how results are generated."""

        config = super().to_config()
        config.update(
            {
                "max_bond_dim": self._max_bond_dim,
                "abs_cutoff": self._abs_cutoff,
                "relative_cutoff": self._relative_cutoff,
                "precision": self._precision,
                "auto_setup_target": self._auto_setup_target,
                "require_gpu": self._require_gpu,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "CudaQMPSBenchmarkEvaluator":
        """Initialize the evaluator from a configuration dictionary."""

        return cls(**config)

    @classmethod
    def parse_init_kwargs(cls, init_kwargs: str | None = None) -> dict:
        """Parse ``max_bond_abs_cutoff_relative_cutoff_precision`` strings."""

        if init_kwargs is None:
            return dict()

        init_args = init_kwargs.split("_")
        config = {}

        if len(init_args) > 0 and init_args[0].lower() != "none":
            config["max_bond_dim"] = int(init_args[0])
        if len(init_args) > 1 and init_args[1].lower() != "none":
            config["abs_cutoff"] = float(init_args[1])
        if len(init_args) > 2 and init_args[2].lower() != "none":
            config["relative_cutoff"] = float(init_args[2])
        if len(init_args) > 3 and init_args[3].lower() != "none":
            config["precision"] = init_args[3]

        return config

    @staticmethod
    def _cost_operator_from_graph(graph: nx.Graph, n_qubits: int | None) -> SparsePauliOp:
        """Convert a graph to the repository MaxCut cost Hamiltonian."""

        graph_copy = nx.Graph()
        graph_copy.add_nodes_from(graph.nodes)
        graph_copy.add_edges_from(graph.edges(data=True))

        if graph_copy.number_of_edges() == 0:
            raise ValueError("graph must contain at least one edge for the CUDA-Q benchmark.")

        if n_qubits is None:
            if any(not isinstance(node, Integral) for node in graph_copy.nodes):
                raise ValueError("CUDA-Q benchmark graph nodes must be integer qubit indices.")
            n_qubits = max(graph_copy.nodes) + 1

        CudaQMPSBenchmarkEvaluator._validate_node_labels(graph_copy.nodes, n_qubits)
        graph_copy.add_nodes_from(range(n_qubits))

        return graph_to_operator(graph_copy, pre_factor=-0.5)

    @staticmethod
    def _cost_operator_from_edges(
        list_of_edges: Iterable,
        n_qubits: int | None,
    ) -> SparsePauliOp:
        """Convert supported edge-list formats to the MaxCut cost Hamiltonian."""

        graph = nx.Graph()
        max_node = -1

        for edge in list_of_edges:
            u, v, weight = CudaQMPSBenchmarkEvaluator._normalize_edge(edge)
            graph.add_edge(u, v, weight=weight)
            max_node = max(max_node, u, v)

        if max_node < 0:
            raise ValueError("list_of_edges must contain at least one edge.")

        if n_qubits is None:
            n_qubits = max_node + 1

        CudaQMPSBenchmarkEvaluator._validate_node_labels(graph.nodes, n_qubits)
        graph.add_nodes_from(range(n_qubits))

        return graph_to_operator(graph, pre_factor=-0.5)

    @staticmethod
    def _normalize_edge(edge) -> tuple[int, int, float]:
        """Normalize ``(u, v)``, ``(u, v, w)``, or ``((u, v), w)`` edges."""

        if len(edge) == 3:
            u, v, weight = edge
            if isinstance(weight, dict):
                weight = weight.get("weight", 1.0)
        elif len(edge) == 2:
            first, second = edge
            if isinstance(first, (tuple, list)) and len(first) == 2:
                u, v = first
                weight = second
            else:
                u, v = first, second
                weight = 1.0
        else:
            raise ValueError(f"Unsupported edge format: {edge!r}")

        if u == v:
            raise NotImplementedError("Self-edges and one-local MaxCut terms are not supported.")

        return int(u), int(v), float(weight)

    @staticmethod
    def _validate_node_labels(nodes: Iterable[int], n_qubits: int) -> None:
        """Ensure graph labels can be used directly as qubit indices."""

        for node in nodes:
            if not isinstance(node, Integral):
                raise ValueError("CUDA-Q benchmark graph nodes must be integer qubit indices.")
            if node < 0 or node >= n_qubits:
                raise ValueError(
                    f"Node index {node} is outside the requested {n_qubits}-qubit register."
                )

    @staticmethod
    def _terms_from_cost_operator(cost_op: SparsePauliOp) -> list[_ZZTerm]:
        """Extract weighted two-local ZZ terms from a SparsePauliOp."""

        terms = []
        for pauli_str, coefficient in cost_op.to_list():
            coefficient = complex(coefficient)
            if abs(coefficient.imag) > 1.0e-12:
                raise NotImplementedError("Complex Hamiltonian coefficients are not supported.")

            non_identity = [idx for idx, char in enumerate(pauli_str[::-1]) if char != "I"]
            if any(char not in {"I", "Z"} for char in pauli_str):
                raise NotImplementedError(
                    "CudaQMPSBenchmarkEvaluator supports only diagonal Z/ZZ MaxCut terms."
                )
            if len(non_identity) == 0:
                if abs(coefficient.real) > 1.0e-12:
                    raise NotImplementedError("Constant identity offsets are not supported.")
                continue
            if len(non_identity) != 2:
                raise NotImplementedError(
                    "CudaQMPSBenchmarkEvaluator supports only weighted two-local ZZ MaxCut terms."
                )

            terms.append(
                _ZZTerm(
                    u=non_identity[0],
                    v=non_identity[1],
                    coefficient=float(coefficient.real),
                )
            )

        if not terms:
            raise ValueError("The CUDA-Q MPS benchmark requires at least one ZZ term.")

        return terms

    def _resolve_evaluate_arguments(
        self,
        cost_op: SparsePauliOp | Sequence[float] | None,
        params: list[float] | Sequence[float] | None,
    ) -> tuple[SparsePauliOp, list[float]]:
        """Handle both ``evaluate(cost_op, params)`` and ``evaluate(betas, gammas)``."""

        if isinstance(cost_op, SparsePauliOp):
            if params is None:
                raise ValueError("params must be provided when cost_op is provided.")
            return cost_op, [float(value) for value in params]

        if cost_op is None:
            if self._cost_op is None:
                raise ValueError(
                    "No cost operator is available. Provide cost_op to evaluate() or "
                    "initialize the evaluator with graph/list_of_edges."
                )
            if params is None:
                raise ValueError("params must be provided.")
            return self._cost_op, [float(value) for value in params]

        if params is None:
            raise ValueError("gammas must be provided when evaluate is called with betas.")

        if self._cost_op is None:
            raise ValueError(
                "evaluate(betas, gammas) requires graph or list_of_edges at initialization."
            )

        betas = [float(value) for value in cost_op]
        gammas = [float(value) for value in params]
        if len(betas) != len(gammas):
            raise ValueError("betas and gammas must have the same length.")

        return self._cost_op, betas + gammas

    @staticmethod
    def _validate_qaoa_features(
        mixer: QuantumCircuit | None,
        initial_state: QuantumCircuit | None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None,
    ) -> None:
        """Reject features outside the first CUDA-Q benchmark scope."""

        if mixer is not None:
            raise NotImplementedError("Custom mixers are not supported; use the default X mixer.")
        if initial_state is not None:
            raise NotImplementedError("Custom initial states are not supported.")
        if ansatz_circuit is not None:
            raise NotImplementedError("Custom ansatz circuits are not supported.")

    def _cudaq_hamiltonian(self, terms: Sequence[_ZZTerm]):
        """Build a CUDA-Q spin Hamiltonian from the weighted ZZ terms."""

        spin = self._cudaq.spin
        hamiltonian = None

        for term in terms:
            term_op = term.coefficient * spin.z(term.u) * spin.z(term.v)
            hamiltonian = term_op if hamiltonian is None else hamiltonian + term_op

        return hamiltonian

    def _set_mps_environment(self) -> None:
        """Set CUDA-Q MPS environment variables for this evaluator."""

        for _, env_var in CUDAQ_MPS_ENV_VARS.items():
            os.environ[env_var] = self._mps_environment()[env_var]

    def _mps_environment(self) -> dict[str, str]:
        """Return the MPS target configuration as environment strings."""
        env = {
            "CUDAQ_MPS_MAX_BOND": str(self._max_bond_dim),
            "CUDAQ_MPS_ABS_CUTOFF": str(self._abs_cutoff),
            "CUDAQ_MPS_RELATIVE_CUTOFF": str(self._relative_cutoff),
        }

        if self._svd_algo is not None:
            env["CUDAQ_MPS_SVD_ALGO"] = self._svd_algo

        return env

    def _validate_cudaq_runtime(self) -> None:
        """Check target and GPU availability when CUDA-Q exposes the probes."""

        if hasattr(self._cudaq, "has_target") and not self._cudaq.has_target(CUDAQ_MPS_TARGET):
            raise RuntimeError(
                f"CUDA-Q target '{CUDAQ_MPS_TARGET}' is not available in this installation."
            )

        if self._require_gpu and hasattr(self._cudaq, "num_available_gpus"):
            try:
                gpu_count = self._cudaq.num_available_gpus()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise RuntimeError("CUDA-Q could not query the number of available GPUs.") from exc

            if gpu_count <= 0:
                raise RuntimeError(
                    f"CUDA-Q target '{CUDAQ_MPS_TARGET}' requires an NVIDIA GPU, "
                    "but CUDA-Q reports zero available GPUs."
                )

    @staticmethod
    def _make_graph_qaoa_kernel(
        cudaq,
        n_qubits: int,
        terms: list[_ZZTerm],
    ):
        """Create a CUDA-Q kernel specialized to a fixed graph.

        The graph data is captured as primitive Python lists because CUDA-Q
        kernel capture does not support tuple-of-tuples.
        """

        edges_src = [int(term.u) for term in terms]
        edges_tgt = [int(term.v) for term in terms]
        edge_coefficients = [float(term.coefficient) for term in terms]
        edge_count = len(edges_src)

        @cudaq.kernel
        def graph_qaoa_kernel(layer_count: int, params: list[float]):
            qreg = cudaq.qvector(n_qubits)
            h(qreg)

            for layer in range(layer_count):
                gamma = params[layer_count + layer]

                for edge_idx in range(edge_count):
                    u = edges_src[edge_idx]
                    v = edges_tgt[edge_idx]
                    coeff = edge_coefficients[edge_idx]

                    x.ctrl(qreg[u], qreg[v])
                    rz(2.0 * gamma * coeff, qreg[v])
                    x.ctrl(qreg[u], qreg[v])

                beta = params[layer]
                for qubit_idx in range(n_qubits):
                    rx(2.0 * beta, qreg[qubit_idx])

        return graph_qaoa_kernel
