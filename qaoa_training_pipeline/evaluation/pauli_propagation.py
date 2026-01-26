# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pauli propagation-based QAOA evaluator."""

import importlib.util
import warnings
from typing import Sequence

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator

# cspell: ignore juliacall seval qarg qargs overlapwithzero
# Safely import Julia if is is installed.

# Ensure names exist at module import time
jl = None  # pylint: disable=invalid-name # set to juliacall.Main if available
convert = None  # pylint: disable=invalid-name # set to juliacall.convert if available
pp = None  # pylint: disable=invalid-name

jl_loader = importlib.util.find_spec("juliacall")
HAS_JL = jl_loader is not None
if HAS_JL:
    # pylint: disable=no-name-in-module
    from juliacall import Main as jl  # type: ignore
    from juliacall import convert

    try:
        jl.seval("using PauliPropagation")

    # pylint: disable=broad-exception-caught
    except Exception as _:
        warnings.warn(
            "juliacall installed but no PauliPropagation. Attempting to install PauliPropagation."
        )
        jl.seval('Pkg.add("PauliPropagation")')
        jl.seval("using PauliPropagation")
    pp = jl.PauliPropagation

    # Here is the mapping between the supported qiskit gates and the corresponding PP gates.
    PAULI_ROTATIONS = {
        "rx": (convert(jl.Symbol, "X"),),
        "ry": (convert(jl.Symbol, "Y"),),
        "rz": (convert(jl.Symbol, "Z"),),
        "rxx": (
            convert(jl.Symbol, "X"),
            convert(jl.Symbol, "X"),
        ),
        "ryy": (
            convert(jl.Symbol, "Y"),
            convert(jl.Symbol, "Y"),
        ),
        "rzz": (
            convert(jl.Symbol, "Z"),
            convert(jl.Symbol, "Z"),
        ),
    }

    CLIFFORD_GATES = {
        "h": convert(jl.Symbol, "H"),
        "x": convert(jl.Symbol, "X"),
        "y": convert(jl.Symbol, "Y"),
        "z": convert(jl.Symbol, "Y"),
        "s": convert(jl.Symbol, "S"),
        "cx": convert(jl.Symbol, "CNOT"),
        "swap": convert(jl.Symbol, "swap"),
    }

    SUPPORTED_GATES = list(CLIFFORD_GATES.keys()) + list(PAULI_ROTATIONS.keys())
else:
    PAULI_ROTATIONS = dict()
    CLIFFORD_GATES = dict()
    SUPPORTED_GATES = []
    # pylint: disable=invalid-name
    pp = None


class PPEvaluator(BaseEvaluator):
    """Evaluator based on the Pauli propagation method.

    This class requires that the system has the Pauli propagation toolkit
    https://github.com/MSRudolph/PauliPropagation.jl installed. Note that this
    toolkit also requires Julia. Therefore, it is not supported by the default
    requirements of the QAOA training pipeline and people need to install
    PauliPropagation and Julia themselves.
    """

    def __init__(self, pp_kwargs: dict | None = None):
        """Initialize the Pauli propagation evaluator.

        Args:
            pp_kwargs: Keyword arguments that will be passed to PauliPropagation.jl.
                The parameters that can be passed to the function are documented in
                https://msrudolph.github.io/PauliPropagation.jl/stable/api/Propagation
                under the PauliPropagation.propagate function. Furthermore, all the types
                must be compatible with juliacall
                https://juliapy.github.io/PythonCall.jl/stable/conversion-to-julia/.
                The most relevant parameters are:

                 - `max_weight`: this defines the maximum Pauli weight of the Pauli operator
                     that are kept in the Heisenberg evolution.
                 - `min_abs_coeff`: this defines the threshold on the absolute value of the
                     operator coefficient below which terms are neglected in the Heisenberg evolution.

                If None is given then we default to `max_weight=9` and `min_abs_coeff=1e-5`.
        """

        # Importing Julia can cause the kernel to crash, typically, on windows.
        # Therefore we first gracefully check for it before importing
        if not HAS_JL:
            raise ImportError(
                f"{self.__class__.__name__} requires Julia and the PauliPropagation.jl package."
                f"Please install Julia and the PauliPropagation.jl package."
                f"See https://github.com/MSRudolph/PauliPropagation.jl for more details."
            )

        # These kwargs match the default ones from PauliPropagation.jl
        self.pp_kwargs = dict(
            max_weight=np.inf,
            min_abs_coeff=1e-10,
            max_freq=np.inf,
            max_sins=np.inf,
            customtruncfunc=None,
        )
        if pp_kwargs is not None:
            self.pp_kwargs.update(pp_kwargs)

    # pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: Sequence[float],
        mixer: BaseOperator | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the QAOA circuit parameters."""

        if ansatz_circuit is not None:
            raise NotImplementedError(
                f"Custom Ansatz circuits are currently not supported in {self.__class__.__name__}."
            )

        circuit = qaoa_ansatz(
            cost_op,
            reps=len(params) // 2,
            initial_state=initial_state,
            mixer_operator=mixer,
        )
        bound_circuit = circuit.assign_parameters(params, inplace=False)
        # Transpile the circuit to the set of supported gates
        circuit = transpile(bound_circuit, basis_gates=SUPPORTED_GATES)

        pp_circuit, parameter_map = self.qc_to_pp(circuit)
        pp_observable = self.sparse_pauli_op_to_pp(cost_op)

        pp_params = [params[i] if isinstance(i, int) else i for i in parameter_map]
        assert pp, "pp must be defined before calling evaluate()"
        pauli_sum = pp.propagate(pp_circuit, pp_observable, pp_params, **self.pp_kwargs)
        return pp.overlapwithzero(pauli_sum)

    def sparse_pauli_op_to_pp(self, op: SparsePauliOp):
        """Returns the PP PauliSum representation of the SparsePauliOp."""
        if not HAS_JL or jl is None or pp is None or convert is None:
            raise RuntimeError(
                "Julia/PauliPropagation is not available. Install `juliacall` and the "
                "PauliPropagation.jl package to use this function."
            )
        n_qubits = op.num_qubits
        assert pp, "pp must be defined before calling sparse_pauli_op_to_pp()"
        pp_pauli_sum = pp.PauliSum(n_qubits)
        for pauli, qubits, coefficient in op.to_sparse_list():
            pauli_symbols = pp.seval("Vector{Symbol}")()
            for p in pauli:
                jl.push_b(pauli_symbols, convert(jl.Symbol, str(p)))
            pp_qubits = pp.seval("Vector{Int}")()
            for q in qubits:
                jl.push_b(pp_qubits, q + 1)

            pp.add_b(pp_pauli_sum, pauli_symbols, pp_qubits, coefficient.real)
        return pp_pauli_sum

    def qc_to_pp(
        self, circuit: QuantumCircuit
    ) -> tuple[list[tuple[str, list[int]]], list[int | float]]:
        """
        Args:
            circuit: The Qiskit circuit with no free parameters.

        Returns:
            pp_circuits: A representation of the circuit on which we can call the Pauli propagation code.
                That is a list of tuples with the name of the gate and the qubits it acts on and a
                list with all the position of non-clifford gates.
            parameter_map: A list mapping between the parameters in the qiskit circuit and the
                Pauli Propagation representation. If the value in the list is a float it means
                the qiskit gate had a bound parameter. Otherwise the value will be an integer
                indicating the index of the unbound qiskit parameter.
        """
        if len(circuit.parameters) > 0:
            raise ValueError("The provided quantum circuit has unassigned parameters.")

        dag = circuit_to_dag(circuit, False)

        # The circuit must only contain supported gates.
        op_nodes = list(dag.topological_op_nodes())
        assert pp, "pp must be defined before calling qc_to_op()"
        pp_circuit = pp.seval("Vector{Gate}")()
        parameter_map = []
        for node in op_nodes:
            q_indices = tuple(dag.find_bit(qarg).index + 1 for qarg in node.qargs)
            name = node.op.name
            if name in PAULI_ROTATIONS:
                pauli_rot = pp.PauliRotation(PAULI_ROTATIONS[name], q_indices)
                pp.push_b(pp_circuit, pauli_rot)
                if isinstance(node.op.params[0], float):
                    parameter_map.append(node.op.params[0])
                else:
                    parameter_map.append(node.op.params[0].index)
            elif name in CLIFFORD_GATES:
                clifford_gate = pp.CliffordGate(CLIFFORD_GATES[name], q_indices)
                pp.push_b(pp_circuit, clifford_gate)
            else:
                print(f"We did not find a gate for {node.op.name}. Skipping Gate.")

        return pp_circuit, parameter_map

    def to_config(self) -> dict:
        """Json serializable config to keep track of how results are generated."""
        config = super().to_config()

        config["pp_kwargs"] = self.pp_kwargs

        return config

    @classmethod
    def from_config(cls, config: dict) -> "PPEvaluator":
        """Initialize the Pauli propagation evaluator from a config dictionary."""
        return cls(**config)

    @classmethod
    # pylint: disable=unused-argument
    def parse_init_kwargs(cls, init_kwargs: str | None = None) -> dict:
        """A hook that sub-classes can implement to parse initialization kwargs."""

        if init_kwargs is None:
            return dict()

        items = init_kwargs.split(":")

        if len(items) % 2 != 0:
            raise ValueError(
                f"Malformed keyword arguments {init_kwargs}: should be k1:v1:k2:v2_...."
            )

        return {"pp_kwargs": {items[idx]: float(items[idx + 1]) for idx in range(0, len(items), 2)}}
