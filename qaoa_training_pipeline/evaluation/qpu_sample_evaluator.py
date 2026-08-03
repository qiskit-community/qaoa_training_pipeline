#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluator that retrieves samples from the quantum hardware"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit_aer import AerSimulator
from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.transpilation.annotated_transpilation_passes import (
    AnnotatedCommuting2qGateRouter,
    AnnotatedPrepareCostLayer,
    SynthesizeAndSimplifyCostLayer,
    UnrollBoxes,
)

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator

_DEFAULT_SHOTS = 20_000


class QPUSampleEvaluator(BaseEvaluator):
    """Backend evaluator."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        backend,
        shots: int = _DEFAULT_SHOTS,
        cvar_alpha: float = 1.00,
        energy_minimization: bool = False,
        samples_folder: str | Path | None = None,
        sampler=None,
    ):
        """Initialize the class."""
        super().__init__()

        self._cost_op = None
        self._counts = []

        self._backend = backend

        if sampler is None:
            self._sampler = BackendSamplerV2(backend=self._backend)
        else:
            self._sampler = sampler

        self._shots = shots

        self._ansatz = None
        self._depth = None
        self._cvar_alpha = cvar_alpha
        self._energy_minimization = energy_minimization
        self._time_history_qpu = []
        self._time_history_cpu = []
        self._samples_folder = Path(samples_folder) if samples_folder is not None else None
        self._ansatz_digest = None
        self._cost_op_digest = None

    @property
    def cost_op(self):
        """Returns the cost operator"""
        return self._cost_op

    @cost_op.setter
    def cost_op(self, cost_op: SparsePauliOp):
        """Sets the cost operator"""
        self._cost_op = cost_op
        self._reals = []
        self._ainds = []
        start = time.time()
        for pauli in self._cost_op:
            indices = tuple(idx for idx, val in enumerate(pauli.paulis[0].z) if val)
            self._ainds.append(indices)
            self._reals.append(np.real(pauli.coeffs[0]))

        self._init_time = time.time() - start

    def energy(self, sample: str):
        """Compute the energy of a single sample."""
        bits = [val == "1" for val in sample[::-1]]

        energy = 0
        for val, aind in zip(self._reals, self._ainds):
            selected_bits = [bits[idx] for idx in aind]

            if sum(selected_bits) % 2 == 0:
                energy += val
            else:
                energy -= val

        return energy

    def energies(self, counts: dict):
        """Make a list of the energies for the counts."""
        energies = []

        for sample, count in counts.items():
            energies += [self.energy(sample)] * count

        return energies

    def average_energy(self, counts: dict):
        """Compute the average energy of the counts."""

        return np.average(self.energies(counts))

    @staticmethod
    def _op_digest(op: SparsePauliOp) -> bytes:
        hash_var = hashlib.sha256()
        hash_var.update(str(op.paulis.to_labels()).encode())
        hash_var.update(np.ascontiguousarray(op.coeffs.view(np.float64)).tobytes())
        return hash_var.digest()

    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ):
        """Evaluate the energy."""

        if isinstance(ansatz_circuit, SparsePauliOp):
            ansatz_op = ansatz_circuit
        elif ansatz_circuit is None:
            ansatz_op = cost_op
        else:
            raise NotImplementedError(
                "Custom ansatz circuits in format"
                f"{ansatz_circuit.__class__.__name__} are not yet supported."
            )

        cost_op_digest = self._op_digest(cost_op)
        if self._cost_op is None or self._cost_op_digest != cost_op_digest:
            self.cost_op = cost_op
            self._cost_op_digest = cost_op_digest

        ansatz_digest = self._op_digest(ansatz_op)
        # Avoid recreating the circuit all the time.
        if (
            self._ansatz is None
            or self._depth != len(params) // 2
            or self._ansatz_digest != ansatz_digest
        ):
            if any(sum(pauli.paulis[0].z) > 2 for pauli in ansatz_op):
                raise NotImplementedError(
                    f"{self.__class__.__name__} only supporst Pauli terms up to order 2 "
                )
            self.prepare_ansatz(
                ansatz_op, len(params) // 2, mixer=mixer, initial_state=initial_state
            )
            self._ansatz_digest = ansatz_digest

        # Time tracked QPU sample collection
        start = time.time()
        pub = (self._ansatz, params, self._shots)
        result = self._sampler.run([pub]).result()
        self._time_history_qpu.append(time.time() - start)

        # Time tracked CPU sample aggregation
        start = time.time()
        self._counts = result[0].data.meas.get_counts()
        energy = self.cvar(self.energies(self._counts))
        self._time_history_cpu.append(time.time() - start)

        if self._samples_folder is not None:
            iteration = len(self._time_history_qpu)
            with (self._samples_folder / f"{iteration}.json").open("w") as fout:
                json.dump(self._counts, fout, indent=4)

        return energy

    def prepare_ansatz(
        self,
        ansatz_circuit: QuantumCircuit,
        depth: int,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
    ):
        """Prepare the circuit for hardware execution."""
        ansatz = annotated_qaoa_ansatz(
            ansatz_circuit, reps=depth, mixer_operator=mixer, initial_state=initial_state
        )
        ansatz.measure_all()

        has_two_qubit_terms = any(sum(pauli.paulis[0].z) > 1 for pauli in ansatz_circuit)

        passes = []
        if has_two_qubit_terms:
            swap_strat = SwapStrategy.from_line(range(ansatz_circuit.num_qubits))
            passes += [AnnotatedPrepareCostLayer(), AnnotatedCommuting2qGateRouter(swap_strat)]
        passes += [
            SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cz", "sx", "rz", "id", "cx"]),
            UnrollBoxes(),
        ]

        swap_pm = PassManager(passes)
        qc = swap_pm.run(ansatz)

        self._ansatz = transpile(qc, self._backend, optimization_level=2)
        self._depth = depth

    def get_results_from_last_iteration(self):
        """Return the results from the last iteration."""
        return {"counts": self._counts}

    @classmethod
    def from_config(cls, config: dict):
        """Initialize class from config"""
        if config["backend"] == "aer_simulator_matrix_product_state":
            backend = AerSimulator(**config["backend_config"])
        else:
            raise ValueError(f"No backend {config['backend']} available")
        return cls(
            backend,
            config.get("shots", _DEFAULT_SHOTS),
            config.get("cvar_alpha", 1.00),
            config.get("energy_minimization", False),
            config.get("samples_folder", None),
            config.get("sampler", None),
        )

    def to_config(self):
        config = super().to_config()
        config["backend"] = self._backend.name
        config["backend_config"] = dict(self._backend.options.__dict__)
        config["cvar_alpha"] = self._cvar_alpha
        config["energy_minimization"] = self._energy_minimization
        config["shots"] = self._shots

        return config

    def cvar(self, energies: list):
        """Compute the CVAR energy."""

        if self._energy_minimization:
            sorted_energies = sorted(energies, key=lambda x: -x)
        else:
            sorted_energies = sorted(energies)

        end_idx = max(int(self._cvar_alpha * len(energies)), 1)

        return float(np.sum(sorted_energies[0:end_idx]) / end_idx)
