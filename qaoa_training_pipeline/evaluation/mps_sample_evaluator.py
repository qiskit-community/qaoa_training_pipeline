#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sample-based MPS evaluator"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


class MPSSampleEvaluator(BaseEvaluator):
    """Approximate the energy by sampling from a MPS.

    This MPS-based energy evaluator does not contract the MPS to compute the value of
    an observable. Instead, we draw samples `x` from the MPS and then evaluate the energy
    of the observable for each sample.
    """

    def __init__(
        self,
        chi: int | None = None,
        max_parallel_threads: int | None = None,
        shots: int | None = None,
    ):
        """Initialize the class.

        Args:
            chi: bond dimension of the MPS via AerSimulator.
            max_parallel_threads: maximum CPU cores used by AerSimulator.
            shots: number of circuit executions.
        """
        super().__init__()

        self._cost_op = None
        self._counts = []

        self.chi = chi or 20
        self.max_parallel_threads = max_parallel_threads or 10

        self._backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=self.chi,
            max_parallel_threads=self.max_parallel_threads,
        )

        self._sampler = BackendSamplerV2(backend=self._backend)

        self._shots = shots or 1000
        self.energies = []

    @property
    def cost_op(self):
        """Returns the cost operator"""
        return self._cost_op

    @cost_op.setter
    def cost_op(self, cost_op: SparsePauliOp):
        """Set the cost operator.

        This property setter computes some internal variables that help speed-up the computation
        of the energy for each sample `x`.
        """
        self._cost_op = cost_op
        self._reals = []
        self._ainds = []
        for pauli in self._cost_op:
            indices = tuple(idx for idx, val in enumerate(pauli.paulis[0].z) if val)
            self._ainds.append(indices)
            self._reals.append(np.real(pauli.coeffs[0]))

    def energy(self, sample: str) -> float:
        """Computes the energy for a given sample"""
        sample = [val == "1" for val in sample[::-1]]

        energy = 0
        for aidx, val in enumerate(self._reals):
            selected_bits = [sample[idx] for idx in self._ainds[aidx]]

            if sum(selected_bits) % 2 == 0:
                energy += val
            else:
                energy -= val

        return energy

    def total_energy(self, counts: dict) -> float:
        """Compute the energy of the counts."""
        tot_energy = 0
        self.energies = []
        shots = sum(counts.values())
        for sample, count in counts.items():
            self.energies.append(self.energy(sample))
            tot_energy += self.energies[-1] * count / shots

        return tot_energy

    # pylint: disable=too-many-positional-arguments
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
        # Set the cost op. We do not validate that the existing cost op,
        # if present, is the same as the given cost op.
        if self._cost_op is None:
            self.cost_op = cost_op

        ansatz = qaoa_ansatz(
            cost_operator=ansatz_op,
            reps=len(params) // 2,
            initial_state=initial_state,
            mixer_operator=mixer,
        ).decompose()

        ansatz.measure_all()

        pub = (ansatz, params, self._shots)
        result = self._sampler.run([pub]).result()

        self._counts = result[0].data.meas.get_counts()

        return self.total_energy(self._counts)

    def get_results_from_last_iteration(self):
        """Return the results from the last iteration."""
        return {"counts": self._counts}

    def to_config(self):
        config = super().to_config()
        config["chi"] = self.chi
        config["max_parallel_threads"] = self.max_parallel_threads
        config["shots"] = self._shots

        return config

    @classmethod
    def from_config(cls, config: dict) -> "MPSSampleEvaluator":
        """Initialize the evaluator from a configuration dictionary."""
        return cls(**config)

    def cvar(self, energies: list, alpha=1.00) -> float:
        """Compute the CVaR for given energies."""
        sorted_energies = sorted(energies)
        end_idx = max(int(alpha * len(energies)), 1)

        return float(np.sum(sorted_energies[0:end_idx]) / end_idx)
