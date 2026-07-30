#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sample-based evaluator abstract class"""

import hashlib
import time
from abc import abstractmethod

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator

_DEFAULT_SHOTS = 1000


class SampleEvaluator(BaseEvaluator):
    """Approximate the energy by sampling from a backend.

    This energy evaluator evalautes the value of the cost function by drawing samples
    from a backend and classically computing the energy of those samples.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        backend,
        shots: int | None = None,
        sampler: type[BaseSamplerV2] | None = None,
        cvar_alpha: float = 1.00,
        energy_minimization: bool = False,
        samples_folder=None,
    ):
        """Initialize the class.

        Args:
            backend: the backend class used to run the circuit and prepare
            the state the class will draw samples from.
            shots: number of circuit executions.
            sampler: the sampler class the backend will use to draw samples from the state.
            cvar_alpha: CVaR alpha value used to compute it
            samples_folder: folder where the class will store the samples
        """
        super().__init__()

        self._cost_op = None
        self._counts = []

        self._backend = backend
        self._sampler = sampler(backend=self._backend)

        self._ansatz = None
        self._shots = shots or _DEFAULT_SHOTS
        self._cvar_alpha = cvar_alpha
        self._energy_minimization = energy_minimization
        self._samples_folder = samples_folder
        self._ansatz_digest = None
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
        start = time.time()
        for pauli in self._cost_op:
            indices = tuple(idx for idx, val in enumerate(pauli.paulis[0].z) if val)
            self._ainds.append(indices)
            self._reals.append(np.real(pauli.coeffs[0]))

        self._init_time = time.time() - start

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

    @abstractmethod
    # pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
    ):
        """Evaluate the energy. This method must be implemented by the sub-classes"""

        # Processing of the ansatz circuit to use, cost operator, and cache is
        # shared amongst subclasses
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

        ansatz_digest = self._ansatz_op_digest(ansatz_op)
        # Avoid recreating the circuit all the time.
        if (
            self._ansatz is None
            or self._depth != len(params) // 2
            or self._ansatz_digest != ansatz_digest
        ):
            self.prepare_ansatz(
                ansatz_op, len(params) // 2, mixer=mixer, initial_state=initial_state
            )
            self._ansatz_digest = ansatz_digest

        raise NotImplementedError("Sub-classes must implement `evaluate`.")

    def prepare_ansatz(
        self,
        ansatz_circuit: QuantumCircuit,
        depth: int,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
    ):
        """Prepares the ansatz to run."""
        raise NotImplementedError("Sub-classes must implement `prepare_ansatz`")

    def _ansatz_op_digest(self, op: SparsePauliOp) -> bytes:
        """Cache ansatz operator via a digest"""
        hash_var = hashlib.sha256()
        hash_var.update(str(op.paulis.to_labels()).encode())
        hash_var.update(np.ascontiguousarray(op.coeffs.view(np.float64)).tobytes())
        return hash_var.digest()

    def get_results_from_last_iteration(self):
        """Return the results from the last iteration."""
        return {"counts": self._counts}

    @abstractmethod
    def to_config(self):
        """Serialize the class to a config dict"""
        raise NotImplementedError("Sub-classes must implement `to_config`")

    @classmethod
    def from_config(cls, config: dict) -> "SampleEvaluator":
        """Create a class from a config dict"""
        raise NotImplementedError("Sub-classes must implement `from_config`")

    def cvar(self, energies: list):
        """Compute the CVAR energy."""

        if self._energy_minimization:
            sorted_energies = sorted(energies, key=lambda x: -x)
        else:
            sorted_energies = sorted(energies)

        end_idx = max(int(self._cvar_alpha * len(energies)), 1)

        return float(np.sum(sorted_energies[0:end_idx]) / end_idx)
