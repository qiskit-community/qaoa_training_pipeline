#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the conventions-consistent QAOA ansatz."""

from functools import reduce

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from qaoa_training_pipeline.utils.qaoa_ansatz import default_mixer_operator, qaoa_ansatz
from tests.training_pipeline_test_case import TrainingPipelineTestCase


@ddt
class TestQAOAAnsatz(TrainingPipelineTestCase):
    """Test that the QAOA ansatz follows the conventions in CONVENTIONS.md."""

    def setUp(self):
        """Setup a cost operator with two-local and one-local terms."""
        self.cost_op = SparsePauliOp.from_list(
            [("IIZZ", 1.0), ("IZZI", -0.7), ("ZIIZ", 0.4), ("IZII", 0.3)]
        )

    @data(1, 3, 4)
    def test_default_mixer_operator(self, num_qubits):
        """Test that the default mixer is -sum_i X_i."""
        x_mat, identity = np.array([[0, 1], [1, 0]]), np.eye(2)

        expected = -sum(
            reduce(np.kron, [x_mat if k == i else identity for k in range(num_qubits)])
            for i in range(num_qubits)
        )

        mixer = default_mixer_operator(num_qubits)

        self.assertEqual(mixer.num_qubits, num_qubits)
        self.assertTrue(np.allclose(mixer.to_matrix(), expected))

    @data(1, 2)
    def test_default_mixer_circuit(self, reps):
        """Test that the default ansatz matches a circuit with explicit Rx(-2 beta) mixers."""
        rng = np.random.default_rng(reps)
        betas, gammas = rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi, reps)
        num_qubits = self.cost_op.num_qubits

        expected = QuantumCircuit(num_qubits)
        expected.h(range(num_qubits))
        for beta, gamma in zip(betas, gammas):
            expected.append(PauliEvolutionGate(self.cost_op, time=gamma), range(num_qubits))
            expected.rx(-2 * beta, range(num_qubits))

        # The parameters are ordered as [beta_1, ..., beta_p, gamma_1, ..., gamma_p].
        ansatz = qaoa_ansatz(self.cost_op, reps=reps, mixer_operator=None)
        actual = ansatz.assign_parameters(np.concatenate([betas, gammas]))

        self.assertTrue(Operator(actual).equiv(Operator(expected)))

    def test_initial_state_is_mixer_ground_state(self):
        """Test that the default initial state is the ground state of the default mixer."""
        num_qubits = self.cost_op.num_qubits
        ansatz = qaoa_ansatz(self.cost_op)

        # With all angles at zero only the initial state remains.
        initial_state = Statevector(ansatz.assign_parameters([0.0] * ansatz.num_parameters))

        mixer = default_mixer_operator(num_qubits)
        ground_energy = np.min(np.linalg.eigvalsh(mixer.to_matrix()))

        self.assertAlmostEqual(ground_energy, -num_qubits)
        self.assertAlmostEqual(initial_state.expectation_value(mixer).real, ground_energy)
