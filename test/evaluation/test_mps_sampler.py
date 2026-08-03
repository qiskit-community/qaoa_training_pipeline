#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Aer MPS-based QAOA evaluator tests."""

from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from qaoa_training_pipeline.evaluation.mps_sample_evaluator import MPSSampleEvaluator


class TestMPSSampleEvaluator(TestCase):
    """Test that the MPS evaluator from qiskit aer works."""

    def setUp(self):
        """Setup the variables."""
        self.cost_op = SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)])
        self.evaluator = MPSSampleEvaluator(shots=40000, chi=32)

    def qiskit_circuit_simulation(self, cost_op, params, initial_state=None):
        """This is the baseline simulation based on Qiskit."""

        ansatz = qaoa_ansatz(cost_op, reps=len(params) // 2, initial_state=initial_state)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return float(result[0].data.evs)

    def test_evaluate(self):
        """Basic test of the evaluator against different cost operators."""
        angles = [0.1, 0.3]

        cost_ops = [
            SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)]),
            SparsePauliOp.from_list([("IZZ", 2.0), ("ZIZ", 3), ("ZZZ", 4)]),
            SparsePauliOp.from_list([("ZZZ", 1.0)]),
            SparsePauliOp.from_list([("Z", 2)]),
        ]

        for cost_op in cost_ops:
            with self.subTest(cost_op=cost_op):
                evaluator = MPSSampleEvaluator(shots=80000, chi=64)
                energy1 = evaluator.evaluate(cost_op, params=angles)
                energy2 = self.qiskit_circuit_simulation(cost_op, angles)
                self.assertTrue(
                    abs(energy1 - energy2) < 0.05,
                    f"{cost_op.paulis}: mps={energy1:.4f} statevector={energy2:.4f} "
                    f"diff={abs(energy1 - energy2):.4f}",
                )

    def test_custom_ansatz(self):
        """Test that we can construct the ansatz from a different operator."""
        ansatz_op = SparsePauliOp.from_list([("ZI", 1)])

        angles = [1.2, 1.3]

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=ansatz_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) > 0.1)

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=self.cost_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) < 0.05)

    def test_from_config(self):
        """Test that we can create the evaluator from a config dictionary"""
        config = {"chi": 32, "max_parallel_threads": 10, "shots": 40000}
        evaluator = MPSSampleEvaluator.from_config(config)

        self.assertIsInstance(evaluator, MPSSampleEvaluator)
        angles = [0.1, 0.3]
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles)
        energy2 = evaluator.evaluate(self.cost_op, params=angles)
        self.assertTrue(abs(energy1 - energy2) < 0.05)

    def test_to_config(self):
        """Test that we can serialize the evaluator to a config dictionary"""
        config = self.evaluator.to_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(
            config,
            {"name": "MPSSampleEvaluator", "chi": 32, "max_parallel_threads": 10, "shots": 40000},
        )

    def test_explicit_initial_state(self):
        """Test that we get the correct energy when the initial state is prepared to
        be a product state |111...>."""
        cost_op = SparsePauliOp.from_list([("IIZZ", 1), ("ZIIZ", 2), ("IZII", 4)])

        angles = [np.pi / 2, 4.56]
        initial_state = QuantumCircuit(4)
        initial_state.x([0, 1, 2, 3])

        energy = self.evaluator.evaluate(
            cost_op,
            angles,  # beta, gamma
            initial_state=initial_state,
        )

        energy_benchmark = self.qiskit_circuit_simulation(
            cost_op, angles, initial_state=initial_state
        )
        # The energy of the state produced by the QAOA circuit with initial state
        # |111...> for this cost_op and beta pi/2 is 7, but the value is still checked
        # against a Statevector simulation
        self.assertAlmostEqual(energy, 7.0)
        self.assertAlmostEqual(energy, energy_benchmark)

    def test_trivial_warm_start(self):
        r"""Test a warm-start like QAOA. We start in 0001.

        In the case of a warm-start the mixer changes from `+X` to

        ..math::

            \sin(\theta)X - \cos(\theta)Z

        which is equivalent to the conventional mixer when theta is pi/2.
        """
        cost_op = SparsePauliOp.from_list([("IIZZ", -0.5), ("ZIIZ", -0.5), ("IZIZ", -0.5)])

        params = [0.333, 4.56]  # beta, gamma

        # Example of a warm-start where q0 is in 1 and the other qubits in 0.
        # In this case the cost-op does nothing and neither does beta.
        init = QuantumCircuit(4)
        init.ry(-np.pi, 0)

        mixer_op = SparsePauliOp.from_list([("ZIII", -1), ("IZII", 1), ("IIZI", 1), ("IIIZ", 1)])

        energy = self.evaluator.evaluate(cost_op, params, initial_state=init, mixer=mixer_op)

        # Prepares the |0001> state which has energy 3/2.
        self.assertAlmostEqual(energy, 1.5)

    def test_warm_start(self):
        """
        Test the efficient depth-one with a custom initial state and standard mixer.
        """
        cost_op = SparsePauliOp.from_list(
            [
                ("ZII", -1),
                ("IZI", +0.81),
                ("IIZ", -0.43),
                ("ZZI", -1.5),
                ("ZIZ", +0.21),
                ("IZZ", -0.11),
            ]
        )
        params = [0.41, 0.34]
        init = QuantumCircuit(3)
        for j in range(3):
            theta = 0.1 * (j + 1)
            init.ry(theta, j)

        # Build the full QAOA circuit to get the reference statevector
        qc = QuantumCircuit(3)
        qc.compose(init, inplace=True)

        # Cost unitary: exp(-i gamma * cost_op)
        cost_gate = PauliEvolutionGate(cost_op, time=params[1])
        qc.append(cost_gate, range(3))

        # Mixer unitary: exp(-i beta * sum_j X_j)  ≡  Rx(2*beta) on each qubit
        for j in range(3):
            qc.rx(2 * params[0], j)

        # Compute <psi | cost_op | psi> via statevector simulation
        sv = Statevector(qc)
        expected_energy = sv.expectation_value(cost_op).real

        energy = self.evaluator.evaluate(cost_op, params, initial_state=init)
        self.assertAlmostEqual(float(energy), expected_energy, delta=0.05)

    def test_custom_ansatz_nodelist(self):
        """Test that we get the correct result when running with a custom ansatz."""
        cost_op = SparsePauliOp.from_list(
            [
                ("IIZZ", -1),
                ("IZIZ", -1),
                ("ZIIZ", 1),
                ("IZZI", 1),
                ("ZIZI", -1),
                ("ZZII", 1),
            ]
        )

        ansatz = SparsePauliOp.from_list([("ZIIZ", 1), ("IZZI", 1)])

        # Construct the QAOA circuit corresponding to the ansatz.
        qaoa_circuit = qaoa_ansatz(SparsePauliOp.from_list([("ZIIZ", 1), ("IZZI", 1)]))
        qaoa_circuit = transpile(qaoa_circuit, basis_gates=["rzz", "h", "rx", "rz"])

        beta, gamma = 1, 2
        estimator = StatevectorEstimator()
        expected = float(
            estimator.run([(qaoa_circuit, cost_op, [beta, gamma])]).result()[0].data.evs
        )

        actual = self.evaluator.evaluate(
            cost_op,
            params=[beta, gamma],
            ansatz_circuit=ansatz,
        )

        self.assertAlmostEqual(actual, expected, places=1)
