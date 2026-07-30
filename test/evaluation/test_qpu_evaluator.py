#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QPU-based QAOA evaluator tests."""

from unittest import TestCase

from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qaoa_training_pipeline.evaluation.qpu_sample_evaluator import QPUSampleEvaluator


class TestQPUSampleEvaluator(TestCase):
    """Test that the QPU evaluator from qiskit aer works."""

    def setUp(self):
        """Setup the variables."""
        self.cost_op = SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)])
        self.evaluator = QPUSampleEvaluator(
            backend=AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=20,
                max_parallel_threads=1,
            )
        )

    @staticmethod
    def qiskit_circuit_simulation(cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = qaoa_ansatz(cost_op, reps=len(params) // 2)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return float(result[0].data.evs)

    def test_evaluate(self):
        """Basic test of the evaluator."""
        angles = [0.1, 0.3]

        cost_ops = [
            SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)]),
            SparsePauliOp.from_list([("Z", 2)]),
        ]

        for cost_op in cost_ops:
            with self.subTest(cost_op=cost_op):
                evaluator = QPUSampleEvaluator(
                    backend=AerSimulator(
                        method="matrix_product_state",
                        matrix_product_state_max_bond_dimension=64,
                        max_parallel_threads=1,
                    ),
                    shots=80000,
                )
                energy1 = evaluator.evaluate(cost_op, params=angles)
                energy2 = self.qiskit_circuit_simulation(cost_op, angles)
                self.assertAlmostEqual(energy1, energy2, delta=0.05)

        cost_ops_high_order = [
            SparsePauliOp.from_list([("IZZ", 2.0), ("ZIZ", 3), ("ZZZ", 4)]),
            SparsePauliOp.from_list([("ZZZ", 1.0)]),
        ]

        for cost_op in cost_ops_high_order:
            with self.subTest(cost_op=cost_op):
                evaluator = QPUSampleEvaluator(
                    backend=AerSimulator(
                        method="matrix_product_state",
                        matrix_product_state_max_bond_dimension=64,
                        max_parallel_threads=1,
                    ),
                    shots=80000,
                )
                with self.assertRaises(ValueError):
                    evaluator.evaluate(cost_op, params=angles)

    def test_custom_ansatz_differs(self):
        """Test that we can construct the ansatz from a different operator."""
        ansatz_op = SparsePauliOp.from_list([("ZI", 1)])

        angles = [1.2, 1.3]
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=ansatz_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=self.cost_op)

        self.assertNotAlmostEqual(energy1, energy2, delta=0.1)

    def test_from_config(self):
        """Test that we can create the evaluator from a config dictionary"""
        config = {
            "backend": "aer_simulator_matrix_product_state",
            "backend_config": {
                "method": "matrix_product_state",
                "matrix_product_state_max_bond_dimension": 20,
                "max_parallel_threads": 1,
            },
            "shots": 40000,
        }
        evaluator = QPUSampleEvaluator.from_config(config)

        self.assertIsInstance(evaluator, QPUSampleEvaluator)
        angles = [0.1, 0.3]
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles)
        energy2 = evaluator.evaluate(self.cost_op, params=angles)
        self.assertTrue(abs(energy1 - energy2) < 0.05)

    def test_to_config(self):
        """Test that we can serialize the evaluator to a config dictionary"""
        config = self.evaluator.to_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["name"], "QPUSampleEvaluator")
        self.assertEqual(config["backend"], "aer_simulator_matrix_product_state")
        self.assertEqual(config["cvar_alpha"], 1)
        self.assertEqual(config["energy_minimization"], False)
        self.assertEqual(config["backend_config"]["method"], "matrix_product_state")
