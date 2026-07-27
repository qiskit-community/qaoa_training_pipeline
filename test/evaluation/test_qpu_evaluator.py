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

    def qiskit_circuit_simulation(self, cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = qaoa_ansatz(cost_op, reps=len(params) // 2)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return float(result[0].data.evs)

    def test_evaluate(self):
        """Basic test of the evaluator."""
        angles = [0.1, 0.3]
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles)
        energy2 = self.qiskit_circuit_simulation(self.cost_op, angles)
        self.assertAlmostEqual(energy1, energy2, delta=0.05)

    def test_custom_ansatz_differs(self):
        """Test that we can construct the ansatz from a different operator."""
        ansatz_op = SparsePauliOp.from_list([("ZI", 1)])

        angles = [1.2, 1.3]
        self.evaluator.prepare_ansatz(ansatz_op, len(angles) // 2)
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=ansatz_op)
        self.evaluator.prepare_ansatz(self.cost_op, len(angles) // 2)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertNotAlmostEqual(energy1, energy2, delta=0.1)
