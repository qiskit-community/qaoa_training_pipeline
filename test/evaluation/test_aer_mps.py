#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Aer MPS-based QAOA evaluator tests."""

from unittest import TestCase

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qaoa_training_pipeline.evaluation.mps_aer import MPSAerEvaluator
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer


class TestStatevectorEvaluator(TestCase):
    """Test that the state vector evaluator works."""

    def setUp(self):
        """Setup the variables."""
        self.cost_op = SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)])
        self.evaluator = MPSAerEvaluator()

    def qiskit_circuit_simulation(self, cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = qaoa_ansatz(cost_op, reps=len(params) // 2)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return float(result[0].data.evs)

    def test_evaluate(self):
        """Basic test of the evaluator."""
        energy1 = self.evaluator.evaluate(self.cost_op, params=[0.1, 0.3])
        energy2 = self.qiskit_circuit_simulation(self.cost_op, [0.1, 0.3])
        self.assertTrue(abs(energy1 - energy2) < 0.1)

    def test_optimize(self):
        """Test that we can use this in a scipy optimization."""
        trainer = ScipyTrainer(self.evaluator, {"options": {"maxiter": 3, "rhobeg": 0.2}})
        result = trainer.train(cost_op=self.cost_op, params0=[0.2, 0.3])

        self.assertGreaterEqual(len(result["energy_history"]), 3)

    def test_custom_ansatz(self):
        """Test that we can construct the ansatz from a different operator."""
        ansatz_op = SparsePauliOp.from_list([("ZI", 1)])

        angles = [1.2, 1.3]

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=ansatz_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) > 0.1)

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=self.cost_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) < 0.1)
