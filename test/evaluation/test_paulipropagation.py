#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Statevector-based QAOA evaluator tests."""

from unittest import TestCase
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from ddt import data, ddt, unpack

from qiskit_aer import Aer

from qaoa_training_pipeline.evaluation.statevector_evaluator import StatevectorEvaluator
from qaoa_training_pipeline.evaluation.pauli_propagation import PPEvaluator
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer


TEST_CASES = [
    (SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)]), [0.2, 0.3]),
    (SparsePauliOp.from_list([("IZ", 0.5), ("ZI", -0.5)]), [0.1, 0.4]),
    (SparsePauliOp.from_list([("XX", 1.0), ("YY", 1.0)]), [0.5, 0.5]),
    (SparsePauliOp.from_list([("XIZX", 3.0), ("YIZY", 1.0)]), [0.3, 0.4, 0.1, 0.5]),
]


@ddt
class TestPPEvaluator(TestCase):
    """Test that the state vector evaluator works."""

    def setUp(self):
        """Setup the evaluators."""
        pp_kwargs = dict(max_weight=9, min_abs_coeff=1e-5)
        self.evaluator = PPEvaluator(pp_kwargs)
        self.sv_evaluator = StatevectorEvaluator()
        self.simulator = Aer.get_backend("aer_simulator_statevector")

    def qiskit_circuit_simulation(self, cost_op, beta, gamma, circ_op=None):
        """This is the baseline simulation based on Qiskit."""
        circ_op = circ_op or cost_op

        depth_one_qaoa = qaoa_ansatz(
            circ_op,
            reps=1,
        ).decompose()

        # Printing parameters gives [ParameterVectorElement(β[0]), ParameterVectorElement(γ[0])]
        depth_one_qaoa.assign_parameters([beta, gamma], inplace=True)
        depth_one_qaoa = transpile(depth_one_qaoa, basis_gates=["cx", "sx", "x", "rz"])
        depth_one_qaoa.save_statevector()

        res = self.simulator.run(depth_one_qaoa).result()
        state = res.get_statevector()

        cost_mat = np.diag(cost_op.to_matrix())

        nqubits = cost_op.num_qubits
        return np.real(sum(state[i].conj() * cost_mat[i] * state[i] for i in range(2**nqubits)))

    @data(*TEST_CASES)
    @unpack
    def test_evaluate(self, cost_op, params):
        """Data-driven test of the evaluator."""
        self.assertAlmostEqual(
            self.evaluator.evaluate(cost_op, params=params),
            self.sv_evaluator.evaluate(cost_op, params=params),
        )

    @data(*TEST_CASES)
    @unpack
    def test_optimize(self, cost_op, params0):
        """Data-driven test of optimization."""
        trainer = ScipyTrainer(self.evaluator, {"options": {"maxiter": 3, "rhobeg": 0.2}})
        result = trainer.train(cost_op=cost_op, params0=params0)
        self.assertGreaterEqual(len(result["energy_history"]), 3)

    def test_from_config(self):
        """Test that we can initialize from a config."""
        init_kwargs = PPEvaluator.parse_init_kwargs("max_weight:4:min_abs_coeff:1e-6")
        evaluator = PPEvaluator.from_config(init_kwargs)

        cost_op = SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)])
        energy = evaluator.evaluate(cost_op, params=[0.2, 0.3])

        self.assertTrue(isinstance(energy, float))

        self.assertEqual(evaluator.pp_kwargs["max_weight"], 4)
        self.assertEqual(evaluator.pp_kwargs["min_abs_coeff"], 1e-6)

    def test_from_none_config(self):
        """Test that we can initialize from a config."""
        init_kwargs = PPEvaluator.parse_init_kwargs(None)
        evaluator = PPEvaluator.from_config(init_kwargs)

        self.assertTrue(isinstance(evaluator, PPEvaluator))

    @data((1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_sparse_ansatz(self, beta, gamma):
        """Test the case where the ansatz structure does not match the graph."""

        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0), ("ZIIZ", 1.0)])
        circ_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0)])

        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma, circ_op)

        energy = self.evaluator.evaluate(cost_op, [beta, gamma], ansatz_circuit=circ_op)
        energy2 = self.evaluator.evaluate(cost_op, [beta, gamma])

        # Ensure circuit simulation and efficient depth-one match.
        self.assertAlmostEqual(energy, baseline, places=6)

        # Ensure efficient depth-one gives different results if the circuit structure is omitted.
        self.assertTrue(abs(energy2 - energy) > 0.02)
