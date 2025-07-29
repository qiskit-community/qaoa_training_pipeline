#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the efficient depth-one evaluator."""

from test import TrainingPipelineTestCase

from ddt import ddt, data, unpack
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import qaoa_ansatz, PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator

from qiskit_aer import Aer

from qaoa_training_pipeline.evaluation.efficient_depth_one import (
    EfficientDepthOneEvaluator,
)


@ddt
class TestEfficientDepthOne(TrainingPipelineTestCase):
    """Test the efficient depth-one evaluator."""

    def setUp(self) -> None:
        """Initialize some variables for testing."""
        self.simulator = Aer.get_backend("aer_simulator_statevector")
        self.evaluator = EfficientDepthOneEvaluator()

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

    @data((0, 0), (1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_basic(self, beta, gamma):
        """Test a few basic points of the efficient depth-one ansatz."""

        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZIIZ", 1.0), ("IZIZ", 1.0)])

        # Qiskit simulation
        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma)

        # Now compute the efficient depth-one energy
        energy = self.evaluator.evaluate(cost_op, [beta, gamma])

        self.assertAlmostEqual(energy, baseline, places=6)

    @data((0, 0), (1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_basic_single_z(self, beta, gamma):
        """Test a few basic points of the efficient depth-one ansatz."""

        cost_op = SparsePauliOp.from_list([("IZZ", 1.0), ("IIZ", 0.0), ("ZIZ", 1.0), ("ZII", 1.0)])

        # Qiskit simulation
        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma)

        # Now compute the efficient depth-one energy
        energy = self.evaluator.evaluate(cost_op, [beta, gamma])

        self.assertAlmostEqual(energy, baseline, places=6)

    @data((0, 0), (1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_weighted_graph(self, beta, gamma):
        """Test that we get the correct result when the graph is weighted."""
        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZIIZ", -1.0), ("IZIZ", 2.3)])

        # Qiskit simulation
        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma)

        # Now compute the efficient depth-one energy
        energy = self.evaluator.evaluate(cost_op, [beta, gamma])

        self.assertAlmostEqual(energy, baseline, places=6)

    @data((1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_sparse_ansatz(self, beta, gamma):
        """Test the case where the ansatz structure does not match the graph."""

        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0), ("ZIIZ", 1.0)])
        circ_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0)])

        # circuit_ansatz corresponds to circ_op which is a sparser cost_op.
        circuit_ansatz = QuantumCircuit(4)
        gamma_ = Parameter("g")
        circuit_ansatz.rzz(2 * gamma_, 0, 1)
        circuit_ansatz.rzz(2 * gamma_, 2, 3)

        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma, circ_op)

        energy = self.evaluator.evaluate(cost_op, [beta, gamma], ansatz_circuit=circuit_ansatz)
        energy2 = self.evaluator.evaluate(cost_op, [beta, gamma])

        # Ensure circuit simulation and efficient depth-one match.
        self.assertAlmostEqual(energy, baseline, places=6)

        # Ensure efficient depth-one gives different results if the circuit structure is omitted.
        self.assertTrue(abs(energy2 - energy) > 0.02)

    def test_default_mixer(self):
        """Ensure the default mixer is the expected one."""

        expected = Operator(PauliEvolutionGate(Pauli("X"), 1)).data

        self.assertTrue((np.allclose(expected, self.evaluator.mixer(1.0))))

    def test_from_config(self):
        """Test that we can instantiate from a config."""
        config = {}

        evaluator = EfficientDepthOneEvaluator.from_config(config)

        self.assertTrue(isinstance(evaluator, EfficientDepthOneEvaluator))

    def test_custom_ansatz_nodelist(self):
        """Test that we get the correct result when running with a custom ansatz.

        This test is specifically designed to check that the adjacency matrix is
        properly constructed when an Ansatz is given. This is because
        `nx.adjacency_matrix` works both with and without the `nodelist` argument.
        When `nodelist` is not given random behaviour can occure.
        """
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

        # Construct an ansatz. The gate order (3, 0) and (2, 1) is specifically designed to trigger
        # wrong behaviour in `nx.adjacency_matrix` in the absence of nodelist.
        gamma_param = Parameter("gamma")
        ansatz = QuantumCircuit(5)
        ansatz.rzz(2 * gamma_param, 3, 0)
        ansatz.rzz(2 * gamma_param, 2, 1)

        # Construct the QAOA circuit corresponding to the ansatz.
        qaoa_circuit = qaoa_ansatz(SparsePauliOp.from_list([("ZIIZ", 1), ("IZZI", 1)]))
        qaoa_circuit = transpile(qaoa_circuit, basis_gates=["rzz", "h", "rx", "rz"])

        beta, gamma = 1, 2
        estimator = StatevectorEstimator()
        expected = float(
            estimator.run([(qaoa_circuit, cost_op, [beta, gamma])]).result()[0].data.evs
        )

        actual = EfficientDepthOneEvaluator().evaluate(
            cost_op,
            params=[beta, gamma],
            ansatz_circuit=ansatz,
        )

        self.assertAlmostEqual(actual, expected, places=8)

    def test_custom_vs_default(self):
        """Test that providing the default mixer and initial state gives standard results."""
        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZIIZ", 1.0), ("IZIZ", 1.0)])

        beta, gamma = 1.23, 4.56

        # Qiskit simulation
        baseline = self.qiskit_circuit_simulation(cost_op, beta, gamma)

        # Now compute the efficient depth-one energy
        initial_state = QuantumCircuit(4)
        initial_state.h(range(4))

        mixer = QuantumCircuit(4)
        mixer.rx(2 * Parameter("b"), range(4))

        energy = self.evaluator.evaluate(
            cost_op,
            [beta, gamma],
            initial_state=initial_state,
            mixer=mixer,
        )

        self.assertAlmostEqual(energy, baseline, places=6)

    def test_no_initial_state(self):
        """Test that we get the correct energy when the initial state is a product state |111...>."""
        cost_op = SparsePauliOp.from_list([("IIZZ", -0.5), ("ZIIZ", -0.5), ("IZIZ", -0.5)])

        energy = self.evaluator.evaluate(
            cost_op,
            [np.pi / 2, 4.56],  # beta, gamma
            initial_state=QuantumCircuit(4),
        )

        # Prepares the |1111> state which has energy -3/2. Gamma is irrelevant.
        self.assertAlmostEqual(energy, -1.5)

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

        mixer = QuantumCircuit(4)
        mixer.ry(np.pi, 0)
        mixer.rz(2 * Parameter("beta"), range(4))
        mixer.ry(-np.pi, 0)

        energy = self.evaluator.evaluate(cost_op, params, initial_state=init, mixer=mixer)

        # Prepares the |0001> state which has energy 3/2.
        self.assertAlmostEqual(energy, 1.5)
