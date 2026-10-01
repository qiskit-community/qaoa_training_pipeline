#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the efficient depth-one evaluator."""

import numpy as np
from ddt import data, ddt, unpack
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector
from qiskit_aer import Aer

from qaoa_training_pipeline.evaluation.efficient_depth_one import (
    EfficientDepthOneEvaluator,
)
from tests.training_pipeline_test_case import TrainingPipelineTestCase


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
        """Test a few basic points of the efficient depth-one ansatz with a focus on single-Z terms."""

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
        When `nodelist` is not given random behaviour can occur.
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
        self.assertAlmostEqual(float(energy), expected_energy)

    def test_warm_start_aligned_driver(self):
        """
        Test the efficient depth-one with a custom initial state and mixer.
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

        # Mixer so that the initial state is the ground state
        mixer = QuantumCircuit(3)
        beta = Parameter("Beta")

        for j in range(3):
            theta = 0.1 * (j + 1)
            mixer.ry(theta, j)
            mixer.rz(-2 * beta, j)
            mixer.ry(-theta, j)

        qc.compose(mixer.assign_parameters({beta: params[0]}), inplace=True)

        # Compute <psi | cost_op | psi> via statevector simulation
        sv = Statevector(qc)
        expected_energy = sv.expectation_value(cost_op).real

        energy = self.evaluator.evaluate(cost_op, params, initial_state=init, mixer=mixer)
        self.assertAlmostEqual(float(energy), expected_energy)

    @staticmethod
    def random_quadratic_op(num_qubits, density, seed):
        """Create a random operator with weighted ZiZj and Zi terms."""
        rng = np.random.default_rng(seed)
        terms = []
        for i in range(num_qubits):
            if rng.random() < 0.5:
                terms.append(("Z", [i], rng.normal()))
            for j in range(i):
                if rng.random() < density:
                    terms.append(("ZZ", [i, j], rng.normal()))

        return SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits)

    @staticmethod
    def warm_start_circuits(num_qubits, seed):
        """Create a warm-start initial state and the matching per-qubit mixer."""
        thetas = np.random.default_rng(seed).uniform(0, np.pi, num_qubits)
        beta = Parameter("beta")

        init, mixer = QuantumCircuit(num_qubits), QuantumCircuit(num_qubits)
        for j, theta in enumerate(thetas):
            init.ry(theta, j)
            mixer.ry(theta, j)
            mixer.rz(-2 * beta, j)
            mixer.ry(-theta, j)

        return init, mixer

    @staticmethod
    def statevector_energy(cost_op, params, circ_op=None, init=None, mixer=None):
        """Reference energy of a depth-one QAOA from a statevector simulation."""
        num_qubits = cost_op.num_qubits
        circ_op = circ_op if circ_op is not None else cost_op

        qc = QuantumCircuit(num_qubits)
        if init is None:
            qc.h(range(num_qubits))
        else:
            qc.compose(init, inplace=True)

        qc.append(PauliEvolutionGate(circ_op, time=params[1]), range(num_qubits))

        if mixer is None:
            qc.rx(2 * params[0], range(num_qubits))
        else:
            qc.compose(mixer.assign_parameters([params[0]]), inplace=True)

        return Statevector(qc).expectation_value(cost_op).real

    @data((0.0, 0), (0.5, 1), (1.0, 2))
    @unpack
    def test_random_graphs(self, density, seed):
        """Test random weighted graphs, from sparse to fully connected, against a statevector."""
        cost_op = self.random_quadratic_op(7, density, seed)

        for params in ([0.41, 0.34], [-1.3, 2.1]):
            expected = self.statevector_energy(cost_op, params)
            energy = self.evaluator.evaluate(cost_op, params)
            self.assertAlmostEqual(energy, expected, places=8)

    @data(0, 1, 2)
    def test_random_warm_start(self, seed):
        """Test warm-start states with a different mixer on each qubit on random graphs."""
        cost_op = self.random_quadratic_op(6, 0.6, seed)
        init, mixer = self.warm_start_circuits(6, seed)

        for params in ([0.41, 0.34], [1.2, -0.7]):
            expected = self.statevector_energy(cost_op, params, init=init, mixer=mixer)
            energy = self.evaluator.evaluate(cost_op, params, mixer=mixer, initial_state=init)
            self.assertAlmostEqual(energy, expected, places=8)

    def test_warm_start_custom_ansatz(self):
        """Test a warm-start with an ansatz that differs from the cost operator.

        The ansatz has edges that are not in the cost operator and vice versa. It also
        leaves some qubits without any neighbour.
        """
        cost_op = SparsePauliOp.from_sparse_list(
            [("ZZ", [0, 1], 1.0), ("ZZ", [1, 2], -0.7), ("ZZ", [3, 4], 0.4), ("Z", [2], 0.3)],
            num_qubits=5,
        )
        circ_op = SparsePauliOp.from_sparse_list(
            [("ZZ", [0, 1], 0.8), ("ZZ", [0, 2], 1.1), ("Z", [2], -0.5)], num_qubits=5
        )

        gamma = Parameter("g")
        ansatz = QuantumCircuit(5)
        ansatz.rzz(2 * 0.8 * gamma, 0, 1)
        ansatz.rzz(2 * 1.1 * gamma, 0, 2)
        ansatz.rz(2 * -0.5 * gamma, 2)

        init, mixer = self.warm_start_circuits(5, 3)
        params = [0.37, 0.91]

        expected = self.statevector_energy(cost_op, params, circ_op, init, mixer)
        energy = self.evaluator.evaluate(
            cost_op, params, mixer=mixer, initial_state=init, ansatz_circuit=ansatz
        )

        self.assertAlmostEqual(energy, expected, places=8)

    def test_chunking(self):
        """Test that splitting the edges into chunks does not change the energy."""
        cost_op = self.random_quadratic_op(8, 0.7, 4)
        init, mixer = self.warm_start_circuits(8, 4)
        params = [0.29, -0.61]

        expected = EfficientDepthOneEvaluator().evaluate(cost_op, params, mixer, init)

        for max_pairs in [1, 7, 30]:
            evaluator = EfficientDepthOneEvaluator(max_pairs=max_pairs)
            self.assertAlmostEqual(evaluator.evaluate(cost_op, params, mixer, init), expected)

    def test_caching(self):
        """Test that repeated evaluations with new arguments do not reuse stale cached data."""
        cost_op1 = self.random_quadratic_op(5, 0.8, 5)
        cost_op2 = self.random_quadratic_op(5, 0.8, 6)
        init, mixer = self.warm_start_circuits(5, 5)

        cases = [
            (cost_op1, [0.1, 0.2], None, None),
            (cost_op1, [0.3, 0.2], None, None),
            (cost_op1, [0.3, 0.2], mixer, init),
            (cost_op1, [0.5, -0.4], mixer, init),
            (cost_op2, [0.5, -0.4], mixer, init),
            (cost_op2, [0.5, -0.4], None, None),
        ]

        for cost_op, params, mix, init_state in cases:
            expected = self.statevector_energy(cost_op, params, init=init_state, mixer=mix)
            energy = self.evaluator.evaluate(cost_op, params, mixer=mix, initial_state=init_state)
            self.assertAlmostEqual(energy, expected, places=8)

    def test_mismatched_mixer(self):
        """Test that a mixer with the wrong number of qubits raises."""
        cost_op = SparsePauliOp.from_list([("IIZZ", 1.0), ("ZIIZ", 1.0)])

        mixer = QuantumCircuit(3)
        mixer.rx(2 * Parameter("b"), range(3))

        with self.assertRaises(ValueError):
            self.evaluator.evaluate(cost_op, [0.1, 0.2], mixer=mixer)
