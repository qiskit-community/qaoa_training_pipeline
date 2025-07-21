#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the Matrix Product State evaluator."""

from test import TrainingPipelineTestCase

import os
from ddt import data, ddt, unpack

import networkx as nx
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qaoa_training_pipeline.evaluation import MPSEvaluator, EfficientDepthOneEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator, load_graph


@ddt
class TestMPSEvaluator(TrainingPipelineTestCase):
    """Tests for the matrix product state energy evaluation."""

    def setUp(self):
        """Initialize the variables we need."""
        # The light-cone only uses QASM. We therefore set a tolerance with
        # respect to the ideal value.
        self.tolerance = 0.0001
        # Get the directory of the current test file
        self.test_dir = os.path.dirname(__file__)

    @staticmethod
    def construct_labs_sparse_pauli_op(n_qubits: int) -> SparsePauliOp:
        """Construct the sparse Pauli operation representation of a LABS problem.

        Args:
            n_qubits (int): number of qubits associated with the problem

        Returns:
            SparsePauliOp: cost operator
        """

        reference_string = "I" * n_qubits
        list_of_terms = []
        list_of_coeffs = []

        # Two-body part
        for idxi in range(1, n_qubits - 1):
            for j in range(1, int(np.floor((n_qubits - idxi) / 2)) + 1):
                tmp_string = list(reference_string)
                tmp_string[idxi - 1] = "Z"
                tmp_string[idxi + 2 * j - 1] = "Z"
                tmp_string.reverse()
                list_of_terms.append("".join(tmp_string))
                list_of_coeffs.append(1.0)

        # Four-body part
        for idxi in range(1, n_qubits - 2):
            for idxt in range(1, int(np.floor((n_qubits - idxi - 1) / 2)) - 1):
                for idxk in range(idxt + 1, n_qubits - idxt - idxi - 1):
                    tmp_string = list(reference_string)
                    tmp_string[idxi - 1] = "Z"
                    tmp_string[idxi + idxt - 1] = "Z"
                    tmp_string[idxi + idxk - 1] = "Z"
                    tmp_string[idxi + idxt + idxk - 1] = "Z"
                    tmp_string.reverse()
                    list_of_terms.append("".join(tmp_string))
                    list_of_coeffs.append(2.0)

        return SparsePauliOp(list_of_terms, list_of_coeffs)

    def qiskit_circuit_simulation(self, cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = QAOAAnsatz(cost_op, reps=len(params) // 2)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        ansatz = transpile(ansatz, basis_gates=["cx", "sx", "x", "rz"])
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return result[0].data.evs

    @data((0, 0), (1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_depth_one(self, beta, gamma):
        """Perform tests for a depth one QAOA unweighted graph."""
        graph = nx.random_regular_graph(d=3, n=12, seed=0)
        cost_op = graph_to_operator(graph)

        # Here we don't pass any argument to `MPSEvaluator` because we don't want
        # to truncate anything and keep the simulation exact (feasible for such
        # a simple circuit)
        energy_mps = MPSEvaluator().evaluate(cost_op, [beta, gamma])
        energy_cs = self.qiskit_circuit_simulation(cost_op, [beta, gamma])

        # The MPS simulator is exact, but for the reference data there is shot noise
        self.assertTrue(abs(energy_cs - energy_mps) < self.tolerance)

    @data((0, 0, 0, 0), (1, 2, 3, 4))
    @unpack
    def test_depth_two(self, beta0, beta1, gamma0, gamma1):
        """Same as above, but for a depth-2 circuit."""
        params = [beta0, beta1, gamma0, gamma1]
        cost_op = SparsePauliOp.from_list([("ZIIZ", 0.123), ("IZIZ", -1), ("IIZZ", 2)])

        evaluator = MPSEvaluator()
        energy_mps = evaluator.evaluate(cost_op, params)
        expected = self.qiskit_circuit_simulation(cost_op, params)
        self.assertTrue(abs(energy_mps - expected) < self.tolerance)

        additional_results = evaluator.get_results_from_last_iteration()
        self.assertTrue(all(0 < i < 5 for i in additional_results["circuit_bond_dimension"]))

    @data(
        (0.1, 0.2, -0.1, -0.2, True, True),
        (-0.1, 0.2, 0.14, 0.5, True, False),
        (0.3, 0.1, 0.12, -0.1, False, False),
        (-0.1, 0.12, -0.4, 0.0, False, True),
    )
    @unpack
    def test_depth_two_fidelity_bounds(self, beta0, beta1, gamma0, gamma1, vidal, swap_strategy):
        """Checks that MPS simualtions respect the bounds on the fidelity."""
        params = [beta0, beta1, gamma0, gamma1]
        cost_op = SparsePauliOp.from_list([("ZIIZ", 0.123), ("IZIZ", -1), ("IIZZ", 2)])

        evaluator = MPSEvaluator(bond_dim_circuit=2, store_intermediate_schmidt_values=True)
        _ = evaluator.evaluate(cost_op, params)

        # Gets the bounds
        fidelity_bound = evaluator.calculate_fidelity_bounds()
        self.assertLess(fidelity_bound, 1)
        self.assertGreater(fidelity_bound, 0)

        # Gets the fidelity approximation
        fidelity_approx = evaluator.calculate_fidelity_approximation()
        self.assertLess(fidelity_approx, 1)
        self.assertGreater(fidelity_approx, 0)

    @data((1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_ansatz_circuit(self, beta, gamma):
        """Test that we can run the evaluator with a custom ansatz circuit."""
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

        # Construct an ansatz.
        gamma_param = Parameter("gamma")
        ansatz = QuantumCircuit(4)
        ansatz.rzz(2 * gamma_param, 3, 0)
        ansatz.rzz(2 * gamma_param, 2, 1)

        # Construct the QAOA circuit corresponding to the ansatz.
        qaoa_circuit = QAOAAnsatz(SparsePauliOp.from_list([("ZIIZ", 1), ("IZZI", 1)]))
        qaoa_circuit = transpile(qaoa_circuit, basis_gates=["rzz", "h", "rx", "rz"])

        estimator = StatevectorEstimator()
        expected = float(
            estimator.run([(qaoa_circuit, cost_op, [beta, gamma])]).result()[0].data.evs
        )

        actual = MPSEvaluator().evaluate(
            cost_op,
            params=[beta, gamma],
            ansatz_circuit=ansatz,
        )

        self.assertAlmostEqual(actual, expected, places=8)

    @data((True, False), (False, True))
    @unpack
    def test_custom_mixer_and_initial_state(self, vidal: bool, store_schmidt_values: bool):
        """Test that the MPS correctly evaluates with custom initial states and mixers."""
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

        energy = MPSEvaluator(
            use_vidal_form=vidal, store_schmidt_values=store_schmidt_values
        ).evaluate(
            cost_op,
            params,
            initial_state=init,
            mixer=mixer,
        )

        # Prepares the |0001> state which has energy 3/2.
        self.assertAlmostEqual(energy, 1.5, places=8)

        # Test raise on multiple parameters
        mixer = QuantumCircuit(4)
        for idx in range(4):
            mixer.rx(2 * Parameter(f"beta{idx}"), idx)

        with self.assertRaises(ValueError):
            MPSEvaluator(use_vidal_form=vidal).evaluate(cost_op, params, mixer=mixer)

    @data((False, False, True), (True, False, True), (False, True, False), (True, True, False))
    @unpack
    def test_single_z(self, vidal, swap_strat, store_schmidt):
        """Tests that cost operators with single-z terms are properly evaluated."""
        cost_op = SparsePauliOp.from_list([("ZZ", -1), ("IZ", 1), ("ZI", 0.5)])

        params = [1.2, 2.3]
        evaluator = MPSEvaluator(
            use_vidal_form=vidal, use_swap_strategy=swap_strat, store_schmidt_values=store_schmidt
        )
        actual = evaluator.evaluate(cost_op, params)
        expected = self.qiskit_circuit_simulation(cost_op, params)

        self.assertAlmostEqual(actual, expected)

    def test_hobo_on_labs_problem_vs_qaoa(self):
        """Checks HOBO MPS vs statevector on a LABS problem.
        Unlike the previous test, here we use the `QAOAAnsatz` qiskit-native class"""

        # Generates the LABS problem
        n_qubits = 8
        sparse_pauli_labs = TestMPSEvaluator.construct_labs_sparse_pauli_op(n_qubits)

        # Evaluator with MPS
        list_of_params = [0.2, 0.3]
        mps_evaluator = MPSEvaluator(use_vidal_form=False)
        energy_from_mps = mps_evaluator.evaluate(sparse_pauli_labs, list_of_params)

        # Statevector simulator via qiskit circuit
        qaoa_circuit = QAOAAnsatz(sparse_pauli_labs, reps=1)
        actual_circuit = qaoa_circuit.assign_parameters([0.2, 0.3])
        estimator = StatevectorEstimator()
        result = estimator.run([(actual_circuit, sparse_pauli_labs, [])]).result()
        energy_from_sv = result[0].data.evs
        self.assertAlmostEqual(energy_from_mps, energy_from_sv)

    def test_from_config(self):
        """Test that we can instantiate from a config."""
        config = {
            "use_vidal_form": False,
            "threshold_circuit": 0.1,
            "bond_dim_circuit": 2,
            "threshold_mpo": 0.5,
            "bond_dim_mpo": 2,
        }

        evaluator = MPSEvaluator.from_config(config)

        self.assertTrue(isinstance(evaluator, MPSEvaluator))

    def test_evaluate_from_file(self):
        """Checks the training starting from a graph stored in a file"""

        # Retrieves the graph
        data_dir = self.test_dir + "/../data/"
        graph_file = "000000_12nodes_minus_plus_weighted.json"
        graph = load_graph(data_dir + graph_file)
        cost_op = graph_to_operator(graph, pre_factor=-0.5)

        # Proceeds to the evaluation
        evaluator_vidal_m_128 = MPSEvaluator(bond_dim_circuit=128, use_vidal_form=True)
        energy_vidal_m_128 = evaluator_vidal_m_128.evaluate(cost_op, params=[1, 2])

        energy_edo = EfficientDepthOneEvaluator().evaluate(cost_op, params=[1, 2])

        self.assertAlmostEqual(energy_vidal_m_128, energy_edo, 5)

    def test_config_from_init_str(self):
        """Test creating an instance from a str."""
        init_str = "1_None_10_None_10_False_False"
        config = MPSEvaluator.parse_init_kwargs(init_str)

        evaluator = MPSEvaluator.from_config(config)

        self.assertTrue(isinstance(evaluator, MPSEvaluator))


@ddt
class TestMPSEvaluatorSWAPs(TrainingPipelineTestCase):
    """Test the swap strategies True option."""

    @data((1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_swap_strategy(self, beta: float, gamma: float):
        """Test swap strategy for depth one QAOA."""
        cost_op = SparsePauliOp.from_list(
            [
                ("IIZZ", 1),
                ("IZIZ", 2),
                ("ZIIZ", 3),
                ("IZZI", 4),
                ("ZIZI", 5),
                ("ZZII", 6),
            ]
        )

        evaluator1 = MPSEvaluator()
        evaluator2 = MPSEvaluator(use_swap_strategy=True)

        energy1 = evaluator1.evaluate(cost_op, [beta, gamma])
        energy2 = evaluator2.evaluate(cost_op, [beta, gamma])

        self.assertTrue(evaluator1.swap_strategy is None)
        self.assertTrue(isinstance(evaluator2.swap_strategy, SwapStrategy))
        self.assertAlmostEqual(energy1, energy2, places=8)

    @data(
        (0, 0, 0, 0, False),
        (1, 2, 3, 4, False),
        (0.5, 0.25, 0.4, 0.25, False),
        (0.5, 0.25, 0.4, 0.25, True),
    )
    @unpack
    # pylint: disable=too-many-positional-arguments
    def test_depth_two_swap_strat(self, beta0, beta1, gamma0, gamma1, vidal):
        """Test swap strategy for depth two QAOA."""
        cost_op = SparsePauliOp.from_list(
            [
                ("IIZZ", 1),
                ("IZIZ", 2),
                ("ZIIZ", 3),
                ("IZZI", 4),
                ("ZIZI", 5),
                ("ZZII", 6),
            ]
        )

        evaluator1 = MPSEvaluator(use_vidal_form=vidal)
        evaluator2 = MPSEvaluator(use_vidal_form=vidal, use_swap_strategy=True)

        energy1 = evaluator1.evaluate(cost_op, [beta0, beta1, gamma0, gamma1])
        energy2 = evaluator2.evaluate(cost_op, [beta0, beta1, gamma0, gamma1])

        self.assertTrue(evaluator1.swap_strategy is None)
        self.assertTrue(isinstance(evaluator2.swap_strategy, SwapStrategy))
        self.assertAlmostEqual(energy1, energy2, places=8)

    @data(
        ("ZIIZ", False),
        ("ZIIIZ", False),
        ("ZIIIIZ", False),
        ("ZIIIIIZ", False),
        ("ZIIIZI", False),
        ("ZIIZ", True),
        ("ZIIIZ", True),
        ("ZIZIII", True),
    )
    @unpack
    def test_correlators(self, pauli_str, use_vidal):
        """Test varying correlators."""
        cost_op = SparsePauliOp.from_list([(pauli_str, 1)])

        evaluator1 = MPSEvaluator(use_vidal_form=use_vidal)
        evaluator2 = MPSEvaluator(use_vidal_form=use_vidal, use_swap_strategy=True)

        beta, gamma = 1, 0.35
        energy1 = evaluator1.evaluate(cost_op, [beta, gamma])
        energy2 = evaluator2.evaluate(cost_op, [beta, gamma])

        self.assertTrue(evaluator1.swap_strategy is None)
        self.assertTrue(isinstance(evaluator2.swap_strategy, SwapStrategy))
        self.assertAlmostEqual(energy1, energy2, places=8)

    @data(
        ["ZIIZ", "ZZZZ"],
        ["ZIIIZ", "ZIZIZ"],
        ["IIZIIZ", "ZIIZIZ"],
        ["ZIIIIIZ", "ZIIZIIZ"],
    )
    def test_correlators_hobo(self, pauli_list):
        """Test varying correlators with Pauli weight > 2."""
        cost_op = SparsePauliOp.from_list([(i, 1) for i in pauli_list])

        evaluator = MPSEvaluator(use_vidal_form=False)
        evaluator_vidal = MPSEvaluator(use_vidal_form=True)

        beta, gamma = 1, 0.35
        energy = evaluator.evaluate(cost_op, [beta, gamma])
        energy_vidal = evaluator_vidal.evaluate(cost_op, [beta, gamma])

        self.assertAlmostEqual(energy, energy_vidal, places=8)

    def test_swap_strategy_hobo_raises_error(self):
        """Test that, if a SWAP strategy is requested for HOBO problems,
        an exception is raised.
        """
        cost_op = SparsePauliOp.from_list(
            [
                ("IIZZ", 1),
                ("IZZZ", 2),
            ]
        )

        evaluator = MPSEvaluator(use_swap_strategy=True)

        with self.assertRaises(NotImplementedError):
            _ = evaluator.evaluate(cost_op, [1.0, 2.0])
