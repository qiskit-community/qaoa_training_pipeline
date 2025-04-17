#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the light-cone evaluator."""

from typing import Dict
from ddt import ddt, data, unpack
import networkx as nx

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import StatevectorEstimator

from qaoa_training_pipeline.evaluation.light_cone import LightConeEvaluator
from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator

from test import TrainingPipelineTestCase


@ddt
class TestLightConeEvaluator(TrainingPipelineTestCase):
    """Tests for the light-cone energy evaluation."""

    def setUp(self):
        """Initialize the variables we need."""
        self.evaluator = LightConeEvaluator(shots=2**14)

        # The light-cone only uses QASM. We therefore set a tolerance with
        # respect to the ideal value.
        self.tolerance = 1e-6
        self.primitive = StatevectorEstimator()

    def qiskit_circuit_simulation(self, cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = QAOAAnsatz(cost_op, reps=len(params) // 2)

        ansatz.assign_parameters(params, inplace=True)
        ansatz = transpile(ansatz, basis_gates=["cx", "sx", "x", "rz"])

        result = self.primitive.run([(ansatz, cost_op)]).result()

        return float(result[0].data.evs)

    @staticmethod
    def expectation_value(counts: Dict[str, int], observable: Pauli) -> float:
        """Compute the expectation value of the given observable."""
        mask = [1 if char == "I" else -1 for char in observable.to_label()]

        exp_val, shots = 0.0, sum(counts.values())

        for bit_str, count in counts.items():
            sign = 1
            for idx, bit in enumerate(bit_str):
                if bit == "1":
                    sign *= mask[idx]

            exp_val += sign * count / shots

        return exp_val

    @data((0, 0), (1, 1), (0.1234, -0.56), (0.25, 0.5), (0.5, 0.25))
    @unpack
    def test_depth_one(self, beta, gamma):
        """Perform tests for a depth one QAOA unweighted graph."""
        graph = nx.random_regular_graph(d=3, n=12, seed=0)

        cost_op = graph_to_operator(graph)

        energy_lc = self.evaluator.evaluate(cost_op, [beta, gamma])
        energy_ed = EfficientDepthOneEvaluator().evaluate(cost_op, [beta, gamma])
        energy_cs = self.qiskit_circuit_simulation(cost_op, [beta, gamma])

        # The light-cone is shot based but the efficient depth-one is numerically exact.
        self.assertTrue(abs(energy_lc - energy_ed) < self.tolerance)

        # Both methods are shot-based and the variations can be larger.
        self.assertTrue(abs(energy_lc - energy_cs) < 2 * self.tolerance)

    @data((0, 0, 0, 0), (1, 2, 3, 4))
    @unpack
    def test_depth_two(self, beta0, beta1, gamma0, gamma1):
        """Perform tests for a depth two QAOA on a weighted graph."""
        params = [beta0, beta1, gamma0, gamma1]

        cost_op = SparsePauliOp.from_list([("ZIIZ", 0.123), ("IZIZ", -1), ("IIZZ", 2)])

        energy_lc = self.evaluator.evaluate(cost_op, params)
        expected = self.qiskit_circuit_simulation(cost_op, params)

        self.assertTrue(abs(energy_lc - expected) < self.tolerance)

    @data((1, 2, 3, 4), (-1, 0.5, 1.57, -0.9))
    @unpack
    def test_large_scale_energy_computation(self, beta0, beta1, gamma0, gamma1):
        """Here we compute the energy with light-cone and brute force."""

        # We use 18 nodes to exceed the size of 14 qubits in the p=2 light-cone
        graph = nx.random_regular_graph(d=3, n=18, seed=0)

        cost_op = graph_to_operator(graph)

        params = [beta0, beta1, gamma0, gamma1]
        energy_lc = self.evaluator.evaluate(cost_op, params)
        expected = self.qiskit_circuit_simulation(cost_op, params)

        self.assertTrue(abs(energy_lc - expected) < 2 * self.tolerance)

    def test_large_scale_circuits(self):
        """Test that a depth-two instance on a RR3 graph has a limited size."""
        graph = nx.random_regular_graph(d=3, n=40, seed=0)

        self.evaluator.graph = graph

        for edge in graph.edges:
            circ, _ = self.evaluator.make_radius_circuit(edge, [1, 2, 3, 4])

            # Depth-two light cone on an RR3 graph has at most 14 qubits.
            self.assertTrue(circ.num_qubits <= 14)

    def test_from_config(self):
        """Test that we can instantiate from a config."""
        config = {"shots": 100}

        evaluator = LightConeEvaluator.from_config(config)

        self.assertTrue(isinstance(evaluator, LightConeEvaluator))
