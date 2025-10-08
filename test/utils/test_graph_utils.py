#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test converting from a cost operator to a graph."""

from test import TrainingPipelineTestCase

import unittest
from ddt import ddt
import networkx as nx
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.quantum_info import SparsePauliOp
import qiskit_optimization.optionals as _optionals

from qaoa_training_pipeline.utils.graph_utils import (
    operator_to_graph,
    circuit_to_graph,
    graph_to_operator,
    solve_max_cut,
)


@ddt
class TestGraphUtils(TrainingPipelineTestCase):
    """Test methods to manipulate graph, circuits, and cost operators."""

    def test_operator_to_graph(self):
        """Test conversion between operator and graph."""

        cost_op = SparsePauliOp.from_list([("IIZZ", -1), ("IZIZ", 1), ("ZIIZ", 2)])

        graph = operator_to_graph(cost_op)

        self.assertListEqual(
            nx.adjacency_matrix(graph, nodelist=[0, 1, 2, 3]).toarray().flatten().tolist(),
            [0, -1, 1, 2, -1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
        )

    def test_operator_to_graph_single_z(self):
        """Test conversion between operator and graph."""

        cost_op = SparsePauliOp.from_list([("IZZ", -1), ("ZIZ", 1), ("IIZ", 2)])

        graph = operator_to_graph(cost_op)

        self.assertListEqual(
            nx.adjacency_matrix(graph, nodelist=[0, 1, 2]).toarray().flatten().tolist(),
            [2.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )

    def test_raise_on_non_quadratic(self):
        """Test that operator_to_graph raises on non-quadratic terms."""

        cost_op = SparsePauliOp.from_list([("IZZZ", 2), ("IZIZ", 1)])

        with self.assertRaises(ValueError):
            operator_to_graph(cost_op)

    def test_circuit_to_graph(self):
        """Test that we can extract a graph from a QAOA cost op."""

        cost_operator = SparsePauliOp.from_list([("IIZZ", -1), ("ZIIZ", 1), ("IZIZ", 2)])

        cost_layer = qaoa_ansatz(
            cost_operator,
            mixer_operator=QuantumCircuit(4),
            initial_state=QuantumCircuit(4),
            reps=1,
        ).decompose()

        pass_manager = PassManager([HighLevelSynthesis(basis_gates=["rzz"])])
        cost_layer = pass_manager.run(cost_layer)
        graph = circuit_to_graph(cost_layer)

        self.assertListEqual(
            nx.adjacency_matrix(graph, nodelist=[0, 1, 2, 3]).toarray().flatten().tolist(),
            [0, -1, 2, 1, -1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0],
        )

    def test_circuit_to_graph_single_z(self):
        """Test that we can extract a graph from a QAOA cost op."""

        cost_operator = SparsePauliOp.from_list([("IZZ", -1), ("IIZ", 1), ("ZIZ", 2)])

        cost_layer = qaoa_ansatz(
            cost_operator,
            mixer_operator=QuantumCircuit(3),
            initial_state=QuantumCircuit(3),
            reps=1,
        ).decompose()

        pass_manager = PassManager([HighLevelSynthesis(basis_gates=["rzz", "rz"])])
        cost_layer = pass_manager.run(cost_layer)
        graph = circuit_to_graph(cost_layer)

        self.assertListEqual(
            nx.adjacency_matrix(graph, nodelist=[0, 1, 2]).toarray().flatten().tolist(),
            [1, -1, 2, -1, 0, 0, 2, 0, 0],
        )

    def test_circuit_to_graph_idle(self):
        """Test that idle qubits are properly accounted for."""

        circuit = QuantumCircuit(3)
        gamma = Parameter("g")
        circuit.rzz(2 * gamma, 0, 2)

        graph = circuit_to_graph(circuit)

        self.assertEqual(graph.order(), 3)

        self.assertListEqual(list(graph.edges()), [(0, 2)])

    def test_circuit_to_graph_errors(self):
        """Test that circuit to graph raises as intended."""

        # Test raise on repeated Rzz.
        circuit = QuantumCircuit(2)
        gamma = Parameter("g")
        circuit.rzz(2 * gamma, 0, 1)
        circuit.rzz(2 * gamma, 0, 1)

        with self.assertRaises(ValueError):
            circuit_to_graph(circuit)

        # Same as before ut ensuring symmetry is not an issue.
        circuit = QuantumCircuit(2)
        circuit.rzz(2 * gamma, 0, 1)
        circuit.rzz(2 * gamma, 1, 0)

        with self.assertRaises(ValueError):
            circuit_to_graph(circuit)

        # Test raise on non-Rzz instructions.
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])

        with self.assertRaises(ValueError):
            circuit_to_graph(circuit)

        # Test raise when parameter is absent.
        circuit = QuantumCircuit(2)
        circuit.rzz(2, 0, 1)

        with self.assertRaises(ValueError):
            circuit_to_graph(circuit)

    def test_graph_to_operator(self):
        """Test the conversion of an unweighted graph to Pauli operator."""
        graph = nx.from_edgelist([(0, 1), (0, 2), (0, 3)])

        paulis = graph_to_operator(graph)
        expected = SparsePauliOp.from_list([("IIZZ", 1.0), ("IZIZ", 1.0), ("ZIIZ", 1.0)])

        self.assertEqual(paulis.paulis, expected.paulis)

    def test_graph_to_operator_single_z(self):
        """Test the conversion of an unweighted graph to Pauli operator."""
        graph = nx.from_edgelist([(0, 1), (0, 2), (0, 0)])

        paulis = graph_to_operator(graph)
        expected = SparsePauliOp.from_list([("IZZ", 1.0), ("ZIZ", 1.0), ("IIZ", 1.0)])

        self.assertEqual(paulis.paulis, expected.paulis)

    def test_weighted_graph_to_operator(self):
        """Test the conversion from a weighted graph to an operator."""
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -1), (0, 2, 0.12), (0, 3, 1)])
        paulis = graph_to_operator(graph)
        expected = SparsePauliOp.from_list([("IIZZ", -1.0), ("IZIZ", 0.12), ("ZIIZ", 1.0)])

        self.assertEqual(paulis.paulis, expected.paulis)
        self.assertTrue(np.allclose(paulis.coeffs, expected.coeffs))

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_max_cut_solutions(self):
        """Test that we can properly solve for the maximum cut."""

        # Graph [(0, 1, w=1), (1, 2, w=1)]
        cost_op = SparsePauliOp.from_list([("IZZ", -0.5), ("ZZI", -0.5)])

        max_cut, min_cut, approx = solve_max_cut(cost_op)

        self.assertEqual(max_cut, 2)
        self.assertEqual(min_cut, 0)
        self.assertIsNone(approx)

        # Graph [(0, 1, w=-1), (1, 2, w=1)]
        cost_op = SparsePauliOp.from_list([("IZZ", 0.5), ("ZZI", -0.5)])

        max_cut, min_cut, approx = solve_max_cut(cost_op, energy=1.0)

        self.assertEqual(max_cut, 1)
        self.assertEqual(min_cut, -1)
        self.assertEqual(approx, 1)
