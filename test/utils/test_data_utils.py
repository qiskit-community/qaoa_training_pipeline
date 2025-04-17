#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test functions that help manipulating data."""

from unittest import TestCase
import networkx as nx

from qiskit.quantum_info import SparsePauliOp

from qiskit_optimization.applications import Maxcut

from qaoa_training_pipeline.utils.data_utils import (
    samples_to_objective_values,
    input_to_operator,
)


class TestDataUtils(TestCase):
    """Test the functions in `data_utils.py`."""

    def test_samples_to_objective_values(self):
        """Test the conversion from samples to objective values."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (0, 3)])

        max_cut = Maxcut(graph)
        program = max_cut.to_quadratic_program()

        samples = {"1110": 10, "0000": 1}

        func_vals = samples_to_objective_values(samples, program)

        self.assertDictEqual(func_vals, {3.0: 10.0, 0.0: 1.0})

    def test_input_to_cost_operator(self):
        """Test that we can create cost operators."""

        # Standard QUBO test.
        input_qubo = {
            "edge list": [
                {"nodes": [0, 1], "weight": 1.0},
                {"nodes": [0, 2], "weight": 1.0},
            ]
        }

        expected = SparsePauliOp.from_list([("IZZ", 1), ("ZIZ", 1)])
        self.assertEqual(input_to_operator(input_qubo), expected)

        # Test the pre-factor.
        expected = SparsePauliOp.from_list([("IZZ", -0.5), ("ZIZ", -0.5)])
        self.assertEqual(input_to_operator(input_qubo, -0.5), expected)

        # Test a HOBO
        input_hobo = {
            "edge list": [
                {"nodes": [0, 1, 2], "weight": -1.0},
                {"nodes": [0, 2], "weight": 1.0},
            ]
        }

        expected = SparsePauliOp.from_list([("ZZZ", -1), ("ZIZ", 1)])
        self.assertEqual(input_to_operator(input_hobo), expected)
