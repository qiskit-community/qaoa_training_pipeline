#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of problem classes."""

import numpy as np
from test import TrainingPipelineTestCase

from ddt import ddt, data, unpack

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.problem_classes import LABS, PROBLEM_CLASSES


@ddt
class TestProblemClasses(TrainingPipelineTestCase):
    """Test the problem classes."""

    @unpack
    @data(("maxcut", None), ("mis", None), ("mis", [3.0]))
    def test_cost_operator(self, class_name: str, init_args):
        """Test that the problem classes give the right cost operators."""

        input_graph = {
            "edge list": [{"nodes": (0, 1), "weight": 1}, {"nodes": (0, 2), "weight": 1}]
        }

        op1 = [("IZZ", -0.5), ("ZIZ", -0.5)]
        op2 = [("IIZ", -0.5), ("IZZ", 0.5), ("ZIZ", 0.5)]
        op3 = [("IIZ", -1.0), ("IZI", -0.25), ("ZII", -0.25), ("IZZ", 0.75), ("ZIZ", 0.75)]

        expected = {
            ("maxcut", None): SparsePauliOp.from_list(op1),
            ("mis", None): SparsePauliOp.from_list(op2),
            ("mis", 3.0): SparsePauliOp.from_list(op3),
        }

        if init_args is not None:
            problem_class = PROBLEM_CLASSES[class_name](*init_args)
            key = init_args[0]
        else:
            problem_class = PROBLEM_CLASSES[class_name]()
            key = None

        cost_op = problem_class.cost_operator(input_graph)

        self.assertEqual(cost_op, expected[(class_name, key)])

    def test_labs_cost_operator_n4(self):
        """Test LABS cost operator for N=4."""
        labs = LABS(4)
        cost_op = labs.cost_operator()

        # N=4: offset=6, two ZZ terms and one ZZZZ term
        expected_terms = [("IIII", 6), ("IZIZ", 2), ("ZIZI", 2), ("ZZZZ", 4)]
        expected = SparsePauliOp.from_list(expected_terms)

        self.assertEqual(cost_op, expected)

    def test_labs_from_str(self):
        """Test LABS.from_str method."""
        # Valid string
        labs1 = LABS.from_str("5")
        self.assertEqual(labs1._num_qubits, 5)

        # Empty string (should use default)
        labs2 = LABS.from_str("")
        self.assertEqual(labs2._num_qubits, LABS.DEFAULT_N)

        # Invalid string (should use default)
        labs3 = LABS.from_str("invalid")
        self.assertEqual(labs3._num_qubits, LABS.DEFAULT_N)

        # None (should use default)
        labs4 = LABS.from_str(None)
        self.assertEqual(labs4._num_qubits, LABS.DEFAULT_N)

    def test_labs_energy_consistency(self):
        """Test that energy computed from spins matches Hamiltonian expectation."""
        labs = LABS(3)
        cost_op = labs.cost_operator()

        # Test with spins = [1, -1, 1]
        spins = np.array([1, -1, 1])
        energy_direct = LABS.compute_energy(spins)

        # Compute expectation value manually from Hamiltonian
        # H = 3*III + 2*ZIZ
        # For spins [1, -1, 1]: ZIZ expectation = spins[0] * spins[2] = 1 * 1 = 1
        # Energy = 3 + 2*1 = 5
        energy_hamiltonian = 3 + 2 * (spins[0] * spins[2])

        self.assertAlmostEqual(energy_direct, energy_hamiltonian, places=10)
