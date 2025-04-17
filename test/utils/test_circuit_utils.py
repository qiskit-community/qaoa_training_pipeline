#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit manipulation methods."""

from test import TrainingPipelineTestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qaoa_training_pipeline.utils.circuit_utils import split_circuit


class TestCircuitUtils(TrainingPipelineTestCase):
    """Class to test circuit utils."""

    def test_split_circuit(self):
        """Test the split of quantum circuits."""

        circ = QuantumCircuit(4)
        circ.h(0)
        circ.rx(-0.4, 1)
        circ.ry(-0.2, 2)
        circ.rz(2 * 1.1, 2)
        circ.ry(0.2, 2)

        splits = split_circuit(circ)

        self.assertEqual(len(splits), 4)

        for sub_circ in splits:
            self.assertEqual(sub_circ.num_qubits, 1)

        expected = QuantumCircuit(1)
        expected.ry(-0.2, 0)
        expected.rz(2 * 1.1, 0)
        expected.ry(0.2, 0)

        self.assertEqual(splits[2], expected)
        self.assertTrue(
            np.allclose(
                Operator(splits[3]).data,
                np.eye(2),
            )
        )

    def test_split_circuit_raise(self):
        """Test that we raise on multi-qubit gate."""
        circ = QuantumCircuit(2)
        circ.cx(0, 1)

        with self.assertRaises(ValueError):
            split_circuit(circ)
