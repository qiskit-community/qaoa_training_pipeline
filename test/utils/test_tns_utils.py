#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the TNS utilities."""

from test import TrainingPipelineTestCase

from typing import List, Tuple

import random
from itertools import product
from math import sqrt
from cmath import exp as cexp
from ddt import ddt, data

import networkx as nx

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli

from quimb.tensor import CircuitMPS, MPS_computational_state

from qaoa_training_pipeline.utils.graph_utils import graph_to_operator
from qaoa_training_pipeline.utils.tns_utils.circuit_mps_vidal import (
    CircuitMPSVidalCanonization,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_circuit_mps import (
    QAOACircuitMPSRepresentation,
    QAOACircuitVidalRepresentation,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_cost_function import QAOACostFunction
from qaoa_training_pipeline.utils.tns_utils.multi_qubit_gates import QAOAManyBodyCorrelator
from qaoa_training_pipeline.utils.tns_utils.symbolic_mpo import SymbolicMPOConstruction

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator


class TestQAOAMPS(TrainingPipelineTestCase):
    """Test methods to get MPS representations of QAOA problems."""

    def generate_adjacency_matrix_linear(self, n_qubits: int) -> np.array:
        """Helper function to generate linear-connectivity adjacency matrices"""
        adjacency_matrix = np.zeros((n_qubits, n_qubits), dtype=int)
        for i_qubit in range(n_qubits - 1):
            adjacency_matrix[i_qubit, i_qubit + 1] = 1
            adjacency_matrix[i_qubit + 1, i_qubit] = 1
        return adjacency_matrix

    def test_trivial_constructor_and_getter(self):
        """Test constructor and trivial getters."""
        n_qubits = 10
        adjacency_matrix = self.generate_adjacency_matrix_linear(n_qubits)
        mps = QAOACircuitMPSRepresentation(n_qubits, adjacency_matrix)
        self.assertEqual(mps.n_qubits, n_qubits)

    def test_construction_from_graph(self):
        """Test constructor from a networkx graph"""
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -0.1), (1, 2, 1.2), (2, 3, 1.4)])
        mps_qaoa = QAOACircuitMPSRepresentation.construct_from_graph(graph)
        self.assertEqual(mps_qaoa.n_qubits, 4)
        underlying_mps = mps_qaoa.get_underlying_tn()
        self.assertEqual(underlying_mps.nsites, 4)

    def test_schmidt_values(self):
        """Checks coherence in the calculation of the Schmidt values"""
        n_qubits = 12
        adjacency_matrix = self.generate_adjacency_matrix_linear(n_qubits)

        # Conventional simulator
        qaoa_mps = QAOACircuitMPSRepresentation(n_qubits, adjacency_matrix)
        qaoa_mps.apply_qaoa_layer([0.1, 0.2], [0.3, 0.4])
        schmidt_values_conventional = []
        for i_site in range(n_qubits - 1):
            schmidt_values = qaoa_mps.get_schmidt_values(i_site)
            schmidt_values_conventional.append(schmidt_values)
            for i in schmidt_values:
                self.assertLess(i, 1.0)

        # Vidal simulator
        qaoa_mps_vidal = QAOACircuitVidalRepresentation(n_qubits, adjacency_matrix)
        qaoa_mps_vidal.apply_qaoa_layer([0.1, 0.2], [0.3, 0.4])
        schmidt_values_vidal = []
        for i_site in range(n_qubits - 1):
            schmidt_values = qaoa_mps_vidal.get_schmidt_values(i_site)
            schmidt_values_vidal.append(schmidt_values)
            for i in schmidt_values:
                self.assertLess(i, 1.0)

        # Checks inter-representation coherence
        for i_conventional, i_vidal in zip(schmidt_values_conventional, schmidt_values_vidal):
            for i, j in zip(i_conventional, i_vidal):
                self.assertAlmostEqual(i, j)

    def test_application_trivial_gates(self):
        """Tests that applying a set of trivial gates does not change the underlying MPS"""
        n_qubits = 20
        adjacency_matrix = self.generate_adjacency_matrix_linear(n_qubits)
        qaoa_mps_one_layer = QAOACircuitMPSRepresentation(n_qubits, adjacency_matrix)
        qaoa_mps_one_layer.apply_qaoa_layer([0.0], [0.0])
        qaoa_mps_two_layers = QAOACircuitMPSRepresentation(n_qubits, adjacency_matrix)
        qaoa_mps_two_layers.apply_qaoa_layer([0.0], [0.0])
        tn_overlap = (
            qaoa_mps_two_layers.get_underlying_tn().H & qaoa_mps_one_layer.get_underlying_tn()
        )
        self.assertAlmostEqual(tn_overlap ^ all, 1.0)

    def test_application_non_trivial_gates(self):
        """Checks that normalization is preserved if non-trivial gates are applied"""
        graph = nx.Graph()
        graph.add_weighted_edges_from([(i, i + 1, 0.1 * (i + 1)) for i in range(19)])
        qaoa_mps = QAOACircuitMPSRepresentation.construct_from_graph(graph, 1.0e-10)
        qaoa_mps.apply_qaoa_layer([0.2, 0.4], [0.6, 0.8])
        self.assertAlmostEqual(qaoa_mps.get_underlying_tn().norm(), 1.0)


@ddt
class TestMultiQubitGates(TrainingPipelineTestCase):
    """Tests on the functionalities to represent multi-qubit gates"""

    def test_trivial_constructor_multi_qubit_gate(self):
        """Basic test on class constructor"""
        n_qubits = 4
        multi_qubit_gate = QAOAManyBodyCorrelator([0, 1, 2], n_qubits, 0.1)
        self.assertEqual(multi_qubit_gate.n_qubits, 4)

    def test_dense_representation_of_gate(self):
        """Test that the dense representation of a gate does not depend on the
        specific qubits it acts on.
        """
        n_qubits = 4
        multi_qubit_gate_1 = QAOAManyBodyCorrelator([0, 1, 2], n_qubits, 0.1)
        multi_qubit_gate_2 = QAOAManyBodyCorrelator([0, 1, 3], n_qubits, 0.1)
        dense_1 = multi_qubit_gate_1.get_dense_representation(1.0)
        dense_2 = multi_qubit_gate_2.get_dense_representation(1.0)
        self.assertAlmostEqual(np.linalg.norm(dense_1 - dense_2), 1.0e-16)

        # Now also does some checks on the values of the tensor
        for i in range(2):
            # Diagonal term
            self.assertAlmostEqual(np.abs(dense_1[i, i, i, i, i, i, 0]), 1.0)
            # Off-diagonal term
            j = 0 if i == 1 else 1
            self.assertAlmostEqual(dense_1[i, j, i, j, i, j, 0], 0)
            self.assertAlmostEqual(dense_2[i, j, i, j, i, j, 0], 0)

    @data([0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3])
    def test_mps_representation_of_gate(self, list_of_qubits):
        """Test that the MPS representation of the gates is coherent with the
        dense one.
        """

        n_qubits = 4
        time_step = 0.1
        reindex_dict_b = {"k" + str(i): "b" + str(i) for i in range(n_qubits)}
        multi_qubit_gate = QAOAManyBodyCorrelator(list_of_qubits, n_qubits, time_step)
        mpo = multi_qubit_gate.get_mpo_representation(1.0, 1.0e-16, 128)
        reference_values = [cexp(-1j * time_step), cexp(1j * time_step)]

        for i_row in product([0, 1], repeat=n_qubits):
            for i_col in product([0, 1], repeat=n_qubits):
                test_mps_row = MPS_computational_state(i_row)
                test_mps_col = MPS_computational_state(i_col)
                test_mps_col.reindex_(reindex_dict_b)
                overlap = test_mps_col & mpo & test_mps_row
                contraction_result = overlap ^ all

                # Checks
                if i_row == i_col:
                    self.assertAlmostEqual(np.abs(contraction_result), 1.0)
                    min_difference = min(abs(contraction_result - ref) for ref in reference_values)
                    self.assertAlmostEqual(min_difference, 0.0)
                else:
                    self.assertAlmostEqual(np.abs(contraction_result), 0.0)

    @data([0, 1, 2], [0, 2, 4], [1, 4, 5])
    def test_mpo_representation_of_gate(self, list_of_qubits: List[int]):
        """Checks that the MPO representation of the gate is constructed properly"""

        # First verifies that the MPO is constructed properly.
        n_qubits = 6
        multi_qubit_gate = QAOAManyBodyCorrelator(list_of_qubits, n_qubits, -0.4)
        mpo = multi_qubit_gate.get_mpo_representation(1.0e-10, 128)
        self.assertEqual(mpo.L, n_qubits)

        # Checks that the dense representation of the MPO is unitary.
        norm = mpo.H @ mpo
        self.assertAlmostEqual(norm, 2**n_qubits)

    def test_application_three_body_interaction(self):
        """Checks that everything works with three-body correlators"""

        # Sets the parameters
        betas = [0.2]
        gammas = [0.4]

        # Usual construction
        graph = nx.Graph()
        graph.add_weighted_edges_from([(i, i + 1, 0.1 * (i + 1)) for i in range(4)])
        qaoa_mps = QAOACircuitMPSRepresentation.construct_from_graph(graph, 1.0e-10)
        qaoa_mps.apply_qaoa_layer(betas, gammas)

        # Construction from list -- as before, but adds a hyperedge with weight 0.
        list_of_edges = [
            [[0, 1], 0.1],
            [[1, 2], 0.2],
            [[2, 3], 0.3],
            [[3, 4], 0.4],
            [[0, 1, 2], 0.0],
        ]
        qaoa_mps_hyper = QAOACircuitMPSRepresentation.construct_from_list_of_edges(
            list_of_edges, truncation_threshold=1.0e-16, max_bond_dim=128
        )
        qaoa_mps_hyper.apply_qaoa_layer(betas, gammas)

        # Checks that the two MPSs are the same
        overlap = qaoa_mps_hyper.get_underlying_tn().H @ qaoa_mps.get_underlying_tn()
        self.assertAlmostEqual(overlap, 1.0)


@ddt
class TestCostFunction(TrainingPipelineTestCase):
    """Test methods for constructing the MPO representation of the cost function"""

    def test_trivial_cost_function_bond_dimension(self):
        """Tests that the bond dim of a trivial cost function is = 1"""
        n_qubits = 10
        origin_string = "IIIIIIIIII"
        list_of_labels = [origin_string[:i] + "X" + origin_string[i + 1 :] for i in range(n_qubits)]
        list_of_coefficients = [1.0] * n_qubits
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function = QAOACostFunction(sparse_pauli)
        mpo_cost_function = cost_function.return_mpo_representation()
        self.assertEqual(mpo_cost_function.get_max_bond_dimension(), 2)

    def test_empty_cost_function_norm_symbolic(self):
        """Tests that the norm of an empty cost function is = 0
        when calculated via the symbolic construction
        """
        n_qubits = 10
        symbolic_mpo = SymbolicMPOConstruction(n_qubits)
        mpo_cost_function = symbolic_mpo.generate_mpo_representation()
        operator_norm = (mpo_cost_function.H & mpo_cost_function) ^ all
        self.assertAlmostEqual(operator_norm, 0)

    def test_cost_function_one_body_symbolic(self):
        """Test the norm of a one-body cost operator using
        the symbolic MPO construction
        """
        n_qubits = 4
        pauli_strings = ["ZIII", "IZII", "IIZI", "IIIZ"]
        pauli_coeffs = [1.0, -2.0, 3.0, -4.0]
        symbolic_mpo = SymbolicMPOConstruction(n_qubits)
        for i_string, i_coeff in zip(pauli_strings, pauli_coeffs):
            symbolic_mpo.add_term(i_string, i_coeff)
        mpo_cost_function = symbolic_mpo.generate_mpo_representation()
        operator_norm = (mpo_cost_function.H & mpo_cost_function) ^ all
        self.assertAlmostEqual(operator_norm.imag, 0.0)

        cost_function_conventional = QAOACostFunction(SparsePauliOp(pauli_strings, pauli_coeffs))
        mpo_cost_function_ref = cost_function_conventional.mpo.mpo
        operator_norm_ref = (mpo_cost_function_ref.H & mpo_cost_function_ref) ^ all
        self.assertAlmostEqual(operator_norm_ref.imag, 0.0)

        fidelity = ((mpo_cost_function_ref.H & mpo_cost_function) ^ all) / (
            sqrt(operator_norm.real) * sqrt(operator_norm_ref.real)
        )
        self.assertAlmostEqual(fidelity, 1.0)

    def test_cost_function_two_body_symbolic(self):
        """Test the norm of a two-body cost operator using
        the symbolic MPO construction
        """
        n_qubits = 5
        pauli_strings = ["ZZIII", "IZZII", "IIZZI", "IIIZZ", "ZIIIZ"]
        pauli_coeffs = [1.5, -2.2, 3.1, -4.6, 5.0]
        symbolic_mpo = SymbolicMPOConstruction(n_qubits)
        for i_string, i_coeff in zip(pauli_strings, pauli_coeffs):
            symbolic_mpo.add_term(i_string, i_coeff)
        mpo_cost_function = symbolic_mpo.generate_mpo_representation()
        operator_norm = (mpo_cost_function.H & mpo_cost_function) ^ all
        self.assertAlmostEqual(operator_norm.imag, 0.0)

        cost_function_conventional = QAOACostFunction(SparsePauliOp(pauli_strings, pauli_coeffs))
        mpo_cost_function_ref = cost_function_conventional.mpo.mpo
        operator_norm_ref = (mpo_cost_function_ref.H & mpo_cost_function_ref) ^ all
        self.assertAlmostEqual(operator_norm_ref.imag, 0.0)

        fidelity = ((mpo_cost_function_ref.H & mpo_cost_function) ^ all) / (
            sqrt(operator_norm.real) * sqrt(operator_norm_ref.real)
        )
        self.assertAlmostEqual(fidelity, 1.0)

    def test_cost_function_many_body_symbolic(self):
        """Test the norm of a two-body cost operator using
        the symbolic MPO construction
        """
        n_qubits = 5
        pauli_strings = ["ZIIII", "ZZIII", "ZZZII", "ZZZZI", "ZZZZZ"]
        pauli_coeffs = [2.0] * len(pauli_strings)
        symbolic_mpo = SymbolicMPOConstruction(n_qubits)
        for i_string, i_coeff in zip(pauli_strings, pauli_coeffs):
            symbolic_mpo.add_term(i_string, i_coeff)
        mpo_cost_function = symbolic_mpo.generate_mpo_representation()
        operator_norm = (mpo_cost_function.H & mpo_cost_function) ^ all
        self.assertAlmostEqual(operator_norm.imag, 0.0)

        cost_function_conventional = QAOACostFunction(SparsePauliOp(pauli_strings, pauli_coeffs))
        mpo_cost_function_ref = cost_function_conventional.mpo.mpo
        operator_norm_ref = (mpo_cost_function_ref.H & mpo_cost_function_ref) ^ all
        self.assertAlmostEqual(operator_norm_ref.imag, 0.0)

        fidelity = ((mpo_cost_function_ref.H & mpo_cost_function) ^ all) / (
            sqrt(operator_norm.real) * sqrt(operator_norm_ref.real)
        )
        self.assertAlmostEqual(fidelity, 1.0)

    @data("IIZIZZIIIII", "IZIZIZIIII", "ZZZIIIIIII")
    def test_trivial_cost_function_three_higher_body(self, pauli_string: str):
        """Tests that the code supports three- (and higher) body operators"""
        sparse_pauli = SparsePauliOp([pauli_string], [0.5])
        cost_function = QAOACostFunction(sparse_pauli, 1.0e-15)
        # Here we must set a truncation threshold != 0 because the MPO in the
        # `QAOACostFunction` is initially a bond dimension 1 MPO with zero elements.
        # The new term is then added and, therefore, if we did not truncate,
        # the overall MPO would have bond dimension 2.
        mpo_cost_function = cost_function.mpo
        self.assertEqual(mpo_cost_function.get_max_bond_dimension(), 1)

    def test_trivial_cost_function_on_mps(self):
        """Tests that the expectation value of a trivial cost function is = norm of the MPS"""
        coeff = 0.16071991
        list_of_labels = ["IIIII"]
        list_of_coefficients = [coeff]
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function = QAOACostFunction(sparse_pauli)
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -0.1), (1, 2, 1.2), (2, 3, 1.4), (3, 4, -0.7)])
        qaoa_mps = QAOACircuitMPSRepresentation.construct_from_graph(
            graph, truncation_threshold=1.0e-10, max_bond_dim=1000
        )
        qaoa_mps.apply_qaoa_layer([0.1, 0.2], [0.1, 0.2])
        cost_function = qaoa_mps.compute_cost_function(cost_function)
        self.assertAlmostEqual(cost_function, coeff)

    def test_add_null_three_body_term(self):
        """Checks that adding a three-body term scaled by zero does not change the energy
        expectation value
        """

        # Constructs the graph
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -0.1), (1, 2, 1.2), (2, 3, 1.4)])
        qaoa_mps = QAOACircuitMPSRepresentation.construct_from_graph(
            graph, truncation_threshold=1.0e-10, max_bond_dim=1000
        )
        qaoa_mps.apply_qaoa_layer([0.1, 0.2], [0.1, 0.2])

        # Constructs the reference cost op
        coeff = 0.16071991
        list_of_labels = ["ZZII", "IZZI", "IIZZ"]
        list_of_coefficients = [coeff, -coeff, coeff * 0.4]
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function_mpo = QAOACostFunction(sparse_pauli)
        cost_function_original = qaoa_mps.compute_cost_function(cost_function_mpo)

        # Add the three-body term
        list_of_labels.append("ZIZZ")
        list_of_coefficients.append(0.0)
        sparse_pauli_modified = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function_mpo_modified = QAOACostFunction(sparse_pauli_modified)
        cost_function_modified = qaoa_mps.compute_cost_function(cost_function_mpo_modified)

        # Checks coherence
        self.assertAlmostEqual(cost_function_modified, cost_function_original)

    def test_cost_function_raises_key_error(self):
        """Tests that, if the cost function and circuit are incoherent, an exception is raised"""
        list_of_labels = ["XX", "YY"]
        list_of_coefficients = [0.1, 0.2]
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function = QAOACostFunction(sparse_pauli)
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -0.1), (1, 2, 1.2), (2, 3, 1.4), (3, 4, -0.7)])
        qaoa_mps = QAOACircuitMPSRepresentation.construct_from_graph(graph)
        qaoa_mps.apply_qaoa_layer([0.1, 0.2], [0.1, 0.2])
        with self.assertRaises(KeyError):
            _ = qaoa_mps.compute_cost_function(cost_function)


@ddt
class TestVidalCircuitMPS(TrainingPipelineTestCase):
    """Tests the class representing a circuit as an MPS in Vidal's format"""

    @staticmethod
    def generate_template_circuit(n_qubits: int, n_rep: int, seed: int = 42) -> CircuitMPS:
        """Generates the template circuit for the subsequent tests"""
        mps_circuit = CircuitMPS(n_qubits)
        random.seed(seed)
        for _ in range(n_rep):
            for i_qubit in range(n_qubits):
                mps_circuit.rz(random.uniform(0, 1) * i_qubit, i_qubit)
            for i_qubit in range(n_qubits - 1):
                mps_circuit.cx(i_qubit, i_qubit + 1)
            for i_qubit in range(n_qubits):
                mps_circuit.rx(random.uniform(0, 1) * i_qubit, i_qubit)
            for i_qubit in range(n_qubits - 1):
                mps_circuit.cx(i_qubit, i_qubit + 1)
            for i_qubit in range(n_qubits):
                mps_circuit.ry(random.uniform(0, 1) * i_qubit, i_qubit)
            for i_qubit in range(n_qubits - 1):
                mps_circuit.cx(i_qubit, i_qubit + 1)
        return mps_circuit

    def test_circuit_mps_vidal_trivial_constructor(self):
        """Tests trivial constructor of a Vidal's gauge MPS circuit"""
        n_qubits = 10
        n_rep = 3
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, n_rep)
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)
        overlap = (mps_circuit.psi.H & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

    def test_circuit_mps_vidal_empty_constructor(self):
        """Tests trivial constructor of a Vidal's gauge MPS circuit"""
        n_qubits = 20
        vidal_circuit = CircuitMPSVidalCanonization.construct_empty_circuit(n_qubits)
        overlap = (vidal_circuit.tensor_network & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

    def test_circuit_mps_vidal_single_qubit_gate(self):
        """Test the functionalities for applying a single-qubit gate onto a
        circuit encoded as an MPS in Vidal's form
        """
        n_qubits = 7
        n_rep = 2
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, n_rep, 44)
        # Now applies a single-qubit rotation to both the Vidal and the conventional
        # form, and checks that the result is the same.
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)
        for i_qubit in range(n_qubits):
            vidal_circuit.apply_rx_gate(i_qubit, 0.16071991 * i_qubit)
            mps_circuit.rx(0.16071991 * i_qubit, i_qubit)
        overlap = (mps_circuit.psi.H & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

    def test_circuit_mps_vidal_two_qubit_gate(self):
        """Test the functionalities for applying a two-qubit gate onto a
        circuit encoded as an MPS in Vidal's form
        """

        # Constructs the relevant data structures
        n_qubits = 3
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, 1, 32)
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)
        self.assertTrue(vidal_circuit.check_if_all_tensors_are_normalized(1.0e-14))

        # Now applies a single-qubit rotation to both the Vidal and the conventional
        # form, and checks that the result is the same.
        theta = 0.321
        vidal_circuit.apply_rzz_gate_nn(0, theta)
        mps_circuit.rzz(theta, 0, 1)
        overlap = (mps_circuit.psi.H & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

        # Check also that the Vidal tensor is properly normalized
        self.assertTrue(vidal_circuit.check_if_all_tensors_are_normalized(1.0e-14))

    def test_circuit_mps_vidal_two_qubit_gate_non_nn(self):
        """Test the functionalities for applying a two-qubit gate onto a
        circuit encoded as an MPS in Vidal's form
        """
        n_qubits = 10
        n_reps = 10
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, n_reps, 101)
        # Now applies a single-qubit rotation to both the Vidal and the conventional
        # form, and checks that the result is the same.
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)
        for i_qubit in range(1, n_qubits - 1):
            vidal_circuit.apply_rzz_gate(0, i_qubit, 0.23)
            mps_circuit.rzz(0.23, 0, i_qubit)
        overlap = (mps_circuit.psi.H & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_conversion_from_vidal_to_canonical(self, canonization_center):
        """Checks that an MPS in Vidal's form can be converted into the canonical form.

        The sanity check is performed by checking that 1) the norm is preserved and that
        2) the expectation value of over a given cost operator is preserved.
        """

        # Prepares the relevant data structures
        n_qubits = 17
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, 1, 111)
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)

        # Check on the norm
        vidal_norm = vidal_circuit.tensor_network.norm()
        mps_representation = vidal_circuit.get_mps_representation(canonization_center)
        mps_norm = mps_representation.norm()
        self.assertAlmostEqual(mps_norm, vidal_norm)

        # Check on the energy
        list_of_labels = ["IZZIIIIIIIIIIIIII", "IIIIIIIIIZZIIIIII", "IIIZIIIIIIIIIIIZI"]
        list_of_coefficients = [0.123, -0.456, 0.789]
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function = QAOACostFunction(sparse_pauli)
        mpo_cost_function = cost_function.return_mpo_representation().mpo
        tn_vidal_dagger = vidal_circuit.tensor_network.H
        tn_vidal_dagger.reindex_(
            dict(zip(mpo_cost_function.lower_inds, mpo_cost_function.upper_inds))
        )
        mpo_expectation_vidal = (
            tn_vidal_dagger & mpo_cost_function & vidal_circuit.tensor_network
        ).contract(...)
        mps_dagger = mps_representation.H
        mps_dagger.reindex_(dict(zip(mpo_cost_function.lower_inds, mpo_cost_function.upper_inds)))
        mpo_expectation = (mps_dagger & mpo_cost_function & mps_representation).contract(...)
        self.assertAlmostEqual(mpo_expectation, mpo_expectation_vidal)

    @data(1, 2, 4, 8, 16, 32)
    def test_mps_vidal_vs_canonical_small_scale_small_m(self, bond_dimension):
        """Checks Vidal vs canonical gauge for a finite bond dimension"""
        list_of_coefficients = [1.0, -1.0, 0.4, -0.3]
        list_of_labels = ["IIIZIIIZIIIIII", "IIIZIIIZIIIIII", "IIIIIZIIIZIIII", "ZIIIIIIIIIIIIZ"]

        cost_function_sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)

        params = [1.0, 2.0]
        bond_dimension = 3

        # Cost function evaluation
        mps_evaluator_with_vidal = MPSEvaluator(
            bond_dim_circuit=bond_dimension, use_vidal_form=True, store_schmidt_values=False
        )
        energy_with_vidal = mps_evaluator_with_vidal.evaluate(cost_function_sparse_pauli, params)

        mps_evaluator_without_vidal = MPSEvaluator(
            bond_dim_circuit=bond_dimension, use_vidal_form=False, store_schmidt_values=True
        )
        energy_without_vidal = mps_evaluator_without_vidal.evaluate(
            cost_function_sparse_pauli, params
        )
        self.assertAlmostEqual(energy_with_vidal, energy_without_vidal)

        # Check on the Schmidt values
        schmidt_values = mps_evaluator_without_vidal.schmidt_values
        schmidt_values_vidal = mps_evaluator_with_vidal.schmidt_values
        self.assertEqual(len(schmidt_values), 13)
        self.assertIs(schmidt_values_vidal, None)

    def test_circuit_mps_vidal_swap_gate(self):
        """Test the functionalities for applying a two-qubit gate onto a
        circuit encoded as an MPS in Vidal's form
        """
        n_qubits = 10
        mps_circuit = TestVidalCircuitMPS.generate_template_circuit(n_qubits, 1, 100)
        # Now swaps and swaps back and check that the original MPS is obtained
        vidal_circuit = CircuitMPSVidalCanonization(mps_circuit)
        for i_qubit in range(n_qubits - 1):
            vidal_circuit.apply_swap_gate(i_qubit)
        for i_qubit in range(n_qubits - 2, -1, -1):
            vidal_circuit.apply_swap_gate(i_qubit)
        overlap = (mps_circuit.psi.H & vidal_circuit.tensor_network).contract(...)
        self.assertAlmostEqual(abs(overlap), 1.0)

    def test_vidal_trivial_cost_function_on_mps(self):
        """Tests that a trivial (i.e., = identity) cost function is = norm of the MPS"""
        coeff = 0.30031989
        list_of_labels = ["IIIII"]
        list_of_coefficients = [coeff]
        sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)
        cost_function = QAOACostFunction(sparse_pauli)
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, -0.1), (1, 2, 1.2), (2, 3, 1.4), (3, 4, -0.7)])
        qaoa_mps = QAOACircuitVidalRepresentation.construct_from_graph(
            graph, truncation_threshold=1.0e-10, max_bond_dim=1000
        )
        qaoa_mps.apply_qaoa_layer([0.1, 0.2], [0.1, 0.2])
        cost_function = qaoa_mps.compute_cost_function(cost_function)
        self.assertAlmostEqual(cost_function, coeff)

    def test_coherence_between_conventional_and_vidal(self):
        """Checks that the MPS simulator yields the same results independently on the chosen
        canonization
        """

        # Generate the cost function
        number_of_qubits = 10
        graph = nx.random_regular_graph(d=3, n=number_of_qubits)
        cost_op = graph_to_operator(graph)
        truncation_threshold = 1.0e-10

        # Vidal gauge
        mps_evaluator_vidal = MPSEvaluator(
            use_vidal_form=True, threshold_circuit=truncation_threshold
        )
        cost_function_vidal = mps_evaluator_vidal.evaluate(cost_op, [1, 2])

        # Conventional gauge
        mps_evaluator_conventional = MPSEvaluator(
            use_vidal_form=False, threshold_circuit=truncation_threshold
        )
        cost_function_conventional = mps_evaluator_conventional.evaluate(cost_op, [1, 2])

        # Final check
        self.assertAlmostEqual(cost_function_conventional, cost_function_vidal)

    def test_mps_vidal_vs_canonical_large_scale(self):
        """Checks Vidal vs canonical gauge for a large circuit.

        Note that "large" means here "with a large number of qubits" -- however, the circuit itself
        is easy (i.e., only one repetition.)
        """
        list_of_coefficients = [
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
        ]
        # pylint: disable=line-too-long
        list_of_labels = [
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZ",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZI",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZ",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIIZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIIZIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "ZIIIIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIIZIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIIZIIIIIIIIIZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IIZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "IZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "ZZIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
        ]

        cost_function_sparse_pauli = SparsePauliOp(list_of_labels, list_of_coefficients)

        params = [1.0, 2.0]
        bond_dimension = 32

        # Cost function evaluation
        mps_evaluator_with_vidal = MPSEvaluator(
            bond_dim_circuit=bond_dimension, use_vidal_form=True
        )
        energy_with_vidal = mps_evaluator_with_vidal.evaluate(cost_function_sparse_pauli, params)
        mps_evaluator_without_vidal = MPSEvaluator(
            bond_dim_circuit=bond_dimension, use_vidal_form=False
        )
        energy_without_vidal = mps_evaluator_without_vidal.evaluate(
            cost_function_sparse_pauli, params
        )
        self.assertAlmostEqual(energy_with_vidal, energy_without_vidal)


@ddt
class TestMatrixProductStateHOBO(TrainingPipelineTestCase):
    """Test on HOBO problems"""

    @staticmethod
    def contruct_labs_hamiltonian(n_qubits: int) -> List[Tuple[List[int], float]]:
        """Constructs an Ising Hamiltonian associated with a LABS problem.

        Args:
            n_qubits (int): number of qubits associated with the problem

        Returns:
            List[List[int], float]: Coupling map associated with the Ising
                Hamiltonian that solves the LABS problem.
        """

        returned_list = []

        # Two-body part
        for i in range(1, n_qubits - 1):
            for j in range(1, int(np.floor((n_qubits - i) / 2)) + 1):
                returned_list.append([(i - 1, i + 2 * j - 1), 1.0])

        # Four-body part
        for i in range(1, n_qubits - 2):
            # pylint: disable=invalid-name
            for t in range(1, int(np.floor((n_qubits - i - 1) / 2)) - 1):
                for k in range(t + 1, n_qubits - t - i - 1):
                    returned_list.append([(i - 1, i + t - 1, i + k - 1, i + t + k - 1), 2.0])

        return returned_list

    @data(4, 8, 16, 20)
    def test_mpo_representation_labs_problem(self, n_qubits: int):
        """Checks validity of MPO representation of LABS problem."""

        # Generates the LABS problem as a SparsePauliOp
        list_of_terms = TestMatrixProductStateHOBO.contruct_labs_hamiltonian(n_qubits)
        overall_sparse_pauli = SparsePauliOp(["I" * n_qubits], [0.0])

        # Prepares data for MPO construction
        symbolic_mpo = SymbolicMPOConstruction(n_qubits)
        for i_term in list_of_terms:
            tmp_pauli = ["I"] * n_qubits
            for i_char in i_term[0]:
                tmp_pauli[i_char] = "Z"
            symbolic_mpo.add_term("".join(tmp_pauli), i_term[1])
            overall_sparse_pauli += SparsePauliOp([Pauli("".join(tmp_pauli))], [i_term[1]])

        # Actual MPO generation
        mpo_from_symbolic_constructor = symbolic_mpo.generate_mpo_representation()
        qaoa_cost_function = QAOACostFunction(overall_sparse_pauli)
        mpo_from_brute_force = qaoa_cost_function.mpo.mpo

        # Checks that the two MPOs are actually the same
        norm_symbolic_constructor = (
            mpo_from_symbolic_constructor.H & mpo_from_symbolic_constructor
        ) ^ all
        norm_brute_force = (mpo_from_brute_force.H & mpo_from_brute_force) ^ all
        self.assertAlmostEqual(norm_symbolic_constructor.imag, 0.0)
        self.assertAlmostEqual(norm_brute_force.imag, 0.0)
        self.assertAlmostEqual(norm_brute_force.real, norm_symbolic_constructor.real)

        # Checks fidelity
        overlap = (mpo_from_brute_force.H & mpo_from_symbolic_constructor) ^ all
        fidelity = overlap / sqrt(norm_brute_force.real * norm_symbolic_constructor.real)
        self.assertAlmostEqual(fidelity, 1.0)

    @data("IIIIIIII", "IIIXXIII", "IIXIXIII", "IXIXIXIX", "IIZZIIII", "ZIIIIIZI", "ZIZIZIZI")
    def test_hobo_on_labs_problem(self, pauli_string: str):
        """Checks HOBO MPS vs statevector on a LABS problem"""

        # Generates the LABS problem
        n_qubits = 8
        list_of_terms = TestMatrixProductStateHOBO.contruct_labs_hamiltonian(n_qubits)

        # MPS simulation
        mps_simulator = QAOACircuitMPSRepresentation.construct_from_list_of_edges(list_of_terms)
        mps_simulator_vidal = QAOACircuitVidalRepresentation.construct_from_list_of_edges(
            list_of_terms
        )

        beta = 0.2
        gamma = 0.3
        mps_simulator.apply_qaoa_layer([beta], [gamma])
        mps_simulator_vidal.apply_qaoa_layer([beta], [gamma])

        # Simple checks for both Vidal and conventional
        self.assertEqual(mps_simulator.n_qubits, n_qubits)
        self.assertAlmostEqual(mps_simulator.get_underlying_tn().norm(), 1.0)
        self.assertEqual(mps_simulator_vidal.n_qubits, n_qubits)
        self.assertAlmostEqual(mps_simulator_vidal.get_underlying_tn().norm(), 1.0)

        # Statevector simulator
        z_gate = SparsePauliOp("Z")
        exp_op = z_gate ^ z_gate ^ z_gate ^ z_gate
        qc = QuantumCircuit(n_qubits)

        # H layer
        for i_qubit in range(n_qubits):
            qc.h(i_qubit)
        # Ansatz layer
        for i_term in list_of_terms:
            if len(i_term[0]) == 2:
                qc.rzz(2 * i_term[1] * gamma, i_term[0][0], i_term[0][1])
            elif len(i_term[0]) == 4:
                quadruple_zeta_gate = PauliEvolutionGate(exp_op, gamma * i_term[1])
                qc.append(quadruple_zeta_gate, i_term[0])
        # Mixing layer
        for i_qubit in range(n_qubits):
            qc.rx(2 * beta, i_qubit)

        # Checks that the overlap is one
        overlap = (
            mps_simulator.get_underlying_tn().H & mps_simulator_vidal.get_underlying_tn()
        ) ^ all
        self.assertAlmostEqual(overlap, 1)

        # Checks coherence in some expectation values for the Vidal and conventional case
        mps_exp_val = mps_simulator.compute_expectation_value_single_pauli_string(pauli_string)
        mps_exp_val_vidal = mps_simulator_vidal.compute_expectation_value_single_pauli_string(
            pauli_string
        )
        self.assertAlmostEqual(mps_exp_val_vidal, mps_exp_val)

        # Check results coherence wrt exact state vector estimator
        estimator = StatevectorEstimator()
        result = estimator.run([(qc, Pauli(pauli_string[::-1]), [])]).result()
        sv_exp_val = result[0].data.evs
        self.assertAlmostEqual(mps_exp_val, sv_exp_val)
        self.assertAlmostEqual(mps_exp_val_vidal, sv_exp_val)
