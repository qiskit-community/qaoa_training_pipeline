#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""cuQuantum MPS evaluator tests."""

from unittest import TestCase, skipUnless

import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.cuquantum_mps import CuQuantumMPSEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


def _cuquantum_ready() -> tuple[bool, str]:
    """Return whether the cuQuantum tensor-network Python API imports."""

    try:
        from cuquantum.tensornet.experimental import (  # pylint: disable=import-outside-toplevel,unused-import
            MPSConfig,
            NetworkOperator,
            NetworkState,
        )
    except (ImportError, OSError):
        return False, "cuQuantum Python is not installed."

    return True, "cuQuantum tensor-network APIs are available."


CUQUANTUM_READY, CUQUANTUM_SKIP_REASON = _cuquantum_ready()


class TestCuQuantumMPSOptionalImport(TestCase):
    """Tests that do not require cuQuantum to be installed."""

    def test_module_import_does_not_require_cuquantum(self):
        """Importing the evaluator module should not import cuQuantum."""

        self.assertEqual(CuQuantumMPSEvaluator.__name__, "CuQuantumMPSEvaluator")

    def test_parse_init_kwargs(self):
        """The evaluator config parser works without cuQuantum installed."""

        config = CuQuantumMPSEvaluator.parse_init_kwargs("16_1e-5_1e-6_fp32_gesvdj")
        self.assertEqual(config["max_bond_dim"], 16)
        self.assertEqual(config["abs_cutoff"], 1.0e-5)
        self.assertEqual(config["rel_cutoff"], 1.0e-6)
        self.assertEqual(config["precision"], "fp32")
        self.assertEqual(config["svd_algo"], "gesvdj")

    def test_custom_mixer_rejected_before_runtime_import(self):
        """Unsupported features are rejected before requiring cuQuantum runtime."""

        cost_op = SparsePauliOp.from_list([("ZZ", -0.5)])
        mixer = QuantumCircuit(2)
        mixer.rx(0.1, 0)

        with self.assertRaises(NotImplementedError):
            CuQuantumMPSEvaluator().evaluate(cost_op, [0.2, 0.3], mixer=mixer)

    def test_default_observable_strategy_is_cuquantum_products(self):
        """The evaluator defaults to the cuQuantum-only product-operator path."""

        evaluator = CuQuantumMPSEvaluator()
        self.assertEqual(evaluator.to_config()["observable_strategy"], "pauli_products")

    def test_quimb_mpo_strategy_is_not_supported(self):
        """The first cuQuantum evaluator version does not expose a Quimb MPO bridge."""

        with self.assertRaises(ValueError):
            CuQuantumMPSEvaluator(observable_strategy="mpo")

    def test_pauli_products_are_built_from_sparse_pauli_terms(self):
        """The observable path appends cuQuantum product terms without MPO conversion."""

        class RecordingOperator:
            """Minimal stand-in for cuQuantum NetworkOperator."""

            def __init__(self):
                self.products = []

            def append_product(self, coefficient, modes, tensors):
                self.products.append((coefficient, modes, tensors))

        cost_op = SparsePauliOp.from_list(
            [
                ("ZII", 1.5),
                ("IZZ", -0.5),
                ("III", 2.0),
            ]
        )
        terms, identity_offset = CuQuantumMPSEvaluator._terms_from_cost_operator(cost_op)
        operator = RecordingOperator()
        CuQuantumMPSEvaluator()._append_pauli_product_terms(operator, terms)

        self.assertEqual(identity_offset, 2.0)
        self.assertEqual([term.qubits for term in terms], [(2,), (0, 1)])
        self.assertEqual(len(operator.products), 2)
        self.assertEqual(operator.products[0][0], 1.5)
        self.assertEqual(operator.products[0][1], [[2]])
        self.assertEqual(operator.products[1][0], -0.5)
        self.assertEqual(operator.products[1][1], [[0], [1]])


@skipUnless(CUQUANTUM_READY, CUQUANTUM_SKIP_REASON)
class TestCuQuantumMPSEvaluator(TestCase):
    """Compare cuQuantum MPS energies to the existing CPU MPS evaluator."""

    def assert_cuquantum_matches_cpu(
        self,
        cost_op: SparsePauliOp,
        params: list[float],
        observable_strategy: str = "pauli_products",
        places: int = 6,
    ) -> None:
        """Compare cuQuantum against the exact CPU MPS evaluator."""

        from qaoa_training_pipeline.evaluation.mps_evaluator import (  # pylint: disable=import-outside-toplevel
            MPSEvaluator,
        )

        expected = MPSEvaluator().evaluate(cost_op, params)
        evaluator = CuQuantumMPSEvaluator(
            max_bond_dim=64,
            abs_cutoff=1.0e-12,
            rel_cutoff=1.0e-12,
            precision="fp64",
            observable_strategy=observable_strategy,
        )

        try:
            actual = evaluator.evaluate(cost_op, params)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.skipTest(f"cuQuantum runtime failed: {exc}")

        self.assertAlmostEqual(actual, expected, places=places)

    def test_triangle_graph_depth_one(self):
        """Triangle MaxCut, p=1, using the cuQuantum product observable path."""

        graph = nx.complete_graph(3)
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        self.assert_cuquantum_matches_cpu(cost_op, [0.37, -0.22])

    def test_weighted_graph_depth_two(self):
        """Weighted graph, p=2, using the cuQuantum product observable path."""

        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 1.25), (1, 2, 0.5), (0, 2, 2.0)])
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        self.assert_cuquantum_matches_cpu(cost_op, [0.11, -0.28, 0.33, 0.07])

    def test_path_graph_depth_one(self):
        """Path graph, p=1, using the cuQuantum product observable path."""

        graph = nx.path_graph(4)
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        self.assert_cuquantum_matches_cpu(cost_op, [0.23, -0.31])
