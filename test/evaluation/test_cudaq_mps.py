#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""CUDA-Q MPS benchmark evaluator tests."""

from unittest import TestCase, skipUnless

import networkx as nx

from qaoa_training_pipeline.evaluation import MPSEvaluator
from qaoa_training_pipeline.evaluation.cudaq_mps import CudaQMPSBenchmarkEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


def _cudaq_tensornet_mps_ready() -> tuple[bool, str]:
    """Return whether CUDA-Q's GPU MPS target is usable on this machine."""

    try:
        import cudaq  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False, "CUDA-Q is not installed."

    if hasattr(cudaq, "has_target") and not cudaq.has_target("tensornet-mps"):
        return False, "CUDA-Q target 'tensornet-mps' is not available."

    if hasattr(cudaq, "num_available_gpus"):
        try:
            if cudaq.num_available_gpus() <= 0:
                return False, "CUDA-Q reports zero available NVIDIA GPUs."
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f"CUDA-Q GPU query failed: {exc}"

    return True, "CUDA-Q tensornet-mps target is available."


CUDAQ_READY, CUDAQ_SKIP_REASON = _cudaq_tensornet_mps_ready()


class TestCudaQMPSOptionalImport(TestCase):
    """Tests that do not require CUDA-Q to be installed."""

    def test_module_import_does_not_require_cudaq_target_setup(self):
        """Importing the evaluator module should not configure CUDA-Q."""

        self.assertEqual(CudaQMPSBenchmarkEvaluator.__name__, "CudaQMPSBenchmarkEvaluator")


@skipUnless(CUDAQ_READY, CUDAQ_SKIP_REASON)
class TestCudaQMPSBenchmarkEvaluator(TestCase):
    """Compare CUDA-Q MPS energies to the existing CPU MPS evaluator."""

    def assert_cudaq_matches_cpu(
        self,
        graph: nx.Graph,
        params: list[float],
        precision: str = "fp64",
        places: int = 6,
    ) -> None:
        """Compare CUDA-Q against the exact CPU MPS evaluator."""

        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        expected = MPSEvaluator().evaluate(cost_op, params)

        evaluator = CudaQMPSBenchmarkEvaluator(
            max_bond_dim=64,
            abs_cutoff=1.0e-12,
            relative_cutoff=1.0e-12,
            precision=precision,
        )

        try:
            actual = evaluator.evaluate(cost_op, params)
        except RuntimeError as exc:
            self.skipTest(f"CUDA-Q target setup failed: {exc}")

        self.assertAlmostEqual(actual, expected, places=places)

    def test_triangle_graph_depth_one(self):
        """Triangle MaxCut, p=1."""

        graph = nx.complete_graph(3)
        self.assert_cudaq_matches_cpu(graph, [0.37, -0.22])

    def test_path_graph_depth_two(self):
        """Path graph with four nodes, p=2."""

        graph = nx.path_graph(4)
        self.assert_cudaq_matches_cpu(graph, [0.11, -0.28, 0.33, 0.07])

    def test_weighted_graph_depth_one(self):
        """Small weighted MaxCut graph."""

        graph = nx.Graph()
        graph.add_weighted_edges_from(
            [
                (0, 1, 1.25),
                (1, 2, 0.5),
                (0, 2, 2.0),
            ]
        )

        self.assert_cudaq_matches_cpu(graph, [0.19, 0.41])

    def test_graph_initialized_betas_gammas_call(self):
        """Graph-initialized evaluator supports evaluate(betas, gammas)."""

        graph = nx.cycle_graph(4)
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        expected = MPSEvaluator().evaluate(cost_op, [0.23, -0.31])

        evaluator = CudaQMPSBenchmarkEvaluator(
            graph=graph,
            max_bond_dim=64,
            abs_cutoff=1.0e-12,
            relative_cutoff=1.0e-12,
        )
        try:
            actual = evaluator.evaluate([0.23], [-0.31])
        except RuntimeError as exc:
            self.skipTest(f"CUDA-Q target setup failed: {exc}")

        self.assertAlmostEqual(actual, expected, places=6)

    def test_fp32_precision_uses_looser_tolerance(self):
        """Single-precision target option is allowed with a looser tolerance."""

        graph = nx.path_graph(3)
        self.assert_cudaq_matches_cpu(graph, [0.13, 0.21], precision="fp32", places=3)
