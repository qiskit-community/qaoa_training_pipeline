#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of problem classes."""

from test import TrainingPipelineTestCase

from ddt import ddt, data, unpack

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES


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

        op1 = [[("IZZ", -0.5), ("ZIZ", -0.5)]]
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
