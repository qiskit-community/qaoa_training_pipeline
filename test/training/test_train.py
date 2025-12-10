#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the train.py script."""

from test import TrainingPipelineTestCase

import glob
import json
import os
import sys

from unittest.mock import patch
from ddt import ddt, data, unpack

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.train import train, get_script_args


@ddt
class TestTrain(TrainingPipelineTestCase):
    """Test the train.py script."""

    def tearDown(self) -> None:
        """Removes the created temporary data"""
        for file_name in glob.glob("*dmp_file*"):
            os.remove(file_name)

    def test_call_train(self):
        """Test that we can call train.py."""

        file_name = "dmp_file_test_call_train"

        test_args = [
            "prog",
            "--input",
            "data/problems/example_graph.json",
            "--config",
            "data/methods/train_method_0.json",
            "--train_kwargs0",
            "num_points:10:parameter_ranges:3/6/3/6",
            "--save",
            "--save_file",
            file_name,
            False,
        ]

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            self.assertEqual(
                result["args"]["train_kwargs0"],
                "num_points:10:parameter_ranges:3/6/3/6",
            )

        # Load from the saved file
        labels_to_test = ["pre_processing", "cost_operator", "0", "args"]
        for file_name in glob.glob("*dmp_file*"):
            with open(file_name, "r") as fin:
                ld_data = json.load(fin)

                for label in labels_to_test:
                    self.assertTrue(label in ld_data)

    @data("maxcut", "mis:3", "mis")
    def test_problem_classes(self, problem_str: str):
        """Test that we can call train.py."""

        test_args = [
            "prog",
            "--input",
            "test/data/test_graph.json",
            "--config",
            "data/methods/train_method_0.json",
            "--train_kwargs0",
            "num_points:2:parameter_ranges:3/6/3/6",
            "--problem_class",
            problem_str,
        ]

        op1 = [("IZZ", -0.5), ("ZIZ", -0.5)]
        op2 = [("IIZ", -0.5), ("IZZ", 0.5), ("ZIZ", 0.5)]
        op3 = [("IIZ", -1.0), ("IZI", -0.25), ("ZII", -0.25), ("IZZ", 0.75), ("ZIZ", 0.75)]

        expected = {
            "maxcut": SparsePauliOp.from_list(op1),
            "mis": SparsePauliOp.from_list(op2),
            "mis:3": SparsePauliOp.from_list(op3),
        }

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            cost_op = SparsePauliOp.from_list(result["cost_operator"])

            self.assertEqual(result["args"]["problem_class"], problem_str)
            self.assertEqual(cost_op, expected[problem_str])

    def test_validate_args(self):
        """Test that the arguments are validated correctly."""

        test_args = [
            "prog",
            "--input",
            "data/problems/example_graph.json",
            "--config",
            "data/methods/train_method_4.json",
            "--problem_class",
            "maxcut",
            "--pre_factor",
            "2.0",
            False,
        ]

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()

            with self.assertRaises(ValueError) as error:
                train(args)
                self.assertTrue("cannott be used together" in error.exception.args[0])

    def test_call_train_schmidt(self):
        """Test that the Schmidt values are returned with an MPS-based training."""

        file_name = "dmp_file_test_call_train"

        test_args = [
            "prog",
            "--input",
            "data/problems/example_graph.json",
            "--config",
            "data/methods/train_method_4.json",
            "--evaluator_init_kwargs0",
            "True_None_4_None_4_False_True",
            "--save",
            "--save_file",
            file_name,
            False,
        ]

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            # We extract here the zero-th element because we
            # are interested in the first trainer (which, in this
            # case, is the only one)
            self.assertIn("schmidt_values", result[0].keys())

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_methods(self, method_idx: int):
        """Test that the different methods run without input args."""

        # First value is the trainer index in the chain and the second one is the param length.
        expected_param_len = {
            0: (0, 2),
            1: (1, 2),
            2: (1, 2),
            3: (0, 2),
            4: (0, 2),
            5: (0, 2),
            6: (1, 6),
            7: (1, 6),
            8: (0, 4),  # The data in test/data/qaoa_angles.json is for p=2.
        }

        file_name = "dmp_file_test_methods_" + str(method_idx)

        test_args = [
            "prog",
            "--input",
            "test/data/test_graph.json",
            "--config",
            f"data/methods/train_method_{method_idx}.json",
            "--save",
            "--save_file",
            file_name,
            False,
        ]

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            trainer_idx, exp_len = expected_param_len[method_idx]
            opt_params = result[trainer_idx]["optimized_params"]
            self.assertEqual(len(opt_params), exp_len)

    @unpack
    @data((6, "reps:2", 4, 1), (6, "reps:3", 6, 1), (4, "params0:0/0/0/0", 4, 0))
    def test_change_reps(self, method_idx: int, trainer_kwars: str, exp_len: int, trainer_idx: int):
        """Test that we can change the number of reps.

        Args:
            method_idx: The index of the method in data/methods/.
            trainer_kwars: The keyword arguments to give to trainer.train. For example, this allows
                us to pass the QAOA depth at runtime if the trainer accepts it.
            exp_len: The expected length of the optimized parameters.
            trainer_idx: The index of the trainer in the trainer chain.
        """

        test_args = [
            "prog",
            "--input",
            "test/data/test_graph.json",
            "--config",
            f"data/methods/train_method_{method_idx}.json",
            f"--train_kwargs{trainer_idx}",
            trainer_kwars,
            False,
        ]

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            opt_params = result[trainer_idx]["optimized_params"]
            self.assertEqual(len(opt_params), exp_len)

    def test_sat_integration(self):
        """Test that the pipeline can call the SATMApper.

        The graph that we load has edges {(0, 1), (0, 2)}.
        The hardcoded SAT mapping returns the edges `{(1, 2), (1, 0)}`,
        which explains the value of `expected_op`.
        """

        test_args = [
            "prog",
            "--input",
            "test/data/test_graph.json",
            "--config",
            "data/methods/train_method_0.json",
            "--train_kwargs0",
            "num_points:2:parameter_ranges:3/6/3/6",
            "--problem_class",
            "maxcut",
            "--pre_processing",
            "sat:10",
        ]

        expected = {
            "pre_processor_name": "SATMapper",
            "timeout": 10,
            "min_k": 0,
            "edge_map": {0: 1, 1: 2, 2: 0},
        }

        expected_op = SparsePauliOp.from_list([("IZZ", -0.5), ("ZZI", -0.5)])

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            # Duration will be unknown, but at least this ensures its existance.
            result["pre_processing"].pop("duration")

            self.assertDictEqual(result["pre_processing"], expected)

            cost_op = SparsePauliOp.from_list(result["cost_operator"])
            self.assertSetEqual(set(cost_op.to_list()), set(expected_op.to_list()))

    def test_tqa_train(self):
        """Test that we can call train.py."""
        reps = 10
        test_args = [
            "prog",
            "--input",
            "test/data/test_graph.json",
            "--config",
            "data/methods/train_method_9.json",
            "--train_kwargs0",
            f"reps:{reps}",
            "--problem_class",
            "maxcut",
        ]

        op1 = [("IZZ", -0.5), ("ZIZ", -0.5)]
        expected = SparsePauliOp.from_list(op1)

        with patch.object(sys, "argv", test_args):
            args, _ = get_script_args()
            result = train(args)

            cost_op = SparsePauliOp.from_list(result["cost_operator"])
            print(result.keys())
            self.assertEqual(result["args"]["problem_class"], "maxcut")
            self.assertEqual(cost_op, expected)
            self.assertEqual(result[0]["x0"], [0.5, 0.5])
            self.assertEqual(len(result[0]["optimized_qaoa_angles"]), 2 * reps)
