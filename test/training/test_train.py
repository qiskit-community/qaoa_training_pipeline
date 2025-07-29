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
import os
import sys

from unittest.mock import patch
from ddt import ddt, data

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

    @data(0, 1, 2, 3, 4, 5, 6)
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
        }

        file_name = "dmp_file_test_methods_" + str(method_idx)

        test_args = [
            "prog",
            "--input",
            "data/problems/example_graph.json",
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
