#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This is the main file that lets us train from the command line.

Example usage:

```
 python -m train --input ../data/example_graph.json --config ../data/example_config.json
```

In the example above, the problem is contained in `example_graph.json` which defines the
graph that we want to optimize. The way in which we want to train is given in the
`example_config.json` file. This file specifies which trainers to use and the input
to initialize them, call them, as well as the evaluators to use. Crucially, the training
is done with a list of trainers. This allows us to, for example, use a given
method for depth `p` and then use another methods at depth `p+1`. The framework is
capable of extracting input from a previously obtained training result.
"""

import argparse
from datetime import datetime
import os
import json

from qaoa_training_pipeline.utils.data_utils import load_input, input_to_operator
from qaoa_training_pipeline.pre_processing import PREPROCESSORS
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES
from qaoa_training_pipeline.pipeline import Pipeline


def get_script_args() -> tuple[argparse.Namespace, list]:
    """Get the command line input arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="The path to the graph or hyper graph to train on.",
    )

    description = "A string to create instances of known optimization problems with "
    description += "the format `problem_class_name` or `problem_class_name:input_str`. "
    description += "Here, problem_class_name is a key in PROBLEM_CLASSES and "
    description += "input_str is a string to initialize the problem class if needed."

    parser.add_argument(
        "--problem_class",
        required=False,
        type=str,
        help=description,
    )

    description = "A pre-factor that multiplies all the weights in the input. "
    description += "This argument is optional, defaults to 1.0, and cannot "
    description += "be used in conjunction with `--problem_class`."

    parser.add_argument(
        "--pre_processing",
        required=False,
        type=str,
        help="A pre-processing hook that acts on the loaded input problem data.",
    )

    parser.add_argument(
        "--pre_factor",
        required=False,
        type=float,
        help=description,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="The path to the json file that specifies the training configuration.",
    )

    parser.add_argument(
        "--save_file",
        required=False,
        type=str,
        help="The name of the file under which the results will be saved.",
    )

    description = "We explicitly require users to say if they want to save their output. "
    description += "If the `--save` option is not given then nothing is saved."

    parser.add_argument(
        "--save",
        action="store_true",
        help=description,
    )

    parser.add_argument(
        "--save_dir",
        required=False,
        default="",
        type=str,
        help="The directory where to save the data relative to the caller location.",
    )

    parser.add_argument(
        "--provider_kwargs",
        required=False,
        default="",
        type=str,
        help="Arguments for the initial ParamsProvider.",
    )

    for idx in range(10):
        help_str = f"A string specifying the train kwargs of trainer {idx}. "
        help_str += "To see how this is used go and check the parse_kwargs method in the trainer."

        parser.add_argument(
            f"--train_kwargs{idx}",
            required=False,
            type=str,
            help=help_str,
        )

    for idx in range(10):
        help_str = f"A string to specify the init kwargs of evaluator in trainer {idx}. To see how"
        help_str += "this is used go and check the parse_init_kwargs method in the evaluators."

        parser.add_argument(
            f"--evaluator_init_kwargs{idx}",
            required=False,
            type=str,
            help=help_str,
        )

    run_args, additional_args = parser.parse_known_args()
    return run_args, additional_args


def prepare_train_kwargs(config: dict) -> None:
    """Deserialize the input arguments for the train function.

    This is a hook that will allow us to prepare the input arguments to the train
    function. This might do steps like deserializer mixer circuits, ansatz circuits,
    etc.
    """
    for name in ["mixer", "initial_state", "ansatz_circuit"]:
        if name in config["train_kwargs"]:
            raise NotImplementedError(f"Serialization is not yet implemented for {name}.")


def train(args: argparse.Namespace) -> dict:
    """Main function that does the training.

    The training is configurable based on system inputs. Use the `help` function to get
    a description of the arguments that are accepted. The training is done in an iterative
    fashion which allows one trainer to leverage the result of a previous trainer.
    """

    # Validate command line options
    class_str = getattr(args, "problem_class", None)
    pre_factor = getattr(args, "pre_factor", None)
    if pre_factor is not None and class_str is not None:
        raise ValueError(
            "Malformed command input. pre_factor and problem_class cannot be used together."
        )

    # Load and optionally pre-process the input.
    input_data = load_input(args.input)

    # Pre-process the input data if needed.
    pre_processing, pre_processor = getattr(args, "pre_processing", None), None
    if pre_processing is not None:
        pre_processing_info = pre_processing.split(":")
        pre_processing_name = pre_processing_info[0]

        pre_processing_init_str = ""
        if len(pre_processing_info) > 1:
            pre_processing_init_str = pre_processing_info[1]

        pre_processor = PREPROCESSORS[pre_processing_name].from_str(pre_processing_init_str)
        input_data = pre_processor(input_data)

    # Create the cost operator either from input or a supported problem class.
    if class_str is not None:
        class_info = class_str.split(":")
        class_name = class_info[0].lower()

        class_init_str = ""
        if len(class_info) > 1:
            class_init_str = class_info[1]

        if class_name not in PROBLEM_CLASSES:
            raise ValueError(
                f"The problem class {class_name} is not supported. "
                f"Valid problem classes are {PROBLEM_CLASSES.keys()}"
            )

        problem_class = PROBLEM_CLASSES[class_name].from_str(class_init_str)
        input_problem = problem_class.cost_operator(input_data)
    else:
        pre_factor = pre_factor or 1.0
        input_problem = input_to_operator(input_data, pre_factor=pre_factor)

    # Load the pipeline config
    with open(args.config, "r") as fin:
        full_config = json.load(fin)

    save_file = getattr(args, "save_file", None)
    all_results = {}

    # Create the pipeline from config and prepare runtime arguments
    pipeline, provider_args, component_args = Pipeline.from_config(full_config, input_problem, args)
    # Execute the pipeline with given argiuments
    pipeline.execute(provider_args, component_args, all_results)

    # Save the results if needed
    if args.save:
        date_tag = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
        if save_file is None:
            save_file_local = date_tag + "_" + full_config["save_file"]
        else:
            save_file_local = date_tag + "_" + save_file

        if not os.path.exists(args.save_dir) and args.save_dir != "":
            os.makedirs(args.save_dir)

        with open(args.save_dir + save_file_local, "w") as f_out:
            save_data = dict()
            for k, v in all_results.items():
                save_data[k] = v.data if isinstance(v, ParamResult) else v

            json.dump(save_data, f_out, indent=4)

    return all_results


if __name__ == "__main__":
    script_args, _ = get_script_args()
    train(script_args)
