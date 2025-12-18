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

In the example above, the problem is contained in `exmaple_graph.json` which defines the
graph that we want to optimize. The way in which we want to train is given in the
`example_config.json` file. This file specifies which trainers to use and the input
to initialize them, call them, as well as the evaluators to use. Crucially, the training
is done with a list of trainers. This allows us to, for example, use a given
method for depth `p` and then use another methods at depth `p+1`. The framework is
capable of extracting input from a previously obtained training result.
"""

import argparse
from datetime import datetime
from typing import Optional, List
import os
import json
import numpy as np

from qaoa_training_pipeline.utils.data_utils import load_input, input_to_operator
from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.pre_processing import PREPROCESSORS
from qaoa_training_pipeline.training import TRAINERS
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES

from qaoa_training_pipeline.utils.labs.labs_utils import (
    apply_labs_training_config_overrides,
)  # pylint: disable=import-outside-toplevel


def get_script_args():
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
    description += "This argument is optional, defaults to 1.0, and connot "
    description += "be used in conjuction with `--problem_class`."

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
        "--objective",
        required=False,
        type=str,
        choices=["energy", "overlap"],
        help=(
            "Override the objective used by ScipyTrainer. "
            "Supported values are 'energy' (default) and 'overlap' (LABS only). "
            "If provided, this overrides any 'objective' field in the method JSON."
        ),
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
        help_str = f"A string to specify the init kwargs of evalutor in trainer {idx}. To see how"
        help_str += "this is used go and check the parse_init_kwargs method in the evaluators."

        parser.add_argument(
            f"--evaluator_init_kwargs{idx}",
            required=False,
            type=str,
            help=help_str,
        )

    run_args, run_additionals = parser.parse_known_args()
    return run_args, run_additionals


def prepare_train_kwargs(config: dict):
    """Deserizalise the input arguments for the train function.

    This is a hook that will allow us to prepare the input arguments to the train
    function. This might do steps like deserializer mixer circuits, ansatz circuits,
    etc.
    """
    for name in ["mixer", "initial_state", "ansatz_circuit"]:
        if name in config["train_kwargs"]:
            raise NotImplementedError(f"Serialization is not yet implemented for {name}.")


def set_problem_class_recursive(trainer_conf: dict, problem_class: str):
    """Recursively set problem_class in trainer config, including nested trainers.

    Args:
        trainer_conf: A trainer config dict or trainer_init dict to modify
        problem_class: The problem class name to set
    """
    if not isinstance(trainer_conf, dict):
        return

    if problem_class == "labs":
        apply_labs_training_config_overrides(trainer_conf)

    # Support being called on either a full trainer config (with "trainer_init") or directly
    # on a trainer_init dict.
    if "trainer_init" in trainer_conf and isinstance(trainer_conf["trainer_init"], dict):
        trainer_conf["trainer_init"]["_problem_class"] = problem_class
        set_problem_class_recursive(trainer_conf["trainer_init"], problem_class)
        return

    trainer_conf["_problem_class"] = problem_class

    # Check if there's a nested trainer (e.g., RecursionTrainer contains ScipyTrainer)
    if "trainer_init" in trainer_conf and isinstance(trainer_conf["trainer_init"], dict):
        set_problem_class_recursive(trainer_conf["trainer_init"], problem_class)


def set_objective_recursive(trainer_conf: dict, objective: str):
    """Recursively set objective for ScipyTrainer configs (including nested trainers).

    This is an in-memory override so users can switch objectives from the CLI without
    editing method JSON files.
    """
    if not isinstance(trainer_conf, dict):
        return

    if trainer_conf.get("trainer") == "ScipyTrainer":
        trainer_init = trainer_conf.get("trainer_init")
        if isinstance(trainer_init, dict):
            trainer_init["objective"] = objective

    for val in trainer_conf.values():
        if isinstance(val, dict):
            set_objective_recursive(val, objective)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    set_objective_recursive(item, objective)


def train(args: Optional[List]):
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

    # Load the training config and prepare the trainer.
    with open(args.config, "r") as fin:
        full_config = json.load(fin)

    # Store problem class info in config for trainers to access
    if class_str is not None:
        class_info = class_str.split(":")
        class_name = class_info[0].lower()
        full_config["_problem_class"] = class_name
    else:
        full_config["_problem_class"] = None

    trainer_chain_config = full_config["trainer_chain"]

    all_results, result = {}, {}

    all_results["args"] = vars(args)

    if pre_processor is not None:
        all_results["pre_processing"] = pre_processor.to_config()
    else:
        all_results["pre_processing"] = None

    # Convert to real for serialization since optimization problems are a diagonal Hc.
    all_results["cost_operator"] = [(l, np.real(c)) for l, c in input_problem.to_list()]

    # Save files specified from the cmd line override file names in the json config.
    save_file = getattr(args, "save_file", None)

    # Loop over all the trainers.
    for train_idx, conf in enumerate(trainer_chain_config):
        trainer_name = conf["trainer"]

        # Optional CLI override: objective for ScipyTrainer (including nested ScipyTrainer).
        objective = getattr(args, "objective", None)
        if objective is not None:
            set_objective_recursive(conf, objective)

        # Parse evaluator init key-word arguments given at runtime.
        evaluator_init_kwargs_str = getattr(args, f"evaluator_init_kwargs{train_idx}")
        if evaluator_init_kwargs_str is not None:
            # Handle standard trainers and recursive trainers (nested)
            trainer_init = conf["trainer_init"]
            target_init = None

            if "evaluator" in trainer_init:
                target_init = trainer_init
            elif "trainer_init" in trainer_init and "evaluator" in trainer_init["trainer_init"]:
                target_init = trainer_init["trainer_init"]

            if target_init:
                evaluator_cls = EVALUATORS[target_init["evaluator"]]
                evaluator_init = target_init.get("evaluator_init", dict())
                evaluator_init.update(evaluator_cls.parse_init_kwargs(evaluator_init_kwargs_str))
                target_init["evaluator_init"] = evaluator_init
            else:
                raise ValueError(
                    f"evaluator_init_kwargs{train_idx} given but no evaluator "
                    f"in trainer {trainer_name}."
                )

        trainer_cls = TRAINERS[trainer_name]
        # Pass problem class info to trainer config if available (including nested trainers)
        if "_problem_class" in full_config and full_config["_problem_class"] is not None:
            set_problem_class_recursive(conf, full_config["_problem_class"])
        trainer = trainer_cls.from_config(conf["trainer_init"])

        # Hook to deserialize any input to train that was serialized.
        prepare_train_kwargs(conf)

        # Get train args based on last result (if any).
        train_kwargs = conf["train_kwargs"]
        if len(result) > 0 and "result" in train_kwargs:
            for result_key, arg_name in train_kwargs["result"].items():
                train_kwargs[arg_name] = result[result_key]

            train_kwargs.pop("result")

        # Allows us to pass training key-word arguments at runtime.
        if hasattr(args, f"train_kwargs{train_idx}"):
            train_args_str = getattr(args, f"train_kwargs{train_idx}")
            cmd_train_kwargs = trainer.parse_train_kwargs(train_args_str)
            train_kwargs.update(cmd_train_kwargs)

        # Perform the optimization.
        result = trainer.train(input_problem, **train_kwargs)

        all_results[train_idx] = result

        if args.save:
            # Prepare the file where to save the result
            date_tag = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
            if save_file is None:
                save_file_local = date_tag + "_" + conf.pop("save_file")
            else:
                save_file_local = date_tag + "_" + save_file

            # If the directory is not existent, creates it
            if not os.path.exists(args.save_dir) and args.save_dir != "":
                os.makedirs(args.save_dir)

            with open(args.save_dir + save_file_local, "w") as fout:
                save_data = dict()
                for k, v in all_results.items():
                    save_data[k] = v.data if isinstance(v, ParamResult) else v

                json.dump(save_data, fout, indent=4)

    return all_results


if __name__ == "__main__":
    script_args, _ = get_script_args()
    train(script_args)
