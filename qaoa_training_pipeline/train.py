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
from typing import Optional, List
import json

import os

from datetime import datetime

from qaoa_training_pipeline.utils.data_utils import load_input, input_to_operator
from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.training import TRAINERS


def get_script_args():
    """Get the command line input arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="The path to the graph or hyper graph to train on.",
    )

    parser.add_argument(
        "--pre_factor",
        required=False,
        default=1.0,
        type=float,
        help="A pre-factor that multiply all the weights in the input.",
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


def train(args: Optional[List]):
    """Main function that does the training.

    The training is configurable based on system inputs. Use the `help` function to get
    a description of the arguments that are accepted. The training is done in an iterative
    fashion which allows one trainer to leverage the result of a previous trainer.
    """

    # Load the input.
    input_problem = input_to_operator(load_input(args.input), pre_factor=args.pre_factor)

    # Load the training config and prepare the trainer.
    with open(args.config, "r") as fin:
        full_config = json.load(fin)

    trainer_chain_config = full_config["trainer_chain"]

    all_results, result = {}, {}

    # Save files specified from the cmd line override file names in the json config.
    if hasattr(args, "save_file"):
        save_file = args.save_file
    else:
        save_file = None

    # Loop over all the trainers.
    for train_idx, conf in enumerate(trainer_chain_config):
        trainer_name = conf["trainer"]

        # Parse evaluator init key-word arguments given at runtime.
        evaluator_init_kwargs_str = getattr(args, f"evaluator_init_kwargs{train_idx}")
        if evaluator_init_kwargs_str is not None:
            if "evaluator" in conf["trainer_init"]:
                evaluator_cls = EVALUATORS[conf["trainer_init"]["evaluator"]]

                evaluator_init = conf["trainer_init"].get("evaluator_init", dict())
                evaluator_init.update(evaluator_cls.parse_init_kwargs(evaluator_init_kwargs_str))
                conf["trainer_init"]["evaluator_init"] = evaluator_init
            else:
                raise ValueError(
                    f"evaluator_init_kwargs{train_idx} given but no evaluator "
                    f"in trainer {trainer_name}."
                )

        trainer_cls = TRAINERS[trainer_name]
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
                json.dump({k: v.data for k, v in all_results.items()}, fout, indent=4)

    all_results["args"] = vars(args)
    return all_results


if __name__ == "__main__":
    script_args, _ = get_script_args()
    train(script_args)
