#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A model for depth-one QAOA parameters on random regular graphs."""

from collections.abc import Callable
import json
from importlib import resources
from time import time
from typing import Dict, NoReturn, Optional
import numpy as np
from networkx.classes.reportviews import DegreeView

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.exceptions import TrainingError
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.models.power_function import PowerFunction
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class RandomRegularDepthOneFit(BaseTrainer):
    """This class wraps a model to transfer depth 1 parameters.

    The class works by transferring parameters of learned graphs to the new instance.
    This is done by computing the average degree of the nodes in the graph and then
    selecting the model for the graph with the closest regularity.

    Important note: The default model on which this class relies only supports graphs
    with regularity `k` in  `[3, 4, 5, 7, 8, 9]`. In the future this should be extended
    to an arbitrary `k`.
    """

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        model_file: Optional[str] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            evaluator: The evaluator is optional. If it is given then we evaluate
                the energy of the recommended parameters. By not providing an evaluator
                we allow this class to make a fast parameter recommendation, albeit without
                knowing the resulting energy.
            model_file: The file from which to load the model coefficients. If this
                is not given we default to the coefficients in this repository.
        """
        super().__init__(evaluator)

        from qaoa_training_pipeline import models

        default_model_file = str(resources.files(models) / "regular_depth1.json")
        model_file = model_file or default_model_file

        self._functions = self._load_model(model_file)

    @staticmethod
    def _load_model(file_name: str) -> Dict[str, Dict[int, Callable]]:
        """Creates the functions and stores them in a dict.

        In this model, to each `k` regular graph are associated `PowerFunction`s whose argument
        is the number of nodes `n` in the graph. Each function outputs one parameter such as beta
        or gamma. The functions are loaded from a file. Therefore, in the returned data structure,
        the first key is the parameter and the second key is the graph order. The value of the
        inner dict is the power function. An example, of this data structure is

        ..parsed-literal::

            "beta1": {
                "degree": {
                    "3": {
                        "-1": -0.20716087,
                        "-2": -0.094972,
                        "0": 0.39203969
                    }
                }
            }

        Here, `beta1` indicates the beta of depth-one QAOA. The degree key makes it clear that
        we are storing data following the degree of the random `k` regular graphs in the nested
        dictionary. Finally, the keys in the inner most dictionary are the power `a` and the
        values are the coefficients `b` of the expression `b * n**a` where `n` is the number
        of nodes in the graphs.

        Args:
            file_name: The name of the file where to find the model coefficients.
                If this is not specified the method will default to an already
                existing model in this repository.
        """
        with open(file_name, "r") as fin:
            data = json.load(fin)

        functions = {"beta1": dict(), "gamma1": dict()}
        for param in ["beta1", "gamma1"]:
            for degree, function_data in data[param]["degree"].items():
                functions[param][int(degree)] = PowerFunction(function_data)

        return functions

    # pylint: disable=arguments-differ
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> Dict:
        """Train based on a model.

        Args:
            cost_op: The cost operator. This will be converted to a graph.
            mixer: The mixer operator. This is not supported.
            initial_state: The initial state which is not supported.
            ansatz_circuit: The ansatz circuit in case it differs from the standard
                QAOA circuit. Also not supported.
        """
        start = time()

        graph = operator_to_graph(cost_op)

        # Get the average degree of the graph and round it to the closest integer.
        assert isinstance(graph.degree, DegreeView)
        degree = round(np.average(list(deg[1] for deg in graph.degree)))

        # get the number of nodes
        order = float(graph.order())

        if any(degree not in self._functions[param] for param in ["beta1", "gamma1"]):
            raise TrainingError(
                f"The model does not support degree {degree}. Please extend the model."
            )

        optimized_params = [self._functions[param][degree](order) for param in ["beta1", "gamma1"]]

        energy = None

        # Optionally, evaluate the energy if an evaluator is provided.
        if self.evaluator is not None:
            energy = self.evaluator.evaluate(
                cost_op,
                params=optimized_params,
                mixer=mixer,
                initial_state=initial_state,
                ansatz_circuit=ansatz_circuit,
            )

        return {
            "optimized_params": optimized_params,
            "trainer": self.to_config(),
            "train_duration": time() - start,
            "energy": energy,
        }

    @classmethod
    def from_config(cls, config: dict) -> "RandomRegularDepthOneFit":
        """Create the class from a config file."""
        evaluator = None
        if "evaluator" in config:
            evaluator_cls = EVALUATORS[config["evaluator"]]
            evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(evaluator, config.get("model_file", None))

    def to_config(self) -> Dict:
        """Creates a serializable dictionary to keep track of how results are created.

        Note: This data structure is not intended for us to recreate the class instance.
        """
        evaluator_str = "None"
        if self._evaluator is not None:
            evaluator_str = self._evaluator.to_config()

        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": evaluator_str,
        }

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse train arguments. In this case there aren't any."""
        return dict()

    @property
    def minimization(self) -> NoReturn:
        """Fits for random regular problems neither minimizes nor maximizes."""
        raise ValueError(f"{self.__class__.__name__} neither minimizes nor maximizes.")
