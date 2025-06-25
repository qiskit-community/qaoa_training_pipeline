#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to load the known angles of the fixed-angle conjecture."""

import json
import os
import warnings
from time import time
from typing import Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class FixedAngleConjecture(BaseTrainer):
    """Fixed angle conjecture.

    This class is an interface to load the known angles of the fixed angle conjecture.
    These angles are given by the following work:

    * Jonathan Wurtz and Peter Love, Phys. Rev. A 103, 042612 (2021)
    * Jonathan Wurtz and Danylo Lykov, Phys. Rev. A 104, 052419 (2021)

    In particular, the angles loaded by the class are obtained from
    https://github.com/danlkv/fixed-angle-QAOA/blob/master/angles_regular_graphs.json

    These angles are designed for maximum cut problems on random k-regular graphs.
    The angles provided are for various QAOA depths p from 1 up to at most 11, and for k ranging
    from 3 to 11. The angles are obtained by optimizing the QAOA angles for tree graphs of the
    corresponding degree. The provided approximation ratios are lower bounds to the approximation
    ratios that can be obtained in practice.
    """

    def __init__(self, evaluator: Optional[BaseEvaluator] = None):
        """Setup the class and load the parameters.

        Args:
            evaluator: If an evaluator is provided the energy of the cost operator at the fixed
                angles will be evaluated and included in the result of `train`.
        """
        super().__init__(evaluator)

        self._data = None
        data_path = os.path.join(os.path.dirname(__file__), "data", "fixed_angle_conjecture.json")
        with open(data_path, "r") as fin:
            self._data = json.load(fin)

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        raise NotImplementedError(
            f"Optimization is currently not implemented by {self.__class__.__name__}."
        )

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        reps: int = 1,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ):
        """Load the fixed-angles based on the degree of the graph in the cost operator.

        This method will extract a graph from the provided cost operator and compute the
        average node degree and round it to the closest integer. This allows the method
        to provide angles for non-regular graphs. However, the performance of such an
        approximation has not yet been tested.
        """

        if mixer is not None:
            warnings.warn(
                f"{self.__class__.__name__} ignores the mixer input. " "Returning standard angles."
            )

        if initial_state is not None:
            warnings.warn(
                f"{self.__class__.__name__} ignores the initial_state input. "
                "Returning standard angles."
            )

        if ansatz_circuit is not None:
            warnings.warn(
                f"{self.__class__.__name__} ignores the ansatz_circuit input. "
                "Returning standard angles."
            )

        start = time()

        # Since we only care about the degree of the graph the pre_factor is irrelevant.
        graph = operator_to_graph(cost_op)

        avg_degree = np.average([degree for _, degree in graph.degree()])

        degree_key = str(int(np.round(avg_degree)))

        if degree_key not in self._data:
            raise NotImplementedError(
                f"The degree {degree_key} is not in the fixed-angle conjecture database. "
                "In the future this class may compute the desired fixed angles."
            )

        if str(reps) not in self._data[degree_key]:
            raise NotImplementedError(
                f"The desired QAOA depth {reps} is not in the fixed-angle conjecture database. "
                "In the future this class may compute the desired fixed angles."
            )

        angles_data = self._data[degree_key][str(reps)]

        energy = None
        if self._evaluator is not None:
            energy = self._evaluator.evaluate(cost_op, angles_data["beta"] + angles_data["gamma"])

        return {
            "optimized_params": angles_data["beta"] + angles_data["gamma"],
            "energy": energy,
            "trainer": self.to_config(),
            "train_duration": time() - start,
            "approximation ratio": angles_data["AR"],
        }

    @classmethod
    def from_config(cls, config: dict) -> "FixedAngleConjecture":
        """Create a class from a config."""
        evaluator = None
        if "evaluator" in config:
            evaluator_cls = EVALUATORS[config["evaluator"]]
            evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(evaluator)

    def to_config(self) -> dict:
        """Creates a serializeable dictionary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": None,
        }

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Extract training key word arguments from a string.

        The input args are only the number of repetitions. There the args_str is only a single int.
        """
        if args_str is None:
            return dict()

        return {"reps": int(args_str)}
