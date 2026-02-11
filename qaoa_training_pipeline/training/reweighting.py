#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trainer for weighted graphs."""

from time import time
from typing import Optional

import numpy as np
from networkx.classes.reportviews import DegreeView
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.training.parameter_scanner import DepthOneScanTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.transition_states import TransitionStatesTrainer
from qaoa_training_pipeline.utils.graph_utils import operator_to_graph


class ReweightingTrainer(BaseTrainer):
    """Train the parameters by unweighting and reweighting the graph."""

    def __init__(self, trainer1: BaseTrainer, trainer2: Optional[BaseTrainer] = None) -> None:
        """Initialize the instance.

        Args:
            trainer1: the trainer that does the training on the unweighted graph.
            trainer2: the trainer the does the training on the weighted graph based on the optimal
                point that trainer1 found. Note that this trainer must accept an initial point
                in its key-word arguments of the `train` function. Otherwise, the results of the
                first training phase is neglected. Note that if the second trainer is not given
                the class will default to a `ScipyTrainer`.
        """
        super().__init__(trainer1.evaluator)

        self._trainer_unweighted = trainer1
        assert isinstance(self.evaluator, BaseEvaluator)
        valid = True

        if trainer2 is None:
            self._trainer_weighted = ScipyTrainer(
                evaluator=self.evaluator,
                energy_minimization=trainer1.minimization,
            )
        else:
            self._trainer_weighted = trainer2

        try:
            valid = self._trainer_unweighted.minimization == self._trainer_weighted.minimization
        except ValueError:
            pass

        if not valid:
            raise ValueError(
                "Conflicting minimization and maximization in the weighted and unweighted trainers."
            )

        self._edge_weights = None
        self._average_degree = None

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return self._trainer_weighted.minimization

    @property
    def unweighted_trainer(self) -> BaseTrainer:
        """Return the trainer of the unweighted graph."""
        return self._trainer_unweighted

    @property
    def weighted_trainer(self) -> BaseTrainer:
        """Return the trainer of the weighted graph."""
        return self._trainer_weighted

    # pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        params0: list[float] | None = None,
        trainer1_kwargs: Optional[dict] = None,
    ) -> ParamResult:
        r"""Train by unweighting the cost_op and then reweighting it.

        First, the unweighted trainer is called to obtain a set of optimal parameters.
        Next, we call the weighted trainer to optimize the parameters based on the
        rescaled optimal parameters found by the unweighted trainer.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\gamma H_C)`.
            trainer1_kwargs: Additional keyword arguments that will be given to the
                subtrainer.
        """
        start = time()

        unweighted_cost_op = self.unweight(cost_op)

        trainer1_kwargs = trainer1_kwargs or {}
        result1 = self._trainer_unweighted.train(
            unweighted_cost_op,
            mixer=mixer,
            initial_state=initial_state,
            ansatz_circuit=ansatz_circuit,
            **trainer1_kwargs,
        )

        params0 = self.scale_parameters(result1)

        result2 = self._trainer_weighted.train(cost_op, params0=params0)

        # Add to the result the intermediate step ParamResult is not serializable but its dict is.
        result2["unweighted_optimization"] = result1.data
        result2["scaled_initial_point"] = params0
        result2["trainer"] = self.to_config()
        result2["train_duration"] = time() - start
        return result2

    def unweight(self, cost_op: SparsePauliOp) -> SparsePauliOp:
        """Replace all the weights in the cost_op by 1.

        Note: this function also sets some internal variables that we will use to
        compute a scaling factor to scale the parameters obtained by the first trainer.
        """
        # Compute the average degree of the node. nx.degree is insensitive to weights.
        graph = operator_to_graph(cost_op)
        assert isinstance(graph.degree, DegreeView)
        self._average_degree = np.average(list(val for _, val in graph.degree))

        # Extract the edge weights. We will need them later on.
        self._edge_weights = [
            np.real(pauli.coeffs[0])
            for pauli in cost_op
            if pauli.coeffs and abs(pauli.coeffs[0]) > 1e-16
        ]

        # Define the new operator.
        new_op = [
            (pauli.paulis.to_labels()[0], 1)
            for pauli in cost_op
            if pauli.coeffs and abs(pauli.coeffs[0]) > 1e-16
        ]
        return SparsePauliOp.from_list(new_op)

    def scale_parameters(self, result: ParamResult) -> list:
        """Rescale the optimal gamma before optimizing again.

        The rescaling is done following the works of Sureshbabu et al. in Quantum 8,
        1231 (2024). In particular, see Theorem 3 (informal) on page 2.
        """
        assert self._edge_weights, "_edge_weights must be defined before calling scale_parameters()"
        scale = 1 / np.sqrt(
            sum(weight**2 for weight in self._edge_weights) / len(self._edge_weights)
        )

        params = result["optimized_params"]
        p = len(params) // 2
        betas = params[:p]
        gammas = params[p:]

        return betas + [gamma * scale for gamma in gammas]

    @staticmethod
    def _trainer_from_config(name: str, trainer_init: dict) -> BaseTrainer:
        """Initialize an instance of a trainer."""

        # Note: we cannot user the TRAINERS mapping otherwise we will circular import outselves.
        if name == "ScipyTrainer":
            return ScipyTrainer.from_config(trainer_init)
        elif name == "DepthOneScanTrainer":
            return DepthOneScanTrainer.from_config(trainer_init)
        elif name == "TransitionStatesTrainer":
            return TransitionStatesTrainer.from_config(trainer_init)

        raise ValueError(f"Unrecognized trainer {name}.")

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse a string into the training kwargs."""
        if args_str is None:
            return dict()
        else:
            raise NotImplementedError("ReweightingTrainer does not support input arg parsing.")

    @classmethod
    def from_config(cls, config: dict) -> "ReweightingTrainer":
        """Create a class instance from a trainer."""

        trainer1_name = config["trainer1"]
        trainer1 = cls._trainer_from_config(trainer1_name, config["trainer1_init"])

        trainer2 = None
        if "trainer2" in config:
            trainer2_name = config["trainer2"]
            trainer2 = cls._trainer_from_config(trainer2_name, config["trainer2_init"])

        return cls(trainer1, trainer2)

    def to_config(self) -> dict:
        """Creates a serializeable dictionnary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        return {
            "trainer_name": self.__class__.__name__,
            "trainer_unweighted": self._trainer_unweighted.to_config(),
            "trainer_weighted": self._trainer_weighted.to_config(),
        }
