#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class that implements Trotterized Quantum Annealing"""


from __future__ import annotations

from typing import TYPE_CHECKING

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.functions import TQAFunction
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer

if TYPE_CHECKING:

    pass


class TQATrainer(ScipyTrainer):
    """A trainer that implements the Trotterized Quantum Annealing (TQA) protocol.

    TQA sets the QAOA angles as a function of a single annealing-time parameter ``dt` using
    the schedule introduced in Sack and Serbyn, Quantum 5, 491 (2021).

    Because the angles are fully determined by ``dt``, the optimization happens over a
    one-dimensional space, which is performed via a ScipyTrainer with an underlying angle mapping
    provided by TQAFunction.

    """

    def __init__(
        self,
        reps: int | None = None,
        evaluator: BaseEvaluator | None = None,
        minimize_args: dict[str, object] | None = None,
        energy_minimization: bool = False,
    ):
        """Initialize the TQA trainer.

        Args:
            evaluator: The energy evaluator to compute the energy at each optimization step
            reps: the QAOA circuit depth.
            minimize_args: Arguments that will be passed to SciPy's `minimize`.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        super().__init__(
            evaluator,
            minimize_args,
            energy_minimization,
            TQAFunction(
                reps=reps,
            ),
        )

    @classmethod
    def from_config(cls, config: dict) -> "TQATrainer":
        """Create a scipy trainer based on a config."""

        evaluator_cls = EVALUATORS[config["evaluator"]]

        return cls(
            evaluator=evaluator_cls.from_config(config["evaluator_init"]),
            reps=config["reps"],
            minimize_args=config.get("minimize_args", None),
            energy_minimization=config.get("energy_minimization", False),
        )
