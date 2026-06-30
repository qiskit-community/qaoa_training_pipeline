#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class that implements Trotterized Quantum Annealing"""


from qaoa_training_pipeline.training.functions import TQATrainerFunction
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


class TQATrainer(ScipyTrainer):
    """A trainer that implements the Trotterized Quantum Annealing (TQA) protocol.

    TQA sets the QAOA angles as a function of a single annealing-time parameter ``dt` using
    the schedule introduced in Sack and Serbyn, Quantum 5, 491 (2021).

    Because the angles are fully determined by ``dt``, the optimization happens over a
    one-dimensional space, which is performed via a ScipyTrainer with an underlying angle mapping
    provided by TQATrainerFunction.

    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        minimize_args: dict[str, object] | None = None,
        energy_minimization: bool = False,
    ):
        """Initialize the TQA trainer.

        Args:
            evaluator: The energy evaluator to compute the energy at each optimization step
            minimize_args: Arguments that will be passed to SciPy's `minimize`.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        super().__init__(
            evaluator, minimize_args, energy_minimization, TQATrainerFunction("tqa_schedule")
        )
