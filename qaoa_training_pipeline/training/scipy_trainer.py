#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module is an interface to SciPy's minimize function."""

from time import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.history_mixin import HistoryMixin
from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult


class ScipyTrainer(BaseTrainer, HistoryMixin):
    """A trainer that wraps SciPy's minimize function."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        minimize_args: Optional[Dict[str, Any]] = None,
        energy_minimization: bool = False,
    ):
        """Initialize the trainer.

        Args:
            evaluator: An instance of `BaseEvaluator` which will evaluate the enrgy
                of the QAOA circuit.
            minimize_args: Arguments that will be passed to SciPy's `minimize`.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        BaseTrainer.__init__(evaluator)
        HistoryMixin.__init__(self)

        self._minimize_args = {"method": "COBYLA"}

        minimize_args = minimize_args or {}
        self._minimize_args.update(minimize_args)

        # Sign to control whether we minimize or maximize the energy
        self._sign = 1 if energy_minimization else -1

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return self._sign == 1

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        params0: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> ParamResult:
        r"""Call SciPy's minimize function to do the optimization.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            params0: The initial point passed to the `minimize` function.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\gamma H_C)`.
        """
        self.recreate_history()

        start = time()

        self._energy_history = []
        self._parameter_history = []
        self._energy_evaluation_time = []

        def _energy(x):
            """Maximize the energy by minimizing the negative energy."""
            estart = time()
            energy = self._sign * self._evaluator.evaluate(
                cost_op=cost_op,
                params=x,
                mixer=mixer,
                initial_state=initial_state,
                ansatz_circuit=ansatz_circuit,
            )

            self._energy_evaluation_time.append(time() - estart)
            self._energy_history.append(self._sign * energy)
            self._parameter_history.append(list(val for val in x))

            return energy

        result = minimize(_energy, np.array(params0), **self._minimize_args)

        param_result = ParamResult.from_scipy_result(
            result, params0, time() - start, self._sign, self
        )
        param_result["energy_history"] = self._energy_history
        param_result["parameter_history"] = self._parameter_history
        param_result["energy_evaluation_time"] = self._energy_evaluation_time

        param_result.update(self._evaluator.get_results_from_last_iteration())

        return param_result

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        **plot_args,
    ):
        """Plot the energy history.

        Args:
            axis: The axis object on which to plot. If None is given then we create one.
            fig: The figure instance. If no axis are given then we create one.
            plot_args: Key word arguments that are given to axis.plot().

        Returns:
            An axis instance and figure handle. These are the inputs when given.
        """

        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        plot_style = {"lw": 2, "color": "dodgerblue"}
        plot_style.update(plot_args)

        axis.plot(self._energy_history, **plot_style)

        axis.set_xlabel("Iteration number")
        axis.set_ylabel("Energy")

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "ScipyTrainer":
        """Create a scipy trainer based on a config."""

        evaluator_cls = EVALUATORS[config["evaluator"]]
        evaluator = evaluator_cls.from_config(config["evaluator_init"])
        minimize_args = config["minimize_args"]

        return cls(evaluator, minimize_args)

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse any train arguments from a string.

        The only argument that can be contained here is params0. It is the values
        of the betas and gammas that make up the initial point given to Scipy's
        minimize function. We give `params0` as in string describing floats seperated
        by uderscores, e.g., 1.2_30.12_0.0
        """
        if args_str is None:
            return dict()

        return [float(val) for val in args_str.split("_")]

    def to_config(self) -> dict:
        """Creates a serializeable dictionary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        config = {
            "trainer_name": self.__class__.__name__,
            "evaluator": self._evaluator.to_config(),
        }

        config.update(self._minimize_args)

        return config
