#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to generate a beta and gamma schedule based on TQA."""

from time import time
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.history_mixin import HistoryMixin
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult


class TQATrainer(BaseTrainer, HistoryMixin):
    """Trotterized Quantum Annealing parameter generation.

    This trainer is based on the Trotterized Annealing parameters initialization
    of QAOA as presented in "Quantum annealing initialization of the quantum approximate
    optimization algorithm" published in Quantum 5, 491 (2021). It is usually intended to
    be used as an initial point generator which is independent of problem instance and
    does not do any optimization. However, we can also use this class to perform a
    SciPy optimization of the end point of the TQA schedule.
    """

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        minimize_args: Optional[Dict[str, Any]] = None,
        energy_minimization: bool = False,
    ) -> None:
        """Initialize an instance.

        Args:
            evaluator: If an evaluator is given then we will try and optimize the
                end points of the TQA schedule using SciPy.
            minimize_args: Arguments for the minimization. By default, we assume COBYLA with
                a small step size since the default TQA value of 0.75 is usually very good.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        BaseTrainer.__init__(evaluator)
        HistoryMixin.__init__(self)

        self._minimize_args = {"method": "COBYLA", "options": {"maxiter": 20, "rhobeg": 0.1}}

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
        reps: int,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> ParamResult:
        """Train the QAOA parameters."""
        self.reset_history()

        def _energy(x):
            """Optimize the energy by minimizing the negative energy.

            If self._sign is -1, i.e., the default set by `energy_minimization`
            then the scipy.minimize is converted into a maximization.
            """
            estart = time()
            energy = self._sign * self._evaluator.evaluate(
                cost_op=cost_op,
                params=self.tqa_schedule(reps, dt=x[0]),
                mixer=mixer,
                initial_state=initial_state,
                ansatz_circuit=ansatz_circuit,
            )

            self._energy_evaluation_time.append(time() - estart)
            self._energy_history.append(self._sign * energy)
            self._parameter_history.append(list(val for val in x))

            return energy

        start = time()

        if self.evaluator is None:
            tqa_dt = 0.75
            param_result = ParamResult(
                self.tqa_schedule(reps, dt=tqa_dt), time() - start, self, None
            )
        else:
            params0 = [0.75]
            result = minimize(_energy, params0, **self._minimize_args)
            param_result = ParamResult.from_scipy_result(
                result, params0, time() - start, self._sign, self
            )
            param_result["optimized_params"] = self.tqa_schedule(
                reps, dt=param_result["optimized_params"]
            )

        param_result["energy_history"] = self._energy_history
        param_result["parameter_history"] = self._parameter_history
        param_result["energy_evaluation_time"] = self._energy_evaluation_time

        return param_result

    @staticmethod
    def tqa_schedule(reps: int, dt: float) -> np.array:
        """Create the TQA schedule."""
        grid = np.arange(1, reps + 1) - 0.5
        return np.concatenate((1 - grid * dt / reps, grid * dt / reps)).tolist()

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        **plot_args,
    ):
        """Plot the optimization."""
        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        plot_style = {"lw": 2, "color": "dodgerblue"}
        plot_style.update(plot_args)

        if len(self._energy_history) > 0:
            line1 = axis.plot(self._energy_history, **plot_style, label="Energy")

            axis2 = axis.twinx()
            plot_style["color"] = "forestgreen"
            line2 = axis2.plot(
                [val[0] for val in self._parameter_history], **plot_style, label="TQA dt"
            )

            axis.set_xlabel("Iteration")
            axis.set_ylabel("Energy")
            axis2.set_ylabel("TQA dt value")
            axis.legend(line1 + line2, [line1[0].get_label(), line2[0].get_label()])

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "TQATrainer":
        """Create a class from a config."""
        evaluator = None
        if "evaluator" in config:
            evaluator_cls = EVALUATORS[config["evaluator"]]
            evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(evaluator, config.get("minimize_args", {}))

    def to_config(self) -> dict:
        """Creates a serializeable dictionary to keep track of how results are created.

        Note: This datastructure is not intended for us to recreate the class instance.
        """
        evaluator_str = "None"
        if self._evaluator is not None:
            evaluator_str = self._evaluator.to_config()

        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": evaluator_str,
        }

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Extract training key word arguments from a string.

        The input args are only the number of repetitions. There the args_str is only a single int.
        """
        if args_str is None:
            return dict()

        return {"reps": int(args_str)}
