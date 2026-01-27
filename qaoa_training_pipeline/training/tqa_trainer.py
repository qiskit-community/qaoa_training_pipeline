#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to generate a beta and gamma schedule based on TQA."""

from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.functions import BaseAnglesFunction
from qaoa_training_pipeline.training.history_mixin import HistoryMixin
from qaoa_training_pipeline.training.param_result import ParamResult


class TQATrainerFunction(BaseAnglesFunction):
    """Wrapper function around TQATrainer.tqa_schedule.

    This function allows for a default ``rep`` value to be set, which is passed
    to ``tqa_schedule``. If no value has been set yet, and no value is provided
    to :meth:`__call__`, then an error is raised.
    """

    def __init__(self, tqa_schedule_method: Callable, reps: int | None = None) -> None:
        """Create an instance of a TQATrainer QAOA angles function.

        Args:
            tqa_schedule_method: The :meth:`TQATrainer.tqa_schedule` method for
                an instance of :class:`TQATrainer`.
            reps: The default reps to use if not overridden by
                :meth:`TQATrainer.train`. If None and ``reps`` is not provided
                in :meth:`__call__`, an error is raised. Defaults to None.
        """
        super().__init__()
        self._tqa_schedule = tqa_schedule_method
        self.reps = reps

    # pylint: disable=unused-argument
    def __call__(self, x: list, reps: int | None = None) -> list:
        if reps is None:
            reps = self.reps
        if reps is None:
            raise ValueError(
                f"reps must be provided to {self.__class__.__name__}(reps=...) or "
                + "set with trainer.train(..., reps=...)"
            )
        return self._tqa_schedule(reps=reps, dt=x)

    # pylint: disable=unused-argument
    @classmethod
    def from_config(cls, config: dict) -> None:
        """Create a TQATrainer from a config dictionary."""
        raise RuntimeError(f"{cls.__name__} cannot be constructed from a config.")


# cspell: ignore Trotterized maxiter rhobeg
class TQATrainer(BaseTrainer, HistoryMixin):
    """Trotterized Quantum Annealing parameter generation.

    This trainer is based on the Trotterized Annealing parameters initialization
    of QAOA as presented in "Quantum annealing initialization of the quantum approximate
    optimization algorithm" published in Quantum 5, 491 (2021). It is usually intended to
    be used as an initial point generator which is independent of problem instance and
    does not do any optimization. However, we can also use this class to perform a
    SciPy optimization of the end point of the TQA schedule.

    Additionally, the trainer accepts a Linear Ramp parameters selection of QAOA as
    presented in "Towards a Linear-Ramp QAOA protocol: Evidence of a scaling advantage in
    solving some combinatorial optimization problems" published in npj Quantum Information
    11, 131 (2025). It can be used to perform a SciPy optimization of the beta and gamma slopes.


    .. note::

        :attr:`qaoa_angles_function` for :class:`TQATrainer` requires knowledge
        of the number of repetitions. ``reps`` can be provided by calling
        ``qaoa_angles_function(params, reps=reps)``. If no value for ``reps`` is
        provided, the value for the most recent call to :meth:`train` will be
        used. If :meth:`train` has not been called yet for the instance of
        :class:`TQATrainer`, then an error is raised.
    """

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        minimize_args: Optional[Dict[str, Any]] = None,
        energy_minimization: bool = False,
        initial_dt: tuple[float, float] | list[float] | None = None,
    ) -> None:
        """Initialize an instance.

        Args:
            evaluator: If an evaluator is given then we will try and optimize: (1) the
                end points of the TQA schedule using SciPy if initial_dt is a float.
                (2) The slopes of the Linear Ramp schedules if initial_dt is Tuple[float, float].
            minimize_args: Arguments for the minimization. By default, we assume COBYLA with
                a small step size since the default TQA value of 0.75 is usually very good.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
            initial_dt: Initial dt if not provided to :meth:`train`. Defaults to
                ``0.75``.
        """
        initial_dt = initial_dt or [0.75]
        initial_dt_list = [initial_dt] if isinstance(initial_dt, float) else initial_dt

        schedule_method = self.tqa_schedule if len(initial_dt_list) == 1 else self.lr_schedule

        BaseTrainer.__init__(
            self,
            evaluator,
            qaoa_angles_function=TQATrainerFunction(schedule_method, reps=None),
        )
        self.qaoa_angles_function: TQATrainerFunction
        self._qaoa_angles_function: TQATrainerFunction
        HistoryMixin.__init__(self)

        self._minimize_args = {"method": "COBYLA", "options": {"maxiter": 20, "rhobeg": 0.1}}
        self.initial_dt = initial_dt

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
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        reps: int | None = None,
        initial_dt: Tuple[float, float] | List[float] | None = None,
    ) -> ParamResult:
        """Train the QAOA parameters."""
        if reps is None:
            raise ValueError(f"class {self.__class__.__name__} requires reps to be set.")
        self.reset_history()

        initial_dt = initial_dt or self.initial_dt

        initial_dt = [initial_dt] if isinstance(initial_dt, float) else initial_dt

        if (
            len(initial_dt) == 2
            and self.qaoa_angles_function._tqa_schedule.__name__ != "lr_schedule"
        ):
            raise ValueError("initial_dt must be a tuple of two floats for Linear Ramp schedule.")
        if (
            len(initial_dt) == 1
            and self.qaoa_angles_function._tqa_schedule.__name__ != "tqa_schedule"
        ):
            raise ValueError("initial_dt must be a single float for TQA schedule.")
        # Set the reps attribute on the angles function, if it supports one.
        # This allow us to override it later with a function that doesn't
        # require reps. We set reps here so that ParamResult.from_scipy_result
        # correctly populates "optimized_qaoa_angles" and so we can call
        # `trainer.qaoa_angles_function()` in scripts.
        if hasattr(self._qaoa_angles_function, "reps"):
            self._qaoa_angles_function.reps = reps

        def _energy(x):
            """Optimize the energy by minimizing the negative energy.

            If self._sign is -1, i.e., the default set by `energy_minimization`
            then the scipy.minimize is converted into a maximization.
            """
            e_start = time()
            assert self._evaluator, "_evaluator must be defined before calling _energy()"
            energy = self._sign * self._evaluator.evaluate(
                cost_op=cost_op,
                params=self.qaoa_angles_function(x),
                mixer=mixer,  # type: ignore
                initial_state=initial_state,
                ansatz_circuit=ansatz_circuit,
            )

            energy = float(energy)
            self._energy_evaluation_time.append(time() - e_start)
            self._energy_history.append(self._sign * energy)
            self._parameter_history.append(list(float(val) for val in x))

            return energy

        start = time()

        if self.evaluator is None:
            param_result = ParamResult(list(initial_dt), time() - start, self, None)
        else:
            result = minimize(_energy, initial_dt, **self._minimize_args)
            param_result = ParamResult.from_scipy_result(
                result, initial_dt, time() - start, self._sign, self
            )
        param_result.add_history(self)

        return param_result

    @staticmethod
    def tqa_schedule(reps: int, dt: Tuple[float] | float) -> np.ndarray:
        """Create the TQA schedule."""
        dt = dt[0] if isinstance(dt, tuple) else dt
        grid = np.arange(1, reps + 1) - 0.5
        return np.concatenate((1 - grid * dt / reps, grid * dt / reps)).tolist()

    @staticmethod
    def lr_schedule(reps: int, dt: Tuple[float, float]):
        """Create the Linear Ramp schedule."""
        betas = np.arange(1, reps + 1)[::-1] * dt[0] / reps
        gammas = np.arange(1, reps + 1) * dt[1] / reps
        return np.concatenate((betas, gammas)).tolist()

    def plot(
        self,
        axis: Optional[Axes] = None,
        fig: Optional[Figure] = None,
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

            if len(self._parameter_history[0]) == 1:
                plot_style["color"] = "forestgreen"
                line2 = axis2.plot(
                    [val[0] for val in self._parameter_history], **plot_style, label="TQA dt"
                )
                axis2.set_ylabel("TQA dt value")
                axis.legend(line1 + line2, [str(line1[0].get_label()), str(line2[0].get_label())])

            elif len(self._parameter_history[0]) == 2:
                plot_style["color"] = "tab:green"
                line2 = axis2.plot(
                    [val[0] for val in self._parameter_history],
                    **plot_style,
                    label=r"$\Delta_{\beta}$",
                )
                plot_style["color"] = "tab:red"
                line3 = axis2.plot(
                    [val[1] for val in self._parameter_history],
                    **plot_style,
                    label=r"$\Delta_{\gamma}$",
                )
                axis2.set_ylabel("LR slope values")
                axis.legend(
                    line1 + line2 + line3,
                    [str(line[0].get_label()) for line in [line1, line2, line3]],
                )

            axis.set_xlabel("Iteration")
            axis.set_ylabel("Energy")

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "TQATrainer":
        """Create a class from a config."""
        evaluator = None
        if "evaluator" in config:
            evaluator_cls = EVALUATORS[config["evaluator"]]
            evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(
            evaluator,
            config.get("minimize_args", {}),
            config.get("energy_minimization", False),
            config.get("initial_dt", 0.75),
        )

    def to_config(self) -> dict:
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
        """Extract training key word arguments from a string.

        The input args are only the number of repetitions. There the args_str is only a single int.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key in ["reps"]:
                train_kwargs[key] = int(val)
            elif key == "initial_dt":
                train_kwargs[key] = self.extract_list(val, dtype=float)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs
