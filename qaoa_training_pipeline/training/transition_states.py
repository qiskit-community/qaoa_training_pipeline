# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transition states trainer."""

from time import time
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.parameter_scanner import DepthOneScanTrainer


class TransitionStatesTrainer(BaseTrainer):
    """A trainer that leverages transition states.

    The approach to training QAOA with transition states is described in Sack et al.
    Phys. Rev. A 107, 062404 (2024). Transition states in a QAOA are created from the local
    optima of p-layer QAOA by adding a QAOA layer. The optimized parameters of the p-layer
    QAOA are used to build a starting point to optimize the p+1-layer QAOA. If this starting
    point is properly constructed then it is guaranteed to be a transition state, i.e., a
    state where at least one of the directions in the optimization landscape will lead to an
    improved local extrema.
    """

    def __init__(self, trainer: BaseTrainer):
        """Initialize the Transition state trainer.

        Args:
            trainer: The trainer, with the evaluator inside of it, to optimize
                the parameters starting from the right initial points.
        """
        super().__init__(trainer.evaluator)
        self._trainer = trainer
        self._all_ts = None

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return self._trainer.minimization

    @property
    def transition_states_data(self) -> list:
        """The optimization results for all the transition states."""
        return self._all_ts

    @property
    def trainer(self):
        """Return the trainer with which each sub-point is trained."""
        return self._trainer

    # pylint: disable=arguments-differ
    def train(
        self,
        cost_op: SparsePauliOp,
        previous_optimal_point: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ):
        r"""Train the parameters based on a previous optimal point.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            previous_optimal_point: This is the optimal point of the Ansatz with
                one less layer. We will derive the Transition states from this input.
                The params are given in the order `[beta0, beta1, ..., gamma0, gamma1, ...].
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\gamma H_C)`.

        Returns:
            A dictionary with optimization results.
        """
        start = time()

        result, self._all_ts = dict(), []

        for idx, ts_state in enumerate(self.make_ts(previous_optimal_point)):
            res = self._trainer.train(cost_op, ts_state, mixer, initial_state, ansatz_circuit)
            res.update({"ts": ts_state})

            self._all_ts.append(res)

            result[f"ts{idx}"] = res

            if not self.minimization:
                keep = res["energy"] > result.get("energy", -np.inf)
            else:
                keep = res["energy"] < result.get("energy", np.inf)

            if keep:
                result["energy"] = res["energy"]
                result["optimized_params"] = res["optimized_params"]
                result["ts"] = ts_state

        result["trainer"] = self.to_config()
        result["train_duration"] = time() - start

        return result

    @staticmethod
    def make_ts(params: List[float]) -> List[List[float]]:
        """Create a list of transition states.

        This method takes the locally optimal parameters of a depth p QAOA and creates the
        transition states (TS) for the depth p+1 QAOA. The TS are found by inserting a zero
        when `i==j` or when `j==i+1` where `i` runs over the gamma parameters and `j` runs
        over the beta parameters. From a set of locally optimal QAOA parameters we generate
        a list of 2p+1 transition states.

        Args:
            params: The locally optimal parameters for a depth p QAOA

        Returns:
            A list of initial points (i.e., a list of floats) to try out in an optimization.
            Each sub-list is a transition point in the p+1-layer QAOA.
        """

        p = len(params) // 2
        betas = params[0:p]

        gammas = params[p:]

        transition_states = list()
        transition_states.append(betas + [0] + gammas + [0])

        # Case for adding zeros at index i (gammas) == j (betas)
        for idx1 in range(p):
            new_betas, new_gammas = [], []
            for idx2 in range(p + 1):
                if idx2 < idx1:
                    new_betas.append(betas[idx2])
                    new_gammas.append(gammas[idx2])
                elif idx2 == idx1:
                    new_betas.append(0)
                    new_gammas.append(0)
                else:
                    new_betas.append(betas[idx2 - 1])
                    new_gammas.append(gammas[idx2 - 1])

            transition_states.append(new_betas + new_gammas)

        # Case for adding zeros at index i + 1 (gammas) == j (betas)
        for idx1 in range(p):
            new_betas, new_gammas = [], []
            for idx2 in range(p + 1):
                if idx2 < idx1:
                    new_betas.append(betas[idx2])
                    new_gammas.append(gammas[idx2])
                elif idx2 == idx1:
                    new_betas.append(betas[idx2])
                    new_gammas.append(0)
                elif idx2 == idx1 + 1:
                    new_betas.append(0)
                    new_gammas.append(gammas[idx2 - 1])
                else:
                    new_betas.append(betas[idx2 - 1])
                    new_gammas.append(gammas[idx2 - 1])

            transition_states.append(new_betas + new_gammas)

        return transition_states

    def plot(
        self,
        parameter_history: bool = False,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        colors: Optional[list] = None,
        **plot_args,
    ):
        """Plot the energy history of the different optimizations.

        Args:
            parameter_history: If True plot the value of the parameters during the optimization.
            axis: The axis object on which to plot. If None is given then we create one.
            fig: The figure instance. If no axis are given then we create one.
            plot_args: Key word arguments that are given to axis.plot().
            colors: A list of colors for the plots. If the parameters are plotted then the line
                style is set by the index of the transition state and the color encodes the
                parameter index.

        Returns:
            An axis instance and figure handle. These are the inputs when given.
        """
        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        plot_style = {"lw": 2}
        plot_style.update(plot_args)

        colors = colors or list(mcolors.TABLEAU_COLORS.values())

        if parameter_history:
            line_styles = ["-", ":", "--", "-."]

            for idx, result in enumerate(self._all_ts):
                params_hist = result["parameter_history"]

                for param_idx in range(len(params_hist[0])):
                    axis.plot(
                        [param[param_idx] for param in params_hist],
                        **plot_style,
                        label=f"TS{idx} param. {param_idx}",
                        ls=line_styles[idx % 4],
                    )
        else:
            for idx, result in enumerate(self._all_ts):
                axis.plot(
                    result["energy_history"],
                    **plot_style,
                    label=f"TS{idx}",
                    color=colors[idx % len(colors)],
                )

        axis.set_xlabel("Iteration number")
        axis.set_ylabel("Energy")

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "TransitionStatesTrainer":
        """Create a transition state trainer from a config."""

        trainer_name = config["trainer"]

        # Note: we cannot user the TRAINERS mapping otherwise we will circular import outselves.
        if trainer_name == "ScipyTrainer":
            return cls(ScipyTrainer.from_config(config["trainer_init"]))
        elif trainer_name == "DepthOneScanTrainer":
            return cls(DepthOneScanTrainer.from_config(config["trainer_init"]))
        else:
            raise ValueError(f"Unrecognized trainer {trainer_name}")

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse a string into the training kwargs.

        Note: This trainer does not take any arguments aside from the default
        ones. We therefore ask the sub-trainer to parse the key word arguments.
        """
        if args_str is None:
            return dict()

        return self._trainer.parse_train_kwargs(args_str)
