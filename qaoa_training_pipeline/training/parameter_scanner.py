#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to scan gamma and beta in depth-one QAOA to get the optimal point."""

from time import time
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.extrema_location import Argmax, Argmin
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult
from qaoa_training_pipeline.evaluation import EVALUATORS


class DepthOneScanTrainer(BaseTrainer):
    """Scan the gamma and beta parameters of QAOA."""

    def __init__(self, evaluator: BaseEvaluator, energy_minimization: bool = False):
        """Initialize the class instance.

        Args:
            evaluator: The evaluator that computes the energy.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.
        """
        super().__init__(evaluator=evaluator)

        # Parameters that will be filled by the scanner.
        self._energies = None
        self._gammas = None
        self._betas = None

        # This could be set in a subsequent PR by other methods, e.g., interpolation.
        # This is a callable that takes as input the 2D energies scan.
        self._extrema_locator = Argmin() if energy_minimization else Argmax()

        self._opt_beta = None
        self._opt_gamma = None

        # Default parameter range over which to scan.
        self._default_range = ((0, np.pi), (0, 2 * np.pi))

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return isinstance(self._extrema_locator, Argmin)

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        parameter_ranges: Optional[List[Tuple[float, float]]] = None,
        num_points: Optional[int] = 15,
    ) -> ParamResult:
        r"""Train the parameters by doing a 2D scan.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\gamma H_C)`.
            parameter_ranges:  The parameter ranges in beta and gamma over which to scan. If
                this argument is not provided we default to `((np.pi), (0, 2 * np.pi))`.
            num_points: The number of points in the gamma and beta ranges to take. This
                method will thus evaluate the energy `num_points**2` times.
        """
        start = time()

        parameter_ranges = parameter_ranges or self._default_range

        self._energies = np.zeros((num_points, num_points), dtype=float)
        self._betas = np.linspace(parameter_ranges[0][0], parameter_ranges[0][1], num_points)
        self._gammas = np.linspace(parameter_ranges[1][0], parameter_ranges[1][1], num_points)

        self._opt_gamma, self._opt_beta = None, None

        for idx1, beta in enumerate(self._betas):
            for idx2, gamma in enumerate(self._gammas):
                energy = self._evaluator.evaluate(
                    cost_op,
                    [beta, gamma],
                    mixer,
                    initial_state,
                    ansatz_circuit,
                )
                self._energies[idx1, idx2] = np.real(energy)

        min_idx, opt_energy = self._extrema_locator(self._energies)
        min_idxb, min_idxg = min_idx // num_points, min_idx % num_points
        opt_beta, opt_gamma = self._betas[min_idxb], self._gammas[min_idxg]

        self._opt_gamma = opt_gamma
        self._opt_beta = opt_beta

        opt_result = ParamResult([opt_beta, opt_gamma], time() - start, self, opt_energy)
        opt_result["num_points"] = num_points
        opt_result["parameter_ranges"] = parameter_ranges

        return opt_result

    def plot(self, axis: Optional[plt.Axes] = None, fig: Optional[plt.Figure] = None):
        """Make a plot of the training.

        If giving the `axis` then the figure object must also be given. Otherwise,
        you get new objects.

        Args:
            axis: Axis on which to plot.
            fig: The figure object.
        """

        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        ggs, bbs = np.meshgrid(self._gammas, self._betas)
        cset = axis.contourf(ggs, bbs, self._energies, levels=30)
        axis.set_xlabel("Gamma")
        axis.set_ylabel("Beta")
        axis.scatter([self._opt_gamma], [self._opt_beta], s=10, marker="*", color="w")
        fig.colorbar(cset, ax=axis, label="Energy")

        return fig, axis

    @classmethod
    def from_config(cls, config: dict) -> "DepthOneScanTrainer":
        """Create an intance from a config."""

        evaluator_cls = EVALUATORS[config["evaluator"]]
        evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(evaluator)

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse the trainig arguments.

        These are given in the form:
        num_points_low_high_low_high_...
        For instance training with 20 points from 0 to 2pi is given as
        20_0_6.283185_0_6.283185.
        """
        if args_str is None:
            return dict()

        split_args = args_str.split("_")
        num_points = int(split_args[0])

        param_ranges = []
        for idx in range(1, len(split_args), 2):
            low = float(split_args[idx])
            high = float(split_args[idx + 1])
            param_ranges.append((low, high))

        return {
            "num_points": num_points,
            "parameter_ranges": param_ranges,
        }
