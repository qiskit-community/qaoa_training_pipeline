"""Recursive transition states trainer."""

from time import time
from typing import Dict, Optional
import matplotlib.pyplot as plt


from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.exceptions import TrainingError
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.transition_states import TransitionStatesTrainer
from qaoa_training_pipeline.training.param_result import ParamResult


class RecursiveTransitionStates(BaseTrainer):
    """Recursively train QAOA by constructing transition states.
    
    This class uses an initial set of parameters for depth `p` QAOA to construct transition states 
    at depth `p+1`, from which the optimized parameters are used to construct the transition 
    states at the next depth `p+2`. This process continues until the specified depth is reached.
    """


    def __init__(self, trainer: BaseTrainer):  # Here the trainer would be e.g. a SciPy trainers
        super().__init__(trainer.evaluator)

        self._trainer = trainer
        self._all_results = None

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return self._trainer.minimization

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        previous_optimal_point: list[float],
        reps: int,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> ParamResult:
        """
        Args:
            cost_op: The cost operator of the problem we want to solve.
            previous_optimal_point: A local minima in beta and gamma from which to start 
                the transition states recursion.
            reps: The number of QAOA layers we want to reach.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit that represents the initial state. If None is
                given, then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit.

        Returns:
            A dictionary with optimization results.
        """
        start = time()
        current_reps = len(previous_optimal_point) // 2
        ts_state = previous_optimal_point
        self._all_results, energy = dict(), None

        while current_reps < reps:
            ts_trainer = TransitionStatesTrainer(self._trainer)
            result = ts_trainer.train(cost_op, ts_state, mixer, initial_state, ansatz_circuit)
            ts_state = result["optimized_params"]
            energy = result["energy"]
            self._all_results[current_reps] = result.data
            current_reps = len(ts_state) // 2

        param_result = ParamResult(ts_state, time() - start, self, energy)
        param_result.update(self._all_results)

        return param_result

    @classmethod
    def from_config(cls, config: Dict) -> "RecursiveTransitionStates":
        """Create the trainer from a config file."""

        if config["trainer"] != "ScipyTrainer":
            raise TrainingError(
                f"{cls.__name__} only uses ScipyTrainer as trainer. Received " + config["trainer"]
            )

        trainer = ScipyTrainer.from_config(config["trainer_init"])

        return cls(trainer)

    def to_config(self) -> Dict:
        """Return the configuration of the trainer."""
        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": self._evaluator.to_config(),
            "trainer": self._trainer.to_config(),
        }

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse a string into the training kwargs."""
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key == "reps":
                train_kwargs[key] = int(val)
            elif key == "previous_optimal_point":
                train_kwargs[key] = self.extract_list(val, dtype=float)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        **plot_args,
    ):
        """Plot the energy progression throughout the recursion."""
        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        rts_idx = sorted(self._all_results.keys())
        energies = [self._all_results[key]["energy"] for key in rts_idx]

        axis.plot(rts_idx, energies, label="Energy", **plot_args)
        axis.set_xlabel("Recursion level")
        axis.set_ylabel("Energy")
        axis.legend()

        return fig, axis
