#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to recursively train QAOA parameters."""

from time import time
from typing import Any, Callable, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.exceptions import TrainingError
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer
from qaoa_training_pipeline.training.parameter_extenders import PARAMETEREXTENDERS


class RecursionTrainer(BaseTrainer):
    """Recursively train QAOA by initializing level `p+1` with level `p`.

    This class uses a function, called the `parameter_extender` to ingest parameters
    for depth `p` QAOA and return an initial set of parameters for depth `p+1`. These
    new parameters are then given as initial point to a ScipyTrainer to further optimize
    them.
    """

    def __init__(self, trainer: ScipyTrainer, parameter_extender: Callable = None):
        """Initialize a recursion trainer.

        Args:
            parameter_extender: This is a callable that takes a list of floats as input
                and returns a longer list of floats as output. The input are the
                optimized parameters values at depth `p` and the output are the initial
                points for the parameter optimization at depth `p+1`.
            trainer: The trainer must be the ScipyTrainer.
        """
        super().__init__(trainer.evaluator)

        # Takes parameters from QAOA depth p to depth p+1.
        self._parameter_extender = parameter_extender or PARAMETEREXTENDERS["extend"]

        # The trainer that optimizes the extended parameters.
        self._trainer = trainer

    @property
    def minimization(self) -> bool:
        """Return True if the energy is minimized."""
        return self._trainer.minimization

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        params0: list[float],
        reps: int,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> Dict[str, Any]:
        """Perform the training.

        Args:
            cost_op: The cost operator for which to train the parameters.
            params0: The initial point from which to train. This initial point should correspond
                to a set of QAOA parameters that is smaller than the desired QAOA depth `reps`.
            reps: The target QAOA depth. Must be greather than the reps implied by params0.
            mixer: The mixer operator is passed on to the Scipy trainer.
            initial_sate: The initial state is passed on to the Scipy trainer.
            ansatz_circuit: The ansatz circuit is passed on to the Scipy trainer.
        """
        start = time()
        current_reps = len(params0) // 2

        all_results, energy = dict(), None

        while current_reps < reps:
            current_reps += 1

            # Generate new parameters by extending the parameter vector.
            new_params0 = self._parameter_extender(params0)

            depth = len(new_params0) // 2

            if current_reps != depth:
                raise TrainingError(
                    f"The depth of {depth} returned by the parameter extender "
                    f"does not match the expected depth of {2*current_reps}. "
                )

            result = self._trainer.train(
                cost_op,
                params0=new_params0,
                mixer=mixer,
                initial_state=initial_state,
                ansatz_circuit=ansatz_circuit,
            )

            params0 = result["optimized_params"]
            energy = result["energy"]
            all_results[current_reps] = result

        all_results["train_duration"] = time() - start
        all_results["optimized_params"] = params0
        all_results["trainer"] = self.to_config()
        all_results["energy"] = energy

        return all_results

    @classmethod
    def from_config(cls, config: Dict) -> "RecursionTrainer":
        """Create the trainer from a config file.

        The parameter extender is chosen from one of the parameter extenders
        specified in parameter_extenders.py.
        """

        if config["trainer"] != "ScipyTrainer":
            raise TrainingError(
                f"{cls.__name__} only uses ScipyTrainer as trainer. Received " + config["trainer"]
            )

        trainer = ScipyTrainer.from_config(config["trainer_init"])
        parameter_extender = PARAMETEREXTENDERS[config["parameter_extender"]]

        return cls(trainer, parameter_extender)

    def to_config(self) -> Dict:
        """Return the configuration of the trainer."""
        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": self._evaluator.to_config(),
            "trainer": self._trainer.to_config(),
            "parameter_extender": self._parameter_extender.__name__,
        }

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse a string into the training kwargs.

        Note: This trainer does not take any arguments aside from the default
        ones. We therefore ask the sub-trainer to parse the key word arguments.
        """
        if args_str is None:
            return dict()

        return self._trainer.parse_train_kwargs(args_str)
