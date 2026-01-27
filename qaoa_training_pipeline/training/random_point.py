#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class to generate random initial points."""

from time import time
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.param_result import ParamResult


class RandomPoint(BaseTrainer):
    """Generate random initial points for QAOA.

    This class generates random initial points for QAOA. Strictly speaking, this
    class does not do any training. However, we wrap the BaseTrainer to incorporate
    random initial point generation in our training pipeline. I.e., this class
    could be used to generate a random initial point that is then picked-up by
    the next trainer in the chain.
    """

    def __init__(
        self,
        lower_bound: float = 0,
        upper_bound: float = np.pi,
        seed: Optional[int] = None,
    ) -> None:
        """Setup an instance to generate random initial points.

        Args:
            lower_bound: A lower bound for the randomly generated parameters. This
                value defaults to 0.
            upper_bound: An upper bound for the randomly generated parameters. This
                value defaults to pi.
            seed: Optional argument. If given this sets the seed of the rng generator.
        """
        super().__init__(None)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

    @property
    def minimization(self):
        """Raises a warning as a random point neither minimizes nor maximizes."""
        raise ValueError(f"{self.__class__.__name__} neither minimizes nor maximizes.")

    # pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        seed: Optional[int] = None,
        reps: int | None = None,
    ) -> ParamResult:
        """Return a random initial point.

        Args:
            cost_op: The cost operator. This argument is not used.
            reps: The number of QAOA repetitions. This argument is used.
            lower_bound: The lower bound for the value of the random initial point. If this
                argument is not given we default to the value set at initialization.
            upper_bound: The upper bound for the value of the random initial point. If this
                argument is not given we default to the value set at initialization.
            seed: The seed for the random number generator. If no seed is given
                the random number generator set at initialization is used. If a seed is
                given then we use a random number generator initialized with this seed. Note
                the the rng defined at initialization is not changed.
            mixer: Not used.
            initial_state: Not used.
            ansatz_circuit: Not used.
        """
        if reps is None:
            raise ValueError(f"class {self.__class__.__name__} requires reps to be specified")
        start = time()

        lb_ = lower_bound or self._lower_bound
        ub_ = upper_bound or self._upper_bound

        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = self._rng

        params = [float(val) for val in rng.uniform(lb_, ub_, 2 * reps)]

        param_result = ParamResult(params, time() - start, self, None)
        param_result["note"] = f"The parameters are uniformly generated with seed {self._seed}."

        return param_result

    @classmethod
    def from_config(cls, config: dict) -> "RandomPoint":
        """Create a random initial point generator from a config."""
        return cls(**config)

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse arguments for the train method from a string.

        Args:
            args_str: A string of keyword arguments of the form `k1:v1:k2:v2`.
                The possible keywords are `reps`, `seed`, `lower_bound`, and `upper_bound`.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key in ["reps", "seed"]:
                train_kwargs[key] = int(val)
            elif key in ["lower_bound", "upper_bound"]:
                train_kwargs[key] = float(val)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs

    def to_config(self) -> dict:
        """Creates a serializable dictionary to keep track of how results are created.

        Note: This data structure is not intended for us to recreate the class instance.
        """
        return {
            "trainer_name": self.__class__.__name__,
            "evaluator": "NA",
        }
