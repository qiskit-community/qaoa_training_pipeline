#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class to generate random initial points."""

from typing import Any, Dict, Optional
from time import time
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qaoa_training_pipeline.training.base_trainer import BaseTrainer


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

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        reps: int,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        seed: Optional[int] = None,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> Dict[str, Any]:
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
        start = time()

        lb_ = lower_bound or self._lower_bound
        ub_ = upper_bound or self._upper_bound

        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = self._rng

        params = [float(val) for val in rng.uniform(lb_, ub_, 2 * reps)]

        return {
            "optimized_params": params,
            "note": f"The parameters are randomly generated with uniform and seed {self._seed}.",
            "train_duration": time() - start,
            "Energy": "NA",
        }

    @classmethod
    def from_config(cls, config: dict) -> "RandomPoint":
        """Create a random initial point generator from a config."""
        return cls(**config)

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Parse arguments for the train method from a string.

        Args:
            args_str: A string of the form reps_low_high_seed where low and high are the
                lower and upper bounds on the parameters respectively. seed can be an
                int or the stirng "None".
        """
        if args_str is None:
            return dict()

        args = args_str.split("_")

        if args[3].lower() == "none":
            seed = None
        else:
            seed = int(args[3])

        return {
            "reps": int(args[0]),
            "lower_bound": float(args[1]),
            "upper_bound": float(args[2]),
            "seed": seed,
        }
