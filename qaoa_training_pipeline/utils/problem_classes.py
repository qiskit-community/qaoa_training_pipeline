#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to create cost operators of known problem classes from input."""

from typing import Optional

from qiskit_optimization.applications import StableSet
from qiskit_optimization.converters import QuadraticProgramToQubo

from qaoa_training_pipeline.utils.data_utils import load_input, input_to_operator
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph


class MaxCut:
    """Produce max-cut operators from input graphs."""

    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "MaxCut":
        """Create the class from a string."""
        return cls()

    def cost_operator(self, input_data: dict):
        """Create the cost operator from the input.
        
        The (weighted) edges of the graph are converted to ZZ with a prefoactor of -0.5.

        Args:
            input: A dict specifying a graph where the keys are tuples of ints representing 
                edges or hyper-edge and the values are the weights.
        """
        return input_to_operator(input_data, pre_factor=-0.5)


class MaxIndependentSet:
    """Produce max independent set operators from input graphs."""

    DEFAULT_PENALTY = 2.0

    def __init__(self, penalty: Optional[float] = None):
        """Create the maximum independent set class."""
        self._penalty = penalty or self.DEFAULT_PENALTY

    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "MaxIndependentSet":
        """Create the class from a string."""
        try:
            penalty = float(input_str)
        except ValueError:
            penalty = cls.DEFAULT_PENALTY

        return cls(penalty)

    def cost_operator(self, input_data: dict):
        """Create the cost operator from the input graph given as dict.
        
        Args:
            input_data: A dict specifying a graph where the keys are tuples of ints representing 
                edges or hyper-edge and the values are the weights.
        """
        graph = dict_to_graph(input_data)
        qp_ = StableSet(graph).to_quadratic_program()
        cost_op, _ = QuadraticProgramToQubo(penalty=self._penalty).convert(qp_).to_ising()

        return cost_op


PROBLEM_CLASSES = {
    "maxcut": MaxCut,
    "mis": MaxIndependentSet,
}
