#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to create instances of known problem classes from input."""

from qaoa_training_pipeline.utils.data_utils import load_input, input_to_operator


class MaxCut:
    """Produce max-cut operators from input graphs."""

    @classmethod
    def from_str(cls, input_str: str) -> "MaxCut":
        """Create the class from a string."""
        return cls()

    def cost_operator(self, input):
        """Create the cost operator from the input.
        
        The (weighted) edges of the graph are converted to ZZ with a prefoactor of -0.5.
        """
        return input_to_operator(input, pre_factor=-0.5)


class MaxIndependentSet:
    """Produce max independent set operators from input graphs."""

    def __init__(self, penalty: float = 2.0):
        """Create the maximum independent set class."""
        self._penalty = penalty

    @classmethod
    def from_str(cls, input_str: Optional[str] = "") -> "MaxIndependentSet":
        """Create the class from a string."""
        penalty = float(input_str) 
        return cls()

    def cost_operator(self, input):
        """Create the cost operator from the input."""


PROBLEM_CLASSES = {
    "MaxCut": MaxCut,
    "MaxIndependentSet": MaxIndependentSet,
}
