#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A mix-in class for trainers that track/optimize energy and its history."""

from typing import List


class HistoryMixin:
    """A small mix-in class to trakc enery history.

    This can be used in e.g. ScipyTrainer and other trainers that
    perform an optimization of the parameters.
    """

    def __init__(self):
        """Initialize the history mixin."""

        # Energy history is saved internally at each optimization for plotting.
        self._energy_history = []

        # Parameter history is saved internally
        self._parameter_history = []

        # Duration of each energy evaluation is saved internally
        self._energy_evaluation_time = []

    @property
    def energy_history(self) -> List[float]:
        """Return the energy history of the last optimization run."""
        return self._energy_history

    @property
    def parameter_history(self) -> List[List[float]]:
        """Return the parameter history of the last optimization run."""
        return self._parameter_history

    @property
    def energy_evaluation_time(self) -> List[float]:
        """Return the duration of each energy evaluation in the last optimization run."""
        return self._energy_evaluation_time

    def reset_history(self):
        """Reset the history by clearing the internal lists."""
        self._energy_history = []
        self._parameter_history = []
        self._energy_evaluation_time = []
