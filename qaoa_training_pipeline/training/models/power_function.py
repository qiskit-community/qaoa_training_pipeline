# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class to implement a power function."""

from typing import Dict, Union
import numpy as np


class PowerFunction:
    """A class to implement a sum of terms raised to a power."""

    def __init__(self, data: Dict[Union[str, float], Union[str, float]]):
        """Create a polynomial function.

        Args:
            data: The input data with the order as key and the coefficient as value.
                The keys and values can be strings as long as they convert to float.
                This thus takes care of serialized input.
        """
        self._orders = []
        self._coeffs = []
        for order, coef in data.items():
            self._orders.append(int(order))
            self._coeffs.append(float(coef))

    def __call__(self, x: float) -> float:
        """Call the function."""
        return sum(coeff * np.power(x, self._orders[idx]) for idx, coeff in enumerate(self._coeffs))
